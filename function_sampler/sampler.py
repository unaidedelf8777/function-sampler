import functools
import time
from json import dumps as json_dumps
from typing import Any, Dict, List, Union

import torch
from transformers import LogitsProcessor, PreTrainedTokenizer

from .config import TokenMap
from .config.config import ToolCallSamplerConfig
from .config.utils import calc_fn_tokens
from .fsm import FSMState, FsmTokenizer, RegexFSM
from .json import build_regex_from_schema
from .logger import get_logger
from .utils import build_masks, bundle_sampling, tokenize_dicts

logger = get_logger()


class ToolCallSampler(LogitsProcessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        functions: List[Dict] = None,
        config: Union[ToolCallSamplerConfig, Dict[str, Any]] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.functions = functions

        # If config is a dictionary or None, parse it with Pydantic model
        if isinstance(config, dict) or config is None:
            config = ToolCallSamplerConfig(**config or {}, **kwargs)
        elif not isinstance(config, ToolCallSamplerConfig):
            raise ValueError(
                "config must be a ToolCallSamplerConfig instance or a dictionary"
            )

        self.config = config

        self.open_func_token = (
            self.config.open_func_token if self.config.open_func_token else "<function>"
        )
        self.close_func_token = (
            self.config.open_func_token
            if self.config.open_func_token
            else "</function>"
        )

        self.open_func_token_length = len(
            self.tokenizer.encode(self.open_func_token, add_special_tokens=False)
        )

        logger.debug(self.open_func_token)
        logger.debug(self.close_func_token)

        self.end_on_function_call = self.config.end_on_function_call

        self.vocab_size = len(tokenizer)

        self.json_tokens = (
            self.config.json_tokens.model_dump()
            if self.config.json_tokens
            else TokenMap.build(tokenizer=tokenizer).model_dump()
        )

        #
        # convert json_tokens dict into a dict with values of long tensors, instead of allowed token ids
        self.token_masks = {}
        build_masks(
            tokenizer=self.tokenizer,
            token_masks=self.token_masks,
            json_tokens=self.json_tokens,
            vocab_size=self.vocab_size,
        )

        # sampling flags and misc
        self.identified_function = None

        self.last_open_key_quote_index = -1

        self.function_maps = tokenize_dicts(self.functions, self.tokenizer)

        self.fsm = None
        self.fsm_state = FSMState(0)
        self.fsm_seq_start_idx = None
        self.generation_finished = False
        self.do_sample = False
        self.last_open_quote_idx = -1
        self.first_fsm_token = False
        # Sampling params. these are only used when generating values for params / args.
        # when not generating a value, they are ignored.
        self.temperature = config.temperature if config.temperature else None
        self.top_p = config.top_p if config.top_p else None
        self.top_k = config.top_k if config.top_k else None
        self.repetition_penalty = (
            config.repetition_penalty if config.repetition_penalty else None
        )

    def _determine_function(self, start_sequence):
        # Convert the start_sequence list to a tuple for comparison
        start_tuple = tuple(start_sequence)

        # Find all key-value pairs where the key starts with start_tuple
        matching_items = {
            key: value
            for key, value in self.function_maps.items()
            if key[: len(start_tuple)] == start_tuple
        }
        logger.debug(matching_items)
        if matching_items:
            return matching_items
        else:
            # Return None if no matching items are found
            return None

    def _allow_tokens(
        self, token_types: List[str] = [], token_ids: List[int] = [], mask=None
    ):
        """
        Returns a mask, initialized with False, with the specified token types and token IDs allowed.
        """
        new_mask = torch.full((self.vocab_size,), False, dtype=torch.bool)

        new_mask = functools.reduce(
            torch.logical_or,
            [
                self.token_masks[t]
                for t in token_types
                if self.token_masks[t] is not None
            ],
            new_mask,
        )

        for id in token_ids:
            new_mask[id] = True

        if mask is not None:
            mask = torch.logical_or(mask, new_mask)
        else:
            mask = new_mask

        return mask

    def _disable_tokens(
        self, token_types: List[str] = [], token_ids: List[int] = None, mask=None
    ):
        """
        Returns a mask, initialized with True, with the specified token types and token IDs disabled.
        """
        new_mask = torch.full((self.vocab_size,), True, dtype=torch.bool)

        for token_type in token_types:
            if token_type in self.token_masks:
                new_mask = torch.logical_and(
                    new_mask, torch.logical_not(self.token_masks[token_type])
                )

        if token_ids is not None:
            new_mask[token_ids] = False

        if mask is not None:
            mask = torch.logical_and(mask, new_mask)
        else:
            mask = new_mask

        return mask

    def _collect_key_tuples(self, input_dict):
        # Initialize an empty list to store tuples of (key, value)
        key_value_pairs = []

        # Iterate over the items of the input dictionary
        for key, value in input_dict.items():
            # Check if the key is a tuple
            if isinstance(key, tuple):
                # Add the (key, value) pair as a tuple to the list
                key_value_pairs.append((key, value))

        return key_value_pairs

    def _check_for_function_call(self, input_ids):
        input_ids = input_ids[0]
        inputs_string = self.tokenizer.decode(input_ids[-15:])

        if inputs_string.strip().endswith(self.open_func_token):
            return True
        else:
            return False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        start_time = time.time()
        if self._check_for_function_call(input_ids):
            function_tokens = []
        else:
            function_tokens = False

        mask = torch.zeros(self.vocab_size, dtype=torch.bool)

        if function_tokens is not False:
            tokens_len = len(function_tokens)

            # check if a bracket has been made yet
            if tokens_len == 0:
                next_tokens = self.tokenizer.encode(
                    '{"name": "', add_special_tokens=False
                )
                self.next_tokens = next_tokens
                self.last_open_quote_idx = len(next_tokens)
                mask = self._allow_tokens(token_types=["space"], mask=mask)

            elif len(self.next_tokens) >= 1:
                tok_id = self.next_tokens.pop(0)
                mask = self._allow_tokens(token_ids=[tok_id], mask=mask)

            elif len(self.next_tokens) == 0:
                if self.identified_function is None:
                    current_val_seq = function_tokens[self.last_open_quote_idx + 1 :]
                    possible_next_tokens = self._determine_function(current_val_seq)

                    # did we identify a function?
                    if len(possible_next_tokens) > 1:
                        # didn't identify a function *yet*
                        tuples = self._collect_key_tuples(possible_next_tokens)
                        next_token_ids = set()
                        for seq in tuples:
                            list_seq = list(seq[0])
                            next_token_ids.update([list_seq[len(current_val_seq)]])
                        self.do_sample = True
                        next_token_ids = list(next_token_ids)
                        mask = self._allow_tokens(token_ids=next_token_ids, mask=mask)

                    elif len(possible_next_tokens) == 1:
                        t = self._collect_key_tuples(possible_next_tokens)
                        self.identified_function = t[0][1]
                        tokens = list(t[0][0])
                        # no reason to sample one token.
                        self.do_sample = False
                        self.next_tokens = (
                            tokens[len(current_val_seq) + 1 :]
                            + self.tokenizer.encode(
                                '", "arguments": ', add_special_tokens=False
                            )[1:]
                        )
                        next_token = tokens[len(current_val_seq)]
                        self.nesting_level = 2
                        self.fsm_seq_start_idx = len(self.next_tokens) + tokens_len
                        mask = self._allow_tokens(token_ids=[next_token], mask=mask)

                elif self.identified_function is not None:
                    if self.fsm is None:
                        regex = build_regex_from_schema(
                            json_dumps(self.identified_function)
                        )
                        self.fsm = RegexFSM(
                            regex, FsmTokenizer(tokenizer=self.tokenizer)
                        )
                        self.first_fsm_token = True

                    if self.first_fsm_token:
                        allowed_tokens = self.fsm.allowed_token_ids(self.fsm_state)
                        self.first_fsm_token = False
                    else:
                        last_token = input_ids[0][-1]
                        self.fsm_state = self.fsm.next_state(
                            self.fsm_state, int(last_token)
                        )
                        allowed_tokens = self.fsm.allowed_token_ids(self.fsm_state)

                    if self.fsm.is_final_state(self.fsm_state):
                        mask = self._allow_tokens(token_types=["close_bracket"])
                        self.next_tokens = (
                            [self.json_tokens["close_bracket"][0]]
                            + self.close_func_token
                            + [self.tokenizer.eos_token_id]
                        )
                    else:
                        mask = self._allow_tokens(token_ids=allowed_tokens)
                        self.do_sample = True

            mask = mask.expand_as(scores)
            scores[~mask] = -float("Inf")

            if self.do_sample:
                bundle_sampling(
                    scores,
                    input_ids,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                )

            logger.debug("#### Time for iteration: " + str(time.time() - start_time))
        return scores
