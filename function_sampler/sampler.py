import functools
import time
from typing import Any, Dict, List, Union
from concurrent.futures import ProcessPoolExecutor

import torch
from transformers import LogitsProcessor, PreTrainedTokenizer

from .config import TokenMap
from .config.config import ToolCallSamplerConfig
from .fsm import FSMState, FsmTokenizer
from .logger import get_logger
from .utils import build_masks, bundle_sampling, tokenize_dicts, compute_fsm

logger = get_logger()


class ToolCallSampler(LogitsProcessor):
    """
    A logits processor designed to facilitate the generation and sampling of function calls and their arguments.

    Attributes:
        tokenizer (PreTrainedTokenizer): A tokenizer compatible with Hugging Face's Transformers library,
            used for encoding and decoding text.
        functions (List[Dict], optional): A list of dictionaries representing the available functions
            and their metadata. Defaults to None.
        config (Union[ToolCallSamplerConfig, Dict[str, Any]], optional): Configuration for the sampler,
            either as a ToolCallSamplerConfig object or as a dictionary that can be parsed into one.
            Defaults to None.

    The class is initialized with a tokenizer for handling text encoding/decoding, a list of function
    definitions for determining valid function calls and arguments, and a configuration object for
    fine-tuning the sampling behavior. It extends the `LogitsProcessor` class from Hugging Face's
    Transformers, enabling it to be integrated into the text generation pipeline to control the
    likelihood of generating specific tokens based on the current context and predefined constraints.
    """

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
        self.generate_close_func_token = (
            config.generate_close_func_token
            if config.generate_close_func_token
            else True
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

        # we launch computation of all FSM's at the begining,
        # if one is needed before it is finished, we block until it is done.
        # otherwise, it should be ready by the time we need it.
        self.fsm_results = {}
        self.executor = ProcessPoolExecutor()

        self.fsm_tokenizer = FsmTokenizer(tokenizer)

        for key, function in self.function_maps.items():
            future = self.executor.submit(compute_fsm, self.fsm_tokenizer, function)
            future.add_done_callback(
                functools.partial(self.populate_fsm_result, key=key)
            )

        self.next_tokens = []
        self.fsm = None
        self.fsm_state = FSMState(0)
        self.fsm_seq_start_idx = None
        self.function_key = None
        self.generation_finished = False
        self.do_sample = False
        self.last_open_quote_idx = -1
        self.first_fsm_token = False
        self.input_ids_split_idx = None
        self.total_time = 0

        # Sampling params. these are only used when generating values for params / args.
        # when not generating a value, they are ignored.
        self.temperature = config.temperature if config.temperature else None
        self.top_p = config.top_p if config.top_p else None
        self.top_k = config.top_k if config.top_k else None
        self.repetition_penalty = (
            config.repetition_penalty if config.repetition_penalty else None
        )

    def populate_fsm_result(self, future, key):
        # Callback function to populate the results dict upon future completion
        self.fsm_results[key] = future.result()

    def __del__(self):
        # Ensure executor resources are freed when the instance is destroyed
        self.executor.shutdown(wait=False)

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

    def _wait_for_fsm_result(self, key, timeout=None):
        """Wait for the FSM result associated with `key` to be populated."""
        start_time = time.time()
        while key not in self.fsm_results:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Waiting for FSM result for '{key}' timed out.")
            time.sleep(0.1)  # Sleep to prevent busy waiting
        return self.fsm_results[key]

    def _allow_tokens(
        self, token_types: List[str] = [], token_ids: List[int] = [], mask=None
    ):
        """
        Returns a mask, initialized with False, with the specified token types and token IDs allowed.
        """
        if mask is None:
            mask = torch.full((self.vocab_size,), False, dtype=torch.bool)

        if token_types:
            # Pre-compute a combined mask for all specified token types
            combined_type_mask = torch.zeros_like(mask)
            for t in token_types:
                if t in self.token_masks:
                    combined_type_mask |= self.token_masks[t]
            mask |= combined_type_mask

        if token_ids:
            # Directly update the mask for specified token IDs
            mask[torch.tensor(token_ids, dtype=torch.long)] = True

        return mask

    def _disable_tokens(
        self, token_types: List[str] = [], token_ids: List[int] = None, mask=None
    ):
        """
        Returns a mask, initialized with True, with the specified token types and token IDs disabled.
        """
        if mask is None:
            mask = torch.full((self.vocab_size,), True, dtype=torch.bool)

        if token_types:
            # Pre-compute a combined negated mask for all specified token types
            combined_type_mask = torch.ones_like(mask)
            for t in token_types:
                if t in self.token_masks:
                    combined_type_mask &= ~self.token_masks[t]
            mask &= combined_type_mask

        if token_ids is not None:
            # Directly update the mask to disable specified token IDs
            mask[torch.tensor(token_ids, dtype=torch.long)] = False

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

    def _decode(self, input_ids):
        return self.tokenizer.decode(input_ids)

    def _encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def _check_for_function_call(self, input_ids):
        input_ids = input_ids[0]
        inputs_string = self.tokenizer.decode(input_ids[-self.open_func_token_length :])
        if inputs_string.strip().endswith(self.open_func_token):
            return True
        else:
            return False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        start_time = time.time()
        if self.input_ids_split_idx is not None:
            function_tokens = input_ids[0][self.input_ids_split_idx :]
        else:
            if self._check_for_function_call(input_ids):
                function_tokens = []
                self.input_ids_split_idx = len(input_ids[0])
            else:
                function_tokens = False

        mask = torch.zeros(self.vocab_size, dtype=torch.bool)

        if function_tokens is not False:
            tokens_len = len(function_tokens)

            # check if a bracket has been made yet
            if tokens_len == 0:
                next_tokens = self._encode('{"name": "')
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
                        self.function_key = t[0][0]
                        tokens = list(t[0][0])
                        # no reason to sample one token.
                        self.do_sample = False
                        self.next_tokens = (
                            tokens[len(current_val_seq) + 1 :]
                            + self._encode('", "arguments": ')[1:]
                        )
                        next_token = tokens[len(current_val_seq)]
                        self.nesting_level = 2
                        self.fsm_seq_start_idx = len(self.next_tokens) + tokens_len
                        mask = self._allow_tokens(token_ids=[next_token], mask=mask)

                elif self.identified_function is not None:
                    if self.fsm is None:
                        fsm_key = self.function_key
                        if fsm_key not in self.fsm_results:
                            self.fsm = self._wait_for_fsm_result(fsm_key)
                            self.first_fsm_token = True
                        else:
                            self.fsm = self.fsm_results[fsm_key]
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

                    if allowed_tokens == [-2]:
                        mask = self._allow_tokens(token_types=["close_bracket"])
                        self.next_tokens = (
                            self.tokenizer.encode(
                                self.close_func_token, add_special_tokens=False
                            )
                            + [self.tokenizer.eos_token_id]
                            if self.generate_close_func_token
                            else [self.tokenizer.eos_token_id]
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

            t = time.time() - start_time
            self.total_time += t
            logger.debug("#### Time for iteration: " + str(time.time() - start_time))
        return scores
