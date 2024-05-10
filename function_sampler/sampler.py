"""
Tool Call sampler.

Yes I know how messy this code is. I'll clean it up when I get the chance.
"""

import functools
import time
from typing import (
    Any,
    Dict,
    List,
    Union
)

from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import LogitsProcessor, PreTrainedTokenizer

from .config import TokenMap
from .config.config import ToolCallSamplerConfig
from .fsm import (
    FSMState, 
    FsmTokenizer
)

from .logger import get_logger

from .utils import (
    build_masks,
    bundle_sampling,
    tokenize_dicts,
    compute_fsm
)

logger = get_logger()


class ToolCallSampler(LogitsProcessor):
    """
    A logits processor designed to facilitate the generation and sampling of function calls and their arguments.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        functions: List[Dict] = None,
        config: Union[ToolCallSamplerConfig, Dict[str, Any]] = None,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.functions = functions or []
        self.config = self._parse_config(config, kwargs)
        self._initialize_tokens()
        self.vocab_size = len(tokenizer)
        self.token_masks = self._build_token_masks()
        self.function_maps = tokenize_dicts(self.functions, self.tokenizer)
        self.fsm_results = {}
        self.executor = ThreadPoolExecutor()
        self._compute_all_fsms()
        self._reset_sampling_state()
        self._set_sampling_params()

    def _parse_config(self, config, kwargs):
        if isinstance(config, dict) or config is None:
            return ToolCallSamplerConfig(**config or {}, **kwargs)
        elif isinstance(config, ToolCallSamplerConfig):
            return config
        else:
            raise ValueError("config must be a ToolCallSamplerConfig instance or a dictionary")

    def _initialize_tokens(self):
        config = self.config
        self.open_func_token = config.open_func_token or "<tool_call>"
        self.close_func_token = config.close_func_token or "</tool_call>"
        self.generate_close_func_token = config.generate_close_func_token
        self.end_on_function_call = config.end_on_function_call
        self.open_func_token_length = len(self.tokenizer.encode(self.open_func_token, add_special_tokens=False))

    def _build_token_masks(self):
        json_tokens = self.config.json_tokens.model_dump() if self.config.json_tokens else TokenMap.build(self.tokenizer).model_dump()
        token_masks = {}
        build_masks(self.tokenizer, token_masks, json_tokens, self.vocab_size)
        return token_masks

    def _compute_all_fsms(self):
        for key, function in self.function_maps.items():
            future = self.executor.submit(compute_fsm, FsmTokenizer(self.tokenizer), function)
            future.add_done_callback(functools.partial(self._populate_fsm_result, key=key))

    def _populate_fsm_result(self, future, key):
        self.fsm_results[key] = future.result()

    def _reset_sampling_state(self):
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

    def _set_sampling_params(self):
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.top_k = self.config.top_k
        self.repetition_penalty = self.config.repetition_penalty

    def __del__(self):
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
        """
        Wait for the FSM result associated with `key` to be populated.
        really only needed to prevent a key error if somehow the fsm isnt returned yet. 
        """
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
        inputs_string = self.tokenizer.decode(input_ids[-self.open_func_token_length:])
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
                            ## With the new rust backend, we shouldnt ever need to actually do this. 
                            ## but it cant really hurt, as it probably accounts for edge cases I cant think of right now.
                            self.fsm = self._wait_for_fsm_result(fsm_key)
                            self.first_fsm_token = True
                        else:
                            self.fsm = self.fsm_results[fsm_key]
                            self.fsm_state = FSMState(0)
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
        return scores
