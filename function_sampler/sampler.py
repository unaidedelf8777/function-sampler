import torch
from typing import List, Dict, Union
from transformers import LogitsProcessor, PreTrainedTokenizer
import unicodedata
import logging
import time
from .config.config import ToolCallSamplerConfig
from .config import TokenMap
import functools

from .utils import build_masks, tokenize_dicts, bundle_sampling
from .config.utils import calc_fn_tokens
from .handlers import apply_constraints
from .logger import get_logger


logger = get_logger()


class ToolCallSampler(LogitsProcessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        functions: List[Dict] = None,
        config: ToolCallSamplerConfig = None,
    ):
        self.tokenizer = tokenizer
        self.functions = functions
        self.config = config or ToolCallSamplerConfig()

        self.open_func_token = self.config.open_func_token or calc_fn_tokens(
            token="<function>", ctx_token="a"
        )
        self.close_func_token = self.config.close_func_token or calc_fn_tokens(
            token="</function>", ctx_token="a"
        )

        logger.debug(self.open_func_token)
        logger.debug(self.close_func_token)

        self.end_on_function_call = config.end_on_function_call or True

        self.vocab_size = len(tokenizer)

        self.json_tokens = (
            config.json_tokens.dict()
            if config.json_tokens
            else TokenMap.build(tokenizer=tokenizer).dict()
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
        self.current_call_stage = None  # either 'fn_name' or 'arguments'
        self.identified_function = None
        # tells what tokens to ban. for instance, if it is type 'int' or 'float', then we dont allow non digits.
        self.arg_type = None
        self.completed_args = []
        self.last_open_key_quote_index = -1
        self.val_started = False
        self.val_first_val_token = False
        self.required_completed = False
        self.all_args_completed = False

        self.function_maps = tokenize_dicts(self.functions, self.tokenizer)
        self.nesting_level = 1

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

    def _is_required_completed(self):
        if self.identified_function is None:
            self.required_completed = False
            return False
        if self.completed_args == self.identified_function["parameters"]["required"]:
            self.required_completed = True
            return True
        elif self.arg_type:
            if (
                self.completed_args + list(self.arg_type[0])
                == self.identified_function["parameters"]["required"]
            ):
                self.required_completed = True
                return True
        else:
            self.required_completed = False
            return False

    def _is_all_args_completed(self):
        if self.identified_function is None:
            self.all_args_completed = False
            return False
        elif self.arg_type:
            if (
                self.completed_args + list(self.arg_type[0])
                == self.identified_function["parameters"]["properties"].keys()
            ):
                self.all_args_completed = True
                return True
        else:
            self.all_args_completed = False
            return False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        start_time = time.time()
        function_tokens = input_ids[0][46:]

        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        self._is_required_completed()
        self._is_all_args_completed()

        if function_tokens is not False:
            tokens_len = len(function_tokens)

            # check if a bracket has been made yet
            if tokens_len == 0:
                next_tokens = self.tokenizer.encode(
                    '{"name": "', add_special_tokens=False
                )
                self.next_tokens = next_tokens
                self.last_open_quote_idx = len(next_tokens)
                mask = self._allow_tokens(token_ids=[28705], mask=mask)

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
                        next_token_ids = list(next_token_ids)
                        mask = self._allow_tokens(token_ids=next_token_ids, mask=mask)

                    elif len(possible_next_tokens) == 1:
                        t = self._collect_key_tuples(possible_next_tokens)
                        self.identified_function = t[0][1]
                        tokens = list(t[0][0])
                        self.next_tokens = (
                            tokens[len(current_val_seq) + 1 :]
                            + self.tokenizer.encode(
                                '", "arguments": {"', add_special_tokens=False
                            )[1:]
                        )
                        next_token = tokens[len(current_val_seq)]
                        self.nesting_level = 2
                        self.last_open_key_quote_index = (
                            len(self.next_tokens) + tokens_len
                        )
                        mask = self._allow_tokens(token_ids=[next_token], mask=mask)

                elif self.identified_function is not None:
                    logger.debug("LOG: Identified function")
                    required_args = [
                        item
                        for item in self.identified_function["parameters"]["required"]
                        if item not in self.completed_args
                    ]
                    optional_args = [
                        item
                        for item in self.identified_function["parameters"][
                            "properties"
                        ].keys()
                        if item not in required_args and item not in self.completed_args
                    ]
                    current_key_seq = function_tokens[
                        self.last_open_key_quote_index + 1 :
                    ]
                    logger.debug(self.tokenizer.decode(current_key_seq))
                    current_key_seq = list(current_key_seq)
                    logger.debug("Current SEQ: " + str(current_key_seq))

                    # if we are already generating a arg, then we must not start another one.
                    # if we are ready for the next arg, then self.arg_type will be None
                    if self.arg_type is not None:
                        logger.debug("#### Entered value sampling stage ####")

                        if (
                            function_tokens[-1] in self.json_tokens["eov"]
                            and self.val_started
                        ):
                            logger.debug("## Value identified as finished ##")
                            if (
                                self.arg_type == "string"
                                and function_tokens[-1] in self.json_tokens["comma"]
                            ):
                                logger.debug("# Nevermind. ##")
                            else:
                                self.completed_args.append(self.arg_type[0])
                                self.arg_type = None
                                self.val_started = False
                                next_token_ids = [28705]
                                self.next_tokens = [self.json_tokens["quote"][0]]
                                self.last_open_key_quote_index = (
                                    len(function_tokens) + 1
                                )
                                logger.debug(
                                    "Required completed" + str(self.required_completed)
                                )
                                logger.debug(
                                    "all args completed? "
                                    + str(self.all_args_completed)
                                )
                                if self.required_completed:
                                    if self.all_args_completed:
                                        # must close now
                                        next_token_ids = self.json_tokens[
                                            "close_bracket"
                                        ]
                                        self.next_tokens = [
                                            self.close_func_token,
                                            self.tokenizer.eos_token_id,
                                        ]
                                    # the model is allowed to not give more params
                                    else:
                                        next_token_ids.append(
                                            self.json_tokens["close_bracket"]
                                        )
                                mask = self._allow_tokens(
                                    token_ids=next_token_ids, mask=mask
                                )

                        else:
                            arg_key, arg_info = self.arg_type
                            logger.debug(
                                "# value not finished. searching for correct case #"
                            )
                            mask = apply_constraints(
                                self, arg_key, arg_info, function_tokens, mask
                            )

                    elif len(required_args) >= 1:
                        logger.debug("required args not fulfilled")
                        key_tuple = tuple(current_key_seq)
                        logger.debug("Key tuple: " + str(key_tuple))
                        logger.debug(required_args)
                        possible_keys = [
                            key
                            for key in required_args
                            if len(current_key_seq) == 0
                            or list(key[: len(current_key_seq)]) == current_key_seq
                        ]
                        logger.debug("Possible keys: " + str(possible_keys))

                        if len(possible_keys) > 1:
                            logger.debug("possible keys available")

                            next_token_ids = set()
                            key_seq_len = len(current_key_seq)
                            for seq in possible_keys:
                                next_token_ids.update([seq[key_seq_len]])
                            mask = self._allow_tokens(
                                token_ids=list(next_token_ids), mask=mask
                            )

                        elif len(possible_keys) == 1:
                            self.arg_type = (
                                possible_keys[0],
                                self.identified_function["parameters"]["properties"][
                                    possible_keys[0]
                                ],
                            )

                            remaining_tokens = list(
                                possible_keys[0][len(current_key_seq) :]
                            )

                            if len(remaining_tokens) > 0:
                                next_token_id = remaining_tokens.pop(0)
                                remaining_tokens += [28705]  # space token
                                self.next_tokens = remaining_tokens
                            elif len(remaining_tokens) == 0:
                                next_token_id = 28705  # space token
                            # Update mask to allow the next token
                            mask = self._allow_tokens(
                                token_ids=[next_token_id], mask=mask
                            )

                    elif len(required_args) == 0 and len(optional_args) >= 1:
                        logger.debug("required args are fulfilled")
                        key_tuple = tuple(current_key_seq)
                        logger.debug("Key tuple: " + str(key_tuple))
                        logger.debug(required_args)
                        possible_keys = [
                            key
                            for key in optional_args
                            if len(current_key_seq) == 0
                            or list(key[: len(current_key_seq)]) == current_key_seq
                        ]
                        logger.debug("Possible keys: " + str(possible_keys))

                        if len(possible_keys) > 1:
                            logger.debug("possible keys available")

                            next_token_ids = set()
                            key_seq_len = len(current_key_seq)
                            for seq in possible_keys:
                                next_token_ids.update([seq[key_seq_len]])
                            mask = self._allow_tokens(
                                token_ids=list(next_token_ids), mask=mask
                            )

                        elif len(possible_keys) == 1:
                            self.arg_type = (
                                possible_keys[0],
                                self.identified_function["parameters"]["properties"][
                                    possible_keys[0]
                                ],
                            )

                            remaining_tokens = list(
                                possible_keys[0][len(current_key_seq) :]
                            )

                            if len(remaining_tokens) > 0:
                                next_token_id = remaining_tokens.pop(0)
                                remaining_tokens += [28705]  # space token
                                self.next_tokens = remaining_tokens
                            elif len(remaining_tokens) == 0:
                                next_token_id = 28705  # space token
                            # Update mask to allow the next token
                            mask = self._allow_tokens(
                                token_ids=[next_token_id], mask=mask
                            )

                    else:
                        mask = self._allow_tokens(
                            token_types=["close_bracket"], mask=mask
                        )
                        self.next_tokens = (
                            []
                            + self.tokenizer.encode(
                                "</function>", add_special_tokens=False
                            )
                            + [self.tokenizer.eos_token_id]
                        )

            mask = mask.expand_as(scores)
            scores[~mask] = -float("Inf")

            if self.val_started:
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
