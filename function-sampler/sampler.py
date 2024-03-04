from .config.token_map import TokenMap
from .logger import get_logger
from transformers import LogitsProcessor

logger = get_logger()


class ToolCallSampler(LogitsProcessor):


    def __init__(self, tokenizer: PreTrainedTokenizer, functions: List[Dict] = None, config: ToolCallSamplerConfig = None, debug: bool = False):
        self.logger = logging.getLogger(__name__)
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        if config and config.open_func_token is not None and config.close_func_token is not None:
            self.open_func_token = config.open_func_token
            self.close_func_token = config.close_func_token
        else:
            # Compute the tokens if not provided in config
            self.open_func_token = self._calc_fn_tokens(token="<function>", ctx_token="a", tokenizer=tokenizer)
            self.close_func_token = self._calc_fn_tokens(token="</function>", ctx_token="a", tokenizer=tokenizer)
            self.logger.debug(self.open_func_token)
            self.logger.debug(self.close_func_token)

        if config and config.end_on_function_call:
          self.end_on_function_call = config.end_on_function_call
        else:
          # Default: True
          self.end_on_function_call = True

        self.tokenizer = tokenizer
        self.functions = functions

        self.vocab_size = 32002

        # Use the json_tokens from config if available, else compute
        if config and config.json_tokens:
            self.json_tokens = config.json_tokens
        else:
            self.json_tokens = config.json_tokens.build()


        #
        # convert json_tokens dict into a dict with values of long tensors, instead of allowed token ids
        self.token_masks = {}
        self._build_masks()

        # sampling flags and misc
        self.current_call_stage = None # either 'fn_name' or 'arguments'
        self.identified_function = None
        self.arg_type = None # tells what tokens to ban. for instance, if it is type 'int' or 'float', then we dont allow non digits.
        self.completed_args = []
        self.last_open_key_quote_index = -1
        self.val_started = False
        self.val_first_val_token = False
        self.required_completed = False
        self.all_args_completed = False

        self.function_maps = self.tokenize_dicts(self.functions)
        self.nesting_level = 1



    def _calc_fn_tokens(self, token, ctx_token, tokenizer, nested=False):

      ctx_tok = tokenizer.encode(ctx_token, add_special_tokens=False)

      with_ctx = tokenizer.encode(ctx_token + token, add_special_tokens=False)

      without_ctx = tokenizer.encode(token, add_special_tokens=False)

      final = set()

      # are they the same length?
      if len(with_ctx) == len(without_ctx):
        final.append(without_ctx)
      else:
        if len(ctx_tok) == 1:
          with_ctx.remove(ctx_tok[0])
          if nested:
            final.update([with_ctx])
            final.update([without_ctx])
          else:
            final.update(with_ctx)

      return list(final)



    def _build_masks(self):
      bad_tokens = []
      if self.tokenizer.eos_token_id:
        bad_tokens.append(self.tokenizer.eos_token_id)
      if self.tokenizer.bos_token_id:
        bad_tokens.append(self.tokenizer.bos_token_id)
      if self.tokenizer.unk_token_id:
        bad_tokens.append(self.tokenizer.unk_token_id)

      for key, token_indexes in self.json_tokens.items():
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for index in token_indexes:
            if index not in bad_tokens:
              mask[index] = True
        self.token_masks[key] = mask



    def _handle_enum(self, options: List[str], scores: torch.FloatTensor) -> torch.FloatTensor:

        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for i in options:
          toks = tokenizer.encode(i, add_special_tokens=False)
          for t in toks:
            mask[t] = True
        mask = mask.expand_as(scores)
        scores[~mask] = -float("inf")

        return scores


    def _determine_function(self, start_sequence):
      # Convert the start_sequence list to a tuple for comparison
      start_tuple = tuple(start_sequence)

      # Find all key-value pairs where the key starts with start_tuple
      matching_items = {key: value for key, value in self.function_maps.items() if key[:len(start_tuple)] == start_tuple}
      self.logger.debug(matching_items)
      if matching_items:
        return matching_items
      else:
          # Return None if no matching items are found
          return None




    def find_tokens_after_unclosed_function(self, input_ids: torch.Tensor) -> List[int]:

      # Initialize counters and positions
      start_count = 0
      end_count = 0
      last_open_pos = None  # Track the position of the last <function> token

      # Function to search for a sequence in the input_ids
      def search_sequence(input_ids, sequence):
          for i in range(len(input_ids) - len(sequence) + 1):
              if input_ids[i:i+len(sequence)] == sequence:
                  return i  # Return the starting index of the sequence
          return -1


      index = 0
      while index < len(input_ids):
          open_pos = search_sequence(input_ids[index:], self.open_func_token)
          if open_pos != -1:

              start_count += 1
              last_open_pos = index + open_pos  # Update the position of the last <function> token
              index += open_pos + len(self.open_func_token)
          else:
              close_pos = search_sequence(input_ids[index:], self.close_func_token)
              if close_pos != -1:
                  end_count += 1
                  index += close_pos + len(self.close_func_token)
              else:
                  index += 1

      # If there is an unclosed <function> token, return the tokens after the last <function>
      if start_count > end_count and last_open_pos is not None:
          # Return the remaining tokens after the last unclosed <function> token
          return input_ids[last_open_pos + len(self.open_func_token):]
      else:
          # Return an empty list or similar if no unclosed <function> token is found
          return False


    def _is_arg_finished(self, tokens):
      """
      is the arg finished and ready to be parsed?
      use method sparingly, as detokenizing and checking is relatively slow
      """
      string = self.tokenizer.decode(tokens)

      if string.strip().endswith('",'):
        return True
      else:
        return False

    def _allow_tokens(self, token_types: List[str] = [], token_ids: List[int] = None, mask=None):
        """
        Returns a mask, initialized with False, with the specified token types and token IDs allowed.
        """
        new_mask = torch.full((self.vocab_size,), False, dtype=torch.bool)

        for token_type in token_types:
            if token_type in self.token_masks:
                new_mask = torch.logical_or(new_mask, self.token_masks[token_type])

        if token_ids is not None:
          for id in token_ids:
            new_mask[id] = True

        if mask is not None:
            mask = torch.logical_or(mask, new_mask)
        else:
            mask = new_mask

        return mask

    def _disable_tokens(self, token_types: List[str] = [], token_ids: List[int] = None, mask=None):
        """
        Returns a mask, initialized with True, with the specified token types and token IDs disabled.
        """
        new_mask = torch.full((self.vocab_size,), True, dtype=torch.bool)

        for token_type in token_types:
            if token_type in self.token_masks:
                new_mask = torch.logical_and(new_mask, torch.logical_not(self.token_masks[token_type]))

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
      if self.completed_args == self.identified_function['parameters']['required']:
        self.required_completed = True
        return True
      elif self.arg_type:
        if self.completed_args + list(self.arg_type[0]) == self.identified_function['parameters']['required']:
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
        if self.completed_args + list(self.arg_type[0]) == self.identified_function['parameters']['properties'].keys():
          self.all_args_completed = True
          return True
      else:
        self.all_args_completed = False
        return False



    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
      start_time = time.time()
      function_tokens = input_ids[0][46:]

      mask = torch.zeros(self.vocab_size, dtype=torch.bool)
      self._is_required_completed()
      self._is_all_args_completed()

      if function_tokens is not False:

        tokens_len = len(function_tokens)

        # check if a bracket has been made yet
        if tokens_len == 0:
          next_tokens = self.tokenizer.encode('{"name": "', add_special_tokens=False)
          self.next_tokens = next_tokens
          self.last_open_quote_idx = len(next_tokens)
          mask = self._allow_tokens(token_ids = [28705], mask=mask)

        elif len(self.next_tokens) >= 1:
          tok_id = self.next_tokens.pop(0)
          mask = self._allow_tokens(token_ids=[tok_id], mask=mask)

        elif len(self.next_tokens) == 0:
          if self.identified_function is None:
            current_val_seq = function_tokens[self.last_open_quote_idx + 1:]
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
              self.next_tokens = tokens[len(current_val_seq) + 1:] + self.tokenizer.encode('", "arguments": {"', add_special_tokens=False)[1:]
              next_token = tokens[len(current_val_seq)]
              self.nesting_level = 2
              self.last_open_key_quote_index = len(self.next_tokens) + tokens_len
              mask = self._allow_tokens(token_ids=[next_token], mask=mask)


          elif self.identified_function is not None:
            self.logger.debug("LOG: Identified function is not None")
            required_args = [item for item in self.identified_function['parameters']['required'] if item not in self.completed_args]
            optional_args = [item for item in self.identified_function['parameters']['properties'].keys() if item not in required_args and item not in self.completed_args]
            current_key_seq = function_tokens[self.last_open_key_quote_index + 1:]
            self.logger.debug(self.tokenizer.decode(current_key_seq))
            current_key_seq = list(current_key_seq)
            self.logger.debug("Current SEQ: " + str(current_key_seq))
            num_args_remaining = len(optional_args) + len(required_args)


            # if we are already generating a arg, then we must not start another one.
            # if we are ready for the next arg, then self.arg_type will be None
            if self.arg_type is not None:
              self.logger.debug("#### Entered value sampling stage ####")

              if function_tokens[-1] in self.json_tokens['eov'] and self.val_started:
                self.logger.debug("## Value identified as finished ##")
                if self.arg_type == 'string' and function_tokens[-1] in self.json_tokens['comma']:
                  self.logger.debug("# Nevermind. ##")
                else:
                  self.completed_args.append(self.arg_type[0])
                  self.arg_type = None
                  self.val_started = False
                  next_token_ids = [28705]
                  self.next_tokens = [self.json_tokens['quote'][0]]
                  self.last_open_key_quote_index = len(function_tokens) + 1
                  self.logger.debug("Required completed" + str(self.required_completed))
                  self.logger.debug("all args completed? " + str(self.all_args_completed))
                  if self.required_completed:
                    if self.all_args_completed:
                      # must close now
                      next_token_ids = self.json_tokens['close_bracket']
                      self.next_tokens = [self.close_func_token, self.tokenizer.eos_token_id]
                    # the model is allowed to not give more params
                    else:
                      next_token_ids.append(self.json_tokens['close_bracket'])
                  mask = self._allow_tokens(token_ids = next_token_ids, mask=mask)

              else:
                arg_key, arg_info = self.arg_type
                self.logger.debug("# value not finished. searching for correct case #")
                match arg_info['type'].lower():


                  case 'string':
                    self.logger.debug("# identified val of type string")
                    self.logger.debug("latest function token: " + str(function_tokens[-1]))
                    self.logger.debug("quote token")
                    if self.val_started:
                      self.logger.debug("# got to generating value")
                      # generate string val now, no else.
                      self.val_started = True
                      mask = self._disable_tokens(token_types=['quote_banned'])
                      if not self.required_completed:
                        mask = self._allow_tokens(token_types=['eov'], mask=mask)


                    else:
                      mask[self.json_tokens['quote'][0]] = True
                      self.val_started = True

                    if self.all_args_completed and self.val_started:
                      self.logger.debug("ONLY CLOSING QUOTE AVAILABLE i think")
                      mask = self._disable_tokens(token_types=['quote_comma'], mask=mask)
                      mask = self._allow_tokens(token_types=['quote'], mask=mask)

                  case 'integer':
                    self.logger.debug("# identified val of type string")
                    self.logger.debug("latest function token: " + str(function_tokens[-1]))
                    self.logger.debug("quote token")
                    self.val_started = True

                    if not self.required_completed:
                      mask = self._allow_tokens(token_types=['comma'], mask=mask)
                    mask = self._allow_tokens(token_types=['int'], mask=mask)
                  
                  case 'float':
                    self.logger.debug("# identified val of type string")
                    self.logger.debug("latest function token: " + str(function_tokens[-1]))
                    self.logger.debug("quote token")
                    self.val_started = True

                    if not self.required_completed:
                      mask = self._allow_tokens(token_types=['comma'], mask=mask)
                    mask = self._allow_tokens(token_types=['float'], mask=mask)




            if len(required_args) >= 1:
              self.logger.debug("required args not fulfilled")
              key_tuple = tuple(current_key_seq)
              self.logger.debug("Key tuple: " + str(key_tuple))
              self.logger.debug(required_args)
              possible_keys = [key for key in required_args if len(current_key_seq) == 0 or list(key[:len(current_key_seq)]) == current_key_seq]
              self.logger.debug("Possible keys: " + str(possible_keys))

              if len(possible_keys) > 1:
                self.logger.debug("possible keys available")

                next_token_ids = set()
                key_seq_len = len(current_key_seq)
                for seq in possible_keys:
                  next_token_ids.update([seq[key_seq_len]])
                mask = self._allow_tokens(token_ids = list(next_token_ids), mask=mask)

              elif len(possible_keys) == 1:
                self.arg_type = (possible_keys[0], self.identified_function['parameters']['properties'][possible_keys[0]])

                remaining_tokens = list(possible_keys[0][len(current_key_seq):])

                if len(remaining_tokens) > 0:
                  next_token_id = remaining_tokens.pop(0)
                  remaining_tokens += [28705] # space token
                  self.next_tokens = remaining_tokens
                elif len(remaining_tokens) == 0:
                  next_token_id = 28705 # space token
                # Update mask to allow the next token
                mask = self._allow_tokens(token_ids=[next_token_id], mask=mask)


            elif len(required_args) == 0 and len(optional_args) >= 1:
              self.logger.debug("required args are fulfilled")
              key_tuple = tuple(current_key_seq)
              self.logger.debug("Key tuple: " + str(key_tuple))
              self.logger.debug(required_args)
              possible_keys = [key for key in optional_args if len(current_key_seq) == 0 or list(key[:len(current_key_seq)]) == current_key_seq]
              self.logger.debug("Possible keys: " + str(possible_keys))

              if len(possible_keys) > 1:
                self.logger.debug("possible keys available")

                next_token_ids = set()
                key_seq_len = len(current_key_seq)
                for seq in possible_keys:
                  next_token_ids.update([seq[key_seq_len]])
                mask = self._allow_tokens(token_ids = list(next_token_ids), mask=mask)

              elif len(possible_keys) == 1:
                self.arg_type = (possible_keys[0], self.identified_function['parameters']['properties'][possible_keys[0]])

                remaining_tokens = list(possible_keys[0][len(current_key_seq):])

                if len(remaining_tokens) > 0:
                  next_token_id = remaining_tokens.pop(0)
                  remaining_tokens += [28705] # space token
                  self.next_tokens = remaining_tokens
                elif len(remaining_tokens) == 0:
                  next_token_id = 28705 # space token
                # Update mask to allow the next token
                mask = self._allow_tokens(token_ids=[next_token_id], mask=mask)

            else:
              mask = self._allow_tokens(token_types=['close_bracket'], mask=mask)
              self.next_tokens = [] + self.tokenizer.encode("</function>", add_special_tokens=False) + [self.tokenizer.eos_token_id]




        mask = mask.expand_as(scores)
        scores[~mask] = -float("Inf")
        self.logger.debug("#### Time for iteration: " + str(time.time() - start_time))
      return scores