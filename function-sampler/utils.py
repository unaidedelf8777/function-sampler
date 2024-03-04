

def collect_key_tuples(self, input_dict):
      # Initialize an empty list to store tuples of (key, value)
      key_value_pairs = []

      # Iterate over the items of the input dictionary
      for key, value in input_dict.items():
          # Check if the key is a tuple
          if isinstance(key, tuple):
              # Add the (key, value) pair as a tuple to the list
              key_value_pairs.append((key, value))

      return key_value_pairs


 def tokenize_dicts(self, input_dicts,tokenizer: PreTrainedTokenizer, exempt_keys=["required", "name", "type", "enum", "parameters"], root=True):
          """
          Tokenize keys of dictionaries using a specified tokenizer,
          excluding keys listed in exempt_keys. Specifically handles 'required' list elements.

          :param input_dicts: A list of dictionaries to process.
          :param exempt_keys: A list of keys to exclude from tokenization.
          :return: A new dictionary with tokenized keys where applicable.
          """

          tokenized_dicts = {}
          for input_dict in input_dicts:
              tokenized_dict = {}

              for key, value in input_dict.items():
                  if key in exempt_keys:
                    if key == 'parameters':
                      # Tokenize parameter names and their descriptions within 'parameters'
                      tokenized_params = {}
                      for param_name, param_info in value['properties'].items():
                          # Tokenize the parameter name
                          param_name_tokens = tuple(tokenizer.encode('"' + param_name + '":', add_special_tokens=False)[1:])
                          # You might also want to tokenize certain values, like descriptions
                          # Here, we're copying the param_info as is, but you could tokenize certain fields as needed
                          tokenized_params[param_name_tokens] = param_info
                      required = tokenized_dict[key] = [tuple(self.tokenizer.encode('"' + v + '":', add_special_tokens=False)[1:]) for v in value['required']]
                      tokenized_dict['parameters'] = {'type': 'object', 'properties': tokenized_params, 'required': required}

                    else:
                      tokenized_dict[key] = value
                  else:
                      # Recursively process nested dictionaries
                      if isinstance(value, dict):
                          tokenized_dict[key] = self.tokenize_dicts([value], exempt_keys, root=False)
                      else:
                          # Tokenizing the key
                          tokenized_key = tuple(self.tokenizer.encode('"' + key + '":', add_special_tokens=False)[1:])
                          tokenized_dict[tokenized_key] = value
              if root:
                # Tokenizing the 'name' key for the main dictionary key
                name_str = '"' + input_dict['name'] + '",'
                name_str = name_str.strip()
                # Tokenizers are wierd. we need to encode it for how its final representation ( after the model "generates" it ), otherwise decoding wont work. so if a single quote aint a thing for you, your fucked :)
                name_tokens = self.tokenizer.encode(name_str, add_special_tokens=False)[1:]
                self.logger.debug(self.tokenizer.decode(name_tokens))
                name_tuple = tuple(name_tokens)
                tokenized_dicts[name_tuple] = tokenized_dict
              else:
                return tokenized_dict

          return tokenized_dicts


def build_masks(tokenizer, json_tokens):
      bad_tokens = []
      if tokenizer.eos_token_id:
        bad_tokens.append(tokenizer.eos_token_id)
      if tokenizer.bos_token_id:
        bad_tokens.append(tokenizer.bos_token_id)
      if tokenizer.unk_token_id:
        bad_tokens.append(tokenizer.unk_token_id)

      for key, token_indexes in json_tokens.items():
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for index in token_indexes:
            if index not in bad_tokens:
              mask[index] = True
        self.token_masks[key] = mask