from pydantic import BaseModel, Field
from typing import List
from transformers import PreTrainedTokenizer
from .utils import *
# Ensure the necessary token finding functions are imported or defined here

class TokenMap(BaseModel):
    """
    Default token mappings for Mistral, Mixtral, and derivatives.
    """
    open_bracket: List[int] = []
    close_bracket: List[int] = []
    comma: List[int] = []
    quote: List[int] = []
    quote_comma: List[int] = []
    eov: List[int] = []
    quote_banned: List[int] = []
    list_open: List[int] = []
    list_close: List[int] = []
    int: List[int] = []
    float: List[int] = []
    colon: List[int] = []
    space: List[int] = []
    name: List[int] = []

    @classmethod
    def build(cls, tokenizer: PreTrainedTokenizer):
        instance = cls(
            open_bracket=find_variant_tokens(tokenizer, '{'),
            close_bracket=cls._calc_fn_tokens('}', 'a', tokenizer),
            comma=find_variant_tokens(tokenizer, ','),
            quote=cls._calc_fn_tokens('"', 'a', tokenizer),
            quote_comma=cls._calc_fn_tokens('",', 'a', tokenizer),
            eov=cls._calc_fn_tokens('",', 'a', tokenizer) + cls._calc_fn_tokens(',', 'a', tokenizer),
            quote_banned=find_tokens_with_char(tokenizer, ["'", '"']),
            list_open=find_variant_tokens(tokenizer, '['),
            list_close=find_variant_tokens(tokenizer, ']'),
            int=get_int_tokens(tokenizer),
            float=get_float_tokens(tokenizer),
            colon=find_variant_tokens(tokenizer, ':'),
            space=tokenizer.encode(" ", add_special_tokens=False) +
                  tokenizer.encode("  ", add_special_tokens=False) +
                  [tokenizer.encode("> ", add_special_tokens=False)[1]] if len(tokenizer.encode("> ", add_special_tokens=False)) == 2 else [],
            name=tokenizer.encode("name", add_special_tokens=False)
        )
        return instance
    


class MistralTokenMap(TokenMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    