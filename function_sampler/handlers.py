from .logger import get_logger
import torch
logger = get_logger()

def apply_string(self, function_tokens, mask) -> torch.LongTensor:
    logger.debug("# identified val of type string")
    logger.debug("latest function token: " + str(function_tokens[-1]))
    logger.debug("quote token")
    if self.val_started:
        logger.debug("# got to generating value")
        # generate string val now, no else.
        self.val_started = True
        mask = self._disable_tokens(token_types=["quote_banned"])
        if not self.required_completed:
            mask = self._allow_tokens(token_types=["eov"], mask=mask)

    else:
        mask[self.json_tokens["quote"][0]] = True
        self.val_started = True

    if self.all_args_completed and self.val_started:
        mask = self._disable_tokens(token_types=["quote_comma"], mask=mask)
        mask = self._allow_tokens(token_types=["quote"], mask=mask)
    return mask

def apply_integer(self, function_tokens, mask) -> torch.LongTensor:
    logger.debug("# identified val of type integer")
    logger.debug("latest function token: " + str(function_tokens[-1]))
    self.val_started = True

    if not self.required_completed:
        mask = self._allow_tokens(token_types=["comma"], mask=mask)
    mask = self._allow_tokens(token_types=["integer_tokens"], mask=mask)
    return mask

def apply_float(self, function_tokens, mask) -> torch.LongTensor:
    logger.debug("# identified val of type float")
    logger.debug("latest function token: " + str(function_tokens[-1]))
    self.val_started = True

    if not self.required_completed:
        mask = self._allow_tokens(token_types=["comma"], mask=mask)
    mask = self._allow_tokens(token_types=["float_tokens"], mask=mask)
    return mask

def apply_constraints(self, arg_key, arg_info, function_tokens, mask):
    match arg_info["type"].lower():
        case "string":
            mask = apply_string(self, function_tokens, mask)
        case "integer":
            mask = apply_integer(self, function_tokens, mask)
        case "float":
            mask = apply_float(self, function_tokens, mask)
    
    return mask