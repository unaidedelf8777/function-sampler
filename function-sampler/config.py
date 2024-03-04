from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from config.token_mapper import TokenMap

class ToolCallSamplerConfig(BaseModel):
    open_func_token: Optional[List[int]] = None
    close_func_token: Optional[List[int]] = None
    json_tokens: Optional[TokenMap] = TokenMap.default()
    vocab_size: Optional[int]
    end_on_function_call: Optional[bool] = True
    top_p: Field(..., ge=0, le=1)
    temperature: Field(..., ge=0, le=2)
    top_k: Field(..., ge=0, le=1)
    repetition_penalty: Field(..., ge=0, le=2)

