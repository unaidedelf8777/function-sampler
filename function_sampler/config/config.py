from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from .token_mapper import TokenMap


class ToolCallSamplerConfig(BaseModel):
    open_func_token: Optional[List[int]] = None
    close_func_token: Optional[List[int]] = None
    json_tokens: Optional[TokenMap] = None
    vocab_size: Optional[int]
    end_on_function_call: Optional[bool] = True
    top_p: Optional[float] = Field(None, ge=0, le=1)
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_k: Optional[float] = Field(None, ge=0, le=1)
    repetition_penalty: Optional[float] = Field(None, ge=0, le=2)
