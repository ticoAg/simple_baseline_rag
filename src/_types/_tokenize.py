# -*- encoding: utf-8 -*-
"""
@Time    :   2024-11-13 22:06:45
@desc    :   tokenize model
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

from pydantic import BaseModel, Field


class TokenizeRequest(BaseModel):
    text: str = Field(..., description="文本")
    tokenizer: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="tokenizer名称", examples=["gpt2", "Qwen/Qwen2.5-0.5B-Instruct"])
