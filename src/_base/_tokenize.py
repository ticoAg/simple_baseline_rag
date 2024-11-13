# -*- encoding: utf-8 -*-
"""
@Time    :   2024-11-13 22:02:45
@desc    :   tokenize
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""
from cachetools import TTLCache
from fastapi import HTTPException
from transformers import AutoTokenizer

from _types._tokenize import TokenizeRequest

# 设置缓存，TTL 为 3600 秒（1 小时）
tokenizer_cache = TTLCache(maxsize=100, ttl=3600)


async def load_tokenizer(tokenizer_name: str):
    tokenizer = tokenizer_cache.get(tokenizer_name)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            tokenizer_cache[tokenizer_name] = tokenizer
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load tokenizer {tokenizer_name}: {str(e)}")
    return tokenizer


async def process_tokenize(request: TokenizeRequest):
    tokenizer_name = request.tokenizer
    text = request.text

    tokenizer = await load_tokenizer(tokenizer_name)

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(tokens)
    return tokens, token_ids
