# -*- encoding: utf-8 -*-
"""
@Time    :   2024-11-13 22:40:40
@desc    :   
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

from chonkie import TokenChunker
from fastapi import HTTPException

from _types._chunker import ChunkerResponse, TokenChunkerRequest


async def _chunked(request: TokenChunkerRequest) -> ChunkerResponse:
    from _base._tokenize import load_tokenizer

    tokenizer = await load_tokenizer(request.tokenizer)
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap)
    if not (request.text or request.texts):
        raise HTTPException(status_code=400, detail="text or texts is required")
    elif request.text:
        chunks = chunker.chunk(request.text)
    else:
        chunks = [chunker.chunk(text) for text in request.texts]
    return ChunkerResponse(chunks=chunks)
