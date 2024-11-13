# -*- encoding: utf-8 -*-
"""
@Time    :   2024-11-13 21:32:02
@desc    :   server
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""


import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse

sys.path.append(Path(Path(__file__).parent).as_posix())
# sys.path.append(Path(Path(__file__).parent, "src").as_posix())
from _base._tokenize import process_tokenize
from _types._chunker import *
from _types._tokenize import TokenizeRequest

load_dotenv()

app = FastAPI(title="Simple Baseline RAG APIs", version="0.0.1", docs_url="/docs", redoc_url="/redoc")


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/hello")
async def _hello():
    return {"message": "Hello World"}


@app.post("/text/tokenize")
async def _text_tokenize(request: TokenizeRequest):
    tokens, token_ids = await process_tokenize(request)
    return JSONResponse(content={"tokens": tokens, "token_ids": token_ids})


@app.post("/text/chunk/token")
async def _text_chunk_token(request: TokenChunkerRequest) -> ChunkerResponse:
    from _trunker._token_chunker import _chunked

    response = await _chunked(request)
    return response


@app.post("/text/chunk/word")
async def _text_chunk_word(request: WordChunkerRequest) -> ChunkerResponse:
    from _trunker._word_chunker import _chunked

    response = await _chunked(request)
    return response


@app.post("/text/chunk/sentence")
async def _text_chunk_sentence(request: SentenceChunkerRequest) -> ChunkerResponse:
    from _trunker._sentence_chunker import _chunked

    response = await _chunked(request)
    return response

@app.post("/text/chunk/semantic")
async def _text_chunk_semantic(request: SentenceChunkerRequest) -> ChunkerResponse:
    from _trunker._semantic_chunker import _chunked

    response = await _chunked(request)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
