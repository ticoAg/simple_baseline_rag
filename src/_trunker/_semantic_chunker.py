# -*- encoding: utf-8 -*-
"""
@Time    :   2024-11-13 22:40:40
@desc    :   
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""


from typing import List, Union

import numpy as np
from chonkie import SemanticChunker
from fastapi import HTTPException
from openai import OpenAI

from _types._chunker import ChunkerResponse, SemanticChunkerRequest


class SemanticChunkerEmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI()

    def encode(self, text: str):
        """
        embeddings = self.embedding_model.encode(raw_sentences, convert_to_numpy=True)
        """
        result = self.client.embeddings.create(input=text, model=self.model_name)
        return np.array(result.data[0].embedding)

    def similarity(self, embedding1: Union[List, np.ndarray], embedding2: Union[List, np.ndarray]):
        """Compute cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


async def _chunked(request: SemanticChunkerRequest) -> ChunkerResponse:
    from _base._tokenize import load_tokenizer

    tokenizer = await load_tokenizer(request.tokenizer)
    embedding_model = SemanticChunkerEmbeddingModel(request.embedding_model)
    chunker = SemanticChunker(
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        max_chunk_size=request.max_chunk_size,
        similarity_threshold=request.similarity_threshold,
    )
    if not (request.text or request.texts):
        raise HTTPException(status_code=400, detail="text or texts is required")
    elif request.text:
        chunks = chunker.chunk(request.text)
    else:
        chunks = [chunker.chunk(text) for text in request.texts]
    return ChunkerResponse(chunks=chunks)
