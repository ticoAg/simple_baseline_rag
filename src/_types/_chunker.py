# -*- encoding: utf-8 -*-
"""
@Time    :   2024-11-13 22:45:15
@desc    :   
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

from typing import List, Literal, Union

from chonkie import Chunk
from pydantic import BaseModel, Field


class TokenChunkerRequest(BaseModel):
    """
    The TokenChunker splits text into chunks based on token count.

    Attributes:
        text (str): The text that needs to be split into chunks.
        tokenizer (str): The name of the tokenizer to use. Default is "Qwen/Qwen2.5-0.5B-Instruct".
        chunk_size (int): The size of each chunk. Default is 1024 tokens.
        chunk_overlap (int): The number of tokens to overlap between chunks. Default is 128 tokens.
    """

    text: str = Field(None, description="text to split.")
    texts: List[str] = Field(None, description="texts to split.")
    tokenizer: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="tokenizer name.")
    chunk_size: int = Field(1024, description="chunk size.")
    chunk_overlap: int = Field(128, description="overlap between chunks")


class ChunkerResponse(BaseModel):
    chunks: Union[List[Chunk], List[List[Chunk]]]


class WordChunkerRequest(BaseModel):
    """
    The TokenChunker splits text into chunks based on token count.

    Attributes:
        text (str): The text that needs to be split into chunks.
        tokenizer (str): The name of the tokenizer to use. Default is "Qwen/Qwen2.5-0.5B-Instruct".
        chunk_size (int): The size of each chunk. Default is 1024 tokens.
        chunk_overlap (int): The number of tokens to overlap between chunks. Default is 128 tokens.
    """

    text: str = Field(None, description="text to split.")
    texts: List[str] = Field(None, description="texts to split.")
    tokenizer: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="tokenizer name.")
    chunk_size: int = Field(1024, description="chunk size.")
    chunk_overlap: int = Field(128, description="overlap between chunks")
    mode: Literal["simple", "advanced"] = Field(
        "simple", description="simple: basic space-based splitting, advanced: handles punctuation and special cases"
    )


class SentenceChunkerRequest(BaseModel):
    """The SentenceChunker preserves sentence boundaries."""

    text: str = Field(None, description="text to split.")
    texts: List[str] = Field(None, description="texts to split.")
    tokenizer: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="tokenizer name.")
    chunk_size: int = Field(1024, description="chunk size.")
    chunk_overlap: int = Field(128, description="overlap between chunks")
    min_sentences_per_chunk: int = Field(1, description="minimum number of sentences per chunk")


class SemanticChunkerRequest(BaseModel):
    """
    The SemanticChunker groups content by semantic similarity. The implementation is inspired by the semantic chunking approach described in the FullStackRetrieval Tutorials, with modifications and optimizations for better performance and integration with Chonkie's architecture.

    This version of SemanticChunker has some optimizations that speed it up considerably, but make the assumption that the tokenizer you used is the same as the one used for embedding_model. This is a valid assumption since most often than not, chunk_size and hence, token_count is dependent on the embedding_model context sizes rather than on the Generative models context length.
    """

    text: str = Field(None, description="text to split.")
    texts: List[str] = Field(None, description="texts to split.")
    tokenizer: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="tokenizer name.")
    max_chunk_size: int = Field(1024, description="chunk size.")
    similarity_threshold: float = Field(0.7, description="similarity threshold")
    embedding_model: str = Field(
        "netease-youdao/bce-embedding-base_v1",
        description="embedding model",
        examples=["netease-youdao/bce-embedding-base_v1", "BAAI/bge-m3", "BAAI/bge-large-en-v1.5", "BAAI/bge-large-zh-v1.5"],
    )
