"""layout-rag: layout-aware PDF chunking and hybrid retrieval, CPU-only, no LLM in the indexing path."""
from .chunker import Chunker, TokenCounter
from .config import ChunkerConfig
from .pipeline import ChunkResult, DocLayoutChunker
from .retrieve import BM25Index, CrossEncoderReranker, Hit, Retriever
from .store import MemoryStore, MilvusStore
from .types import Block, Chunk, Section, Word

__version__ = "0.1.0"
__all__ = ["BM25Index", "Block", "Chunk", "ChunkResult", "Chunker", "ChunkerConfig", "CrossEncoderReranker",
           "DocLayoutChunker", "Hit", "MemoryStore", "MilvusStore", "Retriever", "Section", "TokenCounter", "Word"]
