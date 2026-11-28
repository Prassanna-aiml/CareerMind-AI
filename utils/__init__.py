"""
Utils Package

Modules for AI Academic Assistant.
"""

from .pdf_reader import PDFReader, read_pdf
from .text_splitter import TextSplitter, split_text_simple
from .embeddings import EmbeddingsManager, create_embeddings_manager
from .llm_model import BaseLLM, OllamaLLM, OpenAILLM, create_llm
from .rag_pipeline import RAGPipeline, FAISSVectorStore

__all__ = [
    'PDFReader',
    'read_pdf',
    'TextSplitter',
    'split_text_simple',
    'EmbeddingsManager',
    'create_embeddings_manager',
    'BaseLLM',
    'OllamaLLM',
    'OpenAILLM',
    'create_llm',
    'RAGPipeline',
    'FAISSVectorStore',
]
