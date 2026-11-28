"""
Text Splitting Module

Splits long documents into manageable chunks with overlap.
Preserves semantic coherence while respecting token limits.
"""

from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)


class TextSplitter:
    """Split text into chunks with configurable size and overlap."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 100):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Size of each chunk (characters)
            overlap: Number of overlapping characters between chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be between 0 and chunk_size")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text input")
        
        text = text.strip()
        if not text:
            return []
        
        chunks = []
        step_size = self.chunk_size - self.overlap
        
        for i in range(0, len(text), step_size):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text by sentences to preserve meaning.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentence chunks
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text input")
        
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"Split text into {len(chunks)} sentence chunks")
        return chunks
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text by paragraphs first, then by size if needed.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraph-aware chunks
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text input")
        
        # Split by multiple newlines to get paragraphs
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too long, split it
                if len(paragraph) > self.chunk_size:
                    sub_chunks = self.split_text(paragraph)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"Split text into {len(chunks)} paragraph-aware chunks")
        return chunks
    
    def create_chunks_with_metadata(self, text: str, source: str = "unknown") -> List[Dict]:
        """
        Create chunks with metadata.
        
        Args:
            text: Text to split
            source: Source identifier (e.g., file name)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = self.split_text(text)
        
        chunks_with_metadata = []
        for idx, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                'chunk_id': idx,
                'text': chunk,
                'source': source,
                'length': len(chunk),
                'start_position': idx * (self.chunk_size - self.overlap),
                'end_position': idx * (self.chunk_size - self.overlap) + len(chunk)
            })
        
        return chunks_with_metadata


def split_text_simple(text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
    """Convenience function for simple text splitting."""
    splitter = TextSplitter(chunk_size=chunk_size, overlap=overlap)
    return splitter.split_text(text)

