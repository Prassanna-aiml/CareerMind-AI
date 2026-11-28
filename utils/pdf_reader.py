"""
PDF Reader Module

Extracts text from PDF files using pdfplumber for better accuracy.
Handles various PDF formats and encoding issues.
"""

import os
from typing import List, Dict, Tuple
import pdfplumber
import PyPDF2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PDFReader:
    """Extract text and metadata from PDF files."""
    
    def __init__(self):
        """Initialize PDF reader."""
        self.supported_formats = ['.pdf']
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text from PDF file using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if not pdf_path.lower().endswith('.pdf'):
                raise ValueError(f"File must be PDF format, got: {pdf_path}")
            
            full_text = ""
            metadata = {}
            
            # Extract using pdfplumber (better for formatted text)
            with pdfplumber.open(pdf_path) as pdf:
                metadata['total_pages'] = len(pdf.pages)
                metadata['title'] = pdf.metadata.get('title', 'Unknown')
                metadata['author'] = pdf.metadata.get('author', 'Unknown')
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        full_text += f"\n[Page {page_num}]\n{text}\n"
            
            metadata['file_name'] = os.path.basename(pdf_path)
            metadata['file_size_kb'] = os.path.getsize(pdf_path) / 1024
            
            logger.info(f"Successfully extracted text from {pdf_path}")
            return full_text, metadata
        
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise Exception(f"Error extracting PDF text: {str(e)}")
    
    def extract_pages(self, pdf_path: str) -> List[Dict]:
        """
        Extract text page by page.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page dictionaries with text and metadata
        """
        pages = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'bbox': page.bbox,
                        'width': page.width,
                        'height': page.height
                    })
            
            return pages
        
        except Exception as e:
            logger.error(f"Error extracting pages: {str(e)}")
            raise Exception(f"Error extracting pages: {str(e)}")
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict:
        """
        Extract PDF metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return {
                    'total_pages': len(pdf.pages),
                    'title': pdf.metadata.get('title', 'Unknown'),
                    'author': pdf.metadata.get('author', 'Unknown'),
                    'creation_date': str(pdf.metadata.get('creationDate', 'Unknown')),
                    'file_name': os.path.basename(pdf_path),
                    'file_size_kb': os.path.getsize(pdf_path) / 1024
                }
        
        except Exception as e:
            logger.error(f"Error reading metadata: {str(e)}")
            raise Exception(f"Error reading metadata: {str(e)}")


# Convenience functions
def read_pdf(pdf_path: str) -> Tuple[str, Dict]:
    """Quick function to read PDF."""
    reader = PDFReader()
    return reader.extract_text_from_pdf(pdf_path)
