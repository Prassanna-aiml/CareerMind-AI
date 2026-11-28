"""
LLM Model Module

Handles interaction with language models.
Supports both local models (Ollama) and API-based models (OpenAI).
"""

from typing import Optional, List, Dict
import logging
import os
from abc import ABC, abstractmethod
import requests

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_with_context(self, context: str, question: str, **kwargs) -> str:
        """Generate answer with context (RAG)."""
        pass


class OllamaLLM(BaseLLM):
    """Local LLM using Ollama."""
    
    def __init__(self, model_name: str = "tinyllama", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama LLM.
        
        Args:
            model_name: Name of the model (llama3, mistral, neural-chat, etc.)
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        
        # Check if Ollama is running
        self._check_connection()
        logger.info(f"Initialized Ollama LLM with model: {model_name}")
    
    def _check_connection(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            is_connected = response.status_code == 200
            if is_connected:
                logger.info(f"Ollama connection verified at {self.base_url}")
            else:
                logger.warning(f"Ollama connection check failed with status: {response.status_code}")
            return is_connected
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            return False
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            return False
    
    def generate(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            temperature: Controls randomness (0.0 - 1.0)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        try:
            logger.info(f"Generating text using Ollama model: {self.model_name}")
            logger.debug(f"Prompt length: {len(prompt)} chars, Temperature: {temperature}")
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False
            }
            
            response = requests.post(self.api_endpoint, json=payload, timeout=90)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            if not generated_text:
                logger.warning(f"Empty response from Ollama. Full response: {result}")
                raise ValueError("Ollama returned empty response")
            
            logger.info(f"✅ Successfully generated {len(generated_text)} chars from {self.model_name}")
            return generated_text
        
        except requests.exceptions.Timeout:
            error_msg = "Ollama request timed out (90 seconds). Model may be slow or hung."
            logger.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to Ollama at {self.base_url}. Is it running? Error: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error generating text with {self.model_name}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def generate_with_context(
        self, 
        context: str, 
        question: str, 
        temperature: float = 0.7
    ) -> str:
        """
        Generate answer with context (RAG).
        
        Args:
            context: Retrieved context/chunks
            question: User question
            temperature: Temperature parameter
            
        Returns:
            Generated answer
        """
        prompt = f"""You are an academic assistant. Answer the question based on the provided context.

Context:
{context}

Question:
{question}

Answer:"""
        
        return self.generate(prompt, temperature=temperature)


class OpenAILLM(BaseLLM):
    """LLM using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI LLM.
        
        Args:
            api_key: OpenAI API key
            model_name: Model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided or found in environment")
        
        logger.info(f"Initialized OpenAI LLM with model: {model_name}")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate text using OpenAI.
        
        Args:
            prompt: Input prompt
            temperature: Temperature parameter
            
        Returns:
            Generated text
        """
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content.strip()
            logger.info(f"Generated text of length: {len(generated_text)}")
            return generated_text
        
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise
    
    def generate_with_context(
        self, 
        context: str, 
        question: str, 
        temperature: float = 0.7
    ) -> str:
        """
        Generate answer with context (RAG).
        
        Args:
            context: Retrieved context/chunks
            question: User question
            temperature: Temperature parameter
            
        Returns:
            Generated answer
        """
        system_message = "You are an academic assistant. Answer the question based on the provided context. Be accurate and cite sources when possible."
        
        user_message = f"""Context:
{context}

Question:
{question}

Answer:"""
        
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content.strip()
            return generated_text
        
        except Exception as e:
            logger.error(f"Error generating answer with context: {e}")
            raise


def create_llm(
    llm_type: str = "ollama",
    model_name: str = "tinyllama",
    api_key: Optional[str] = None
) -> BaseLLM:
    """
    Factory function to create LLM instance.
    
    Args:
        llm_type: Type of LLM ('ollama' or 'openai')
        model_name: Name of the model
        api_key: API key for OpenAI (if needed)
        
    Returns:
        LLM instance
    """
    if llm_type.lower() == "ollama":
        return OllamaLLM(model_name=model_name)
    elif llm_type.lower() == "openai":
        return OpenAILLM(api_key=api_key, model_name=model_name)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
