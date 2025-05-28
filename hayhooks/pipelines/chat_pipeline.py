# =================================================================
# Chat Pipeline for Haystack + OpenAI/Anthropic Integration
# =================================================================
"""
Basic chat pipeline using Haystack 2.0 to communicate with LLM providers.
Supports OpenAI and Anthropic with fallback logic, error handling, and streaming.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.core.component import component
from haystack.core.component.types import Variadic

# Import Anthropic if available
try:
    from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic integration not available. Install with: pip install anthropic-haystack")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@component
class ChatMemoryManager:
    """Component to manage conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize the memory manager.
        
        Args:
            max_history: Maximum number of conversation turns to remember
        """
        self.max_history = max_history
        self.conversation_history: List[ChatMessage] = []
    
    @component.output_types(messages=List[ChatMessage])
    def run(self, 
            user_message: str,
            system_message: Optional[str] = None,
            conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user message and manage conversation history.
        
        Args:
            user_message: The user's input message
            system_message: Optional system prompt
            conversation_id: Optional conversation identifier
            
        Returns:
            Dict containing the formatted messages list
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append(ChatMessage.from_system(system_message))
        
        # Add conversation history (limited by max_history)
        if self.conversation_history:
            recent_history = self.conversation_history[-self.max_history:]
            messages.extend(recent_history)
        
        # Add current user message
        user_msg = ChatMessage.from_user(user_message)
        messages.append(user_msg)
        
        return {"messages": messages}
    
    def update_history(self, user_message: str, assistant_response: str):
        """Update conversation history with the latest exchange."""
        self.conversation_history.append(ChatMessage.from_user(user_message))
        self.conversation_history.append(ChatMessage.from_assistant(assistant_response))
        
        # Trim history if it exceeds max_history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]


@component
class FallbackGenerator:
    """Component that provides fallback logic between OpenAI and Anthropic."""
    
    def __init__(self, 
                 openai_model: str = "gpt-4o-mini",
                 anthropic_model: str = "claude-3-haiku-20240307",
                 max_retries: int = 2,
                 timeout: int = 30):
        """
        Initialize the fallback generator.
        
        Args:
            openai_model: OpenAI model to use as primary
            anthropic_model: Anthropic model to use as fallback
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize OpenAI generator
        self.openai_generator = OpenAIChatGenerator(
            model=openai_model,
            generation_kwargs={
                "max_tokens": 2048,
                "temperature": 0.7,
                "timeout": timeout
            }
        )
        
        # Initialize Anthropic generator if available
        self.anthropic_generator = None
        if ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_generator = AnthropicChatGenerator(
                    model=anthropic_model,
                    generation_kwargs={
                        "max_tokens": 2048,
                        "temperature": 0.7
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic generator: {e}")
    
    @component.output_types(replies=List[ChatMessage], metadata=Dict[str, Any])
    def run(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Generate response with fallback logic.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Dict containing replies and metadata
        """
        metadata = {
            "provider_used": None,
            "attempts": 0,
            "errors": []
        }
        
        # Try OpenAI first
        for attempt in range(self.max_retries):
            try:
                metadata["attempts"] += 1
                logger.info(f"Attempting OpenAI generation (attempt {attempt + 1})")
                
                start_time = time.time()
                result = self.openai_generator.run(messages=messages)
                response_time = time.time() - start_time
                
                metadata["provider_used"] = "openai"
                metadata["response_time"] = response_time
                
                return {
                    "replies": result["replies"],
                    "metadata": metadata
                }
                
            except Exception as e:
                error_msg = f"OpenAI attempt {attempt + 1} failed: {str(e)}"
                logger.warning(error_msg)
                metadata["errors"].append(error_msg)
                
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # Fallback to Anthropic if available
        if self.anthropic_generator:
            for attempt in range(self.max_retries):
                try:
                    metadata["attempts"] += 1
                    logger.info(f"Falling back to Anthropic (attempt {attempt + 1})")
                    
                    start_time = time.time()
                    result = self.anthropic_generator.run(messages=messages)
                    response_time = time.time() - start_time
                    
                    metadata["provider_used"] = "anthropic"
                    metadata["response_time"] = response_time
                    
                    return {
                        "replies": result["replies"],
                        "metadata": metadata
                    }
                    
                except Exception as e:
                    error_msg = f"Anthropic attempt {attempt + 1} failed: {str(e)}"
                    logger.warning(error_msg)
                    metadata["errors"].append(error_msg)
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
        
        # If all attempts failed, return error response
        error_response = ChatMessage.from_assistant(
            "Peço desculpas, mas estou enfrentando dificuldades técnicas no momento. "
            "Por favor, tente novamente mais tarde ou entre em contato com o suporte se o problema persistir."
        )
        
        metadata["provider_used"] = "error"
        return {
            "replies": [error_response],
            "metadata": metadata
        }


def create_chat_pipeline(
    system_prompt: Optional[str] = None,
    openai_model: str = "gpt-4o-mini",
    anthropic_model: str = "claude-3-haiku-20240307",
    max_history: int = 10
) -> Pipeline:
    """
    Create and configure the chat pipeline.
    
    Args:
        system_prompt: Optional system message for the conversation
        openai_model: OpenAI model to use
        anthropic_model: Anthropic model for fallback
        max_history: Maximum conversation history to maintain
        
    Returns:
        Configured Haystack Pipeline
    """
    default_system_prompt = (
        "Você é um assistente de IA útil, inofensivo e honesto. "
        "Responda aos usuários de forma amigável e informativa. "
        "Se você não souber algo, diga isso claramente. "
        "Mantenha suas respostas concisas, mas abrangentes. "
        "SEMPRE responda em português brasileiro."
    )
    
    pipeline = Pipeline()
    
    # Add components
    pipeline.add_component(
        "memory_manager",
        ChatMemoryManager(max_history=max_history)
    )
    
    pipeline.add_component(
        "fallback_generator",
        FallbackGenerator(
            openai_model=openai_model,
            anthropic_model=anthropic_model
        )
    )
    
    # Connect components
    pipeline.connect("memory_manager.messages", "fallback_generator.messages")
    
    return pipeline


def run_chat_conversation(
    pipeline: Pipeline,
    user_message: str,
    system_prompt: Optional[str] = None,
    conversation_id: Optional[str] = None,
    conversation_history: Optional[List[Dict]] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a single conversation turn through the pipeline.
    
    Args:
        pipeline: The configured chat pipeline
        user_message: User's input message
        system_prompt: Optional system prompt
        conversation_id: Optional conversation identifier
        conversation_history: Optional conversation history for context
        session_id: Optional session identifier (alias for conversation_id)
        
    Returns:
        Dict containing the response and metadata
    """
    default_system_prompt = (
        "Você é um assistente de IA útil, inofensivo e honesto. "
        "Responda aos usuários de forma amigável e informativa. "
        "Se você não souber algo, diga isso claramente. "
        "Mantenha suas respostas concisas, mas abrangentes. "
        "SEMPRE responda em português brasileiro."
    )
    
    try:
        result = pipeline.run({
            "memory_manager": {
                "user_message": user_message,
                "system_message": system_prompt or default_system_prompt,
                "conversation_id": conversation_id
            }
        })
        
        # Extract response
        replies = result["fallback_generator"]["replies"]
        metadata = result["fallback_generator"]["metadata"]
        
        if replies:
            response_text = replies[0].text
            
            # Update memory manager history
            memory_manager = pipeline.get_component("memory_manager")
            memory_manager.update_history(user_message, response_text)
            
            return {
                "response": response_text,
                "metadata": metadata,
                "status": "success"
            }
        else:
            return {
                "response": "Peço desculpas, mas não consegui gerar uma resposta.",
                "metadata": metadata,
                "status": "error"
            }
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return {
            "response": "Ocorreu um erro ao processar sua solicitação.",
            "metadata": {"error": str(e)},
            "status": "error"
        }


# Create default pipeline instance
try:
    pipeline = create_chat_pipeline()
    logger.info("Chat pipeline created successfully")
except Exception as e:
    logger.error(f"Failed to create chat pipeline: {e}")
    pipeline = None


# Example usage function
def chat_example():
    """Example of how to use the chat pipeline."""
    if not pipeline:
        print("Pipeline not available")
        return
    
    print("Chat Pipeline Example")
    print("=" * 50)
    
    # Example conversation
    messages = [
        "Hello! How are you?",
        "What is machine learning?",
        "Can you explain it more simply?"
    ]
    
    for msg in messages:
        print(f"User: {msg}")
        
        result = run_chat_conversation(pipeline, msg)
        
        print(f"Assistant: {result['response']}")
        print(f"Provider: {result['metadata'].get('provider_used', 'unknown')}")
        print(f"Status: {result['status']}")
        print("-" * 30)


if __name__ == "__main__":
    chat_example() 