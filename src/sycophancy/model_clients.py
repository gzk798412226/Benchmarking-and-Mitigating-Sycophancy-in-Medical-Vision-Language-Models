from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Protocol

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class ModelClient(Protocol):
    """Protocol for model clients that can generate responses."""
    def generate(self, prompt: str, image_path: str | None = None) -> str:
        """Generate response from prompt, optionally with image."""
        ...


def encode_image(path: str) -> str:
    """Encode image file to base64 string."""
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


class OpenAIChatClient:
    """OpenAI-compatible API client."""
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        # Use environment variable if api_key not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        """Generate response using OpenAI API."""
        content = [{"type": "text", "text": prompt}]
        if image_path and Path(image_path).exists():
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}
            })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()


class ClaudeClient:
    """Anthropic Claude API client."""
    def __init__(self, model: str = "claude-3-opus-20240229", api_key: str | None = None):
        if anthropic is None:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        """Generate response using Claude API."""
        content = [{"type": "text", "text": prompt}]
        
        if image_path and Path(image_path).exists():
            encoded_image = encode_image(image_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encoded_image
                }
            })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text.strip()


class GeminiClient:
    """Google Gemini API client."""
    def __init__(self, model: str = "gemini-pro-vision", api_key: str | None = None):
        if genai is None:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
        
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        """Generate response using Gemini API."""
        if image_path and Path(image_path).exists():
            from PIL import Image
            image = Image.open(image_path)
            response = self.model.generate_content([prompt, image])
        else:
            response = self.model.generate_content(prompt)
        return response.text.strip()


class EchoClient:
    """Echo client for testing - returns last line of prompt."""
    def generate(self, prompt: str, image_path: str | None = None) -> str:
        """Echo back the last line of the prompt (for testing)."""
        return prompt.split("\n")[-1].strip() if prompt else "A"


def build_client(
    kind: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None
) -> ModelClient:
    """
    Build a model client based on type.
    
    Args:
        kind: Client type ("openai", "claude", "gemini", "echo")
        model: Model name
        api_key: API key (optional, can use environment variable)
        base_url: Base URL for API (optional, mainly for OpenAI)
    
    Returns:
        ModelClient instance
    """
    if kind == "openai":
        return OpenAIChatClient(model=model, api_key=api_key, base_url=base_url)
    elif kind == "claude":
        return ClaudeClient(model=model, api_key=api_key)
    elif kind == "gemini":
        return GeminiClient(model=model, api_key=api_key)
    else:
        return EchoClient()
