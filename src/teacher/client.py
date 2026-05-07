"""Teacher model API client."""

import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv

load_dotenv("configs/teacher.env")


@dataclass
class TeacherConfig:
    """Teacher model configuration."""
    api_url: str = os.getenv("TEACHER_API_URL", "")
    api_key: str = os.getenv("TEACHER_API_KEY", "")
    model: str = os.getenv("TEACHER_MODEL", "gpt-4")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))


class TeacherClient(Protocol):
    """Teacher model protocol."""

    def generate(self, prompt: str) -> str:
        """Generate response from prompt."""
        ...


class MockTeacher:
    """Mock teacher for testing."""

    def generate(self, prompt: str) -> str:
        """Generate mock response."""
        return f"Mock response to: {prompt[:50]}..."


class APITeacher:
    """API-based teacher model for OpenAI-compatible APIs."""

    def __init__(self, config: TeacherConfig | None = None):
        self.config = config or TeacherConfig()
        self._client = None

    def _get_client(self):
        """Lazy init HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.Client(
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate response via API (OpenAI-compatible)."""
        import httpx

        client = self._get_client()
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        try:
            response = client.post(self.config.api_url, json=payload)
            response.raise_for_status()
            data = response.json()

            # OpenAI-style response
            return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"API request failed: {e.response.status_code}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Invalid API response format: {e}")
        except Exception as e:
            raise RuntimeError(f"API request failed: {e}")


def get_teacher() -> TeacherClient:
    """Get teacher client based on config."""
    config = TeacherConfig()
    if not config.api_key:
        return MockTeacher()
    return APITeacher(config)