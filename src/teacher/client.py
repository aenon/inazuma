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
    """API-based teacher model."""

    def __init__(self, config: TeacherConfig | None = None):
        self.config = config or TeacherConfig()

    def generate(self, prompt: str) -> str:
        """Generate response via API."""
        import httpx
        # Simple implementation - replace with actual API call
        raise NotImplementedError("API teacher needs implementation")


def get_teacher() -> TeacherClient:
    """Get teacher client based on config."""
    config = TeacherConfig()
    if not config.api_key:
        return MockTeacher()
    return APITeacher(config)