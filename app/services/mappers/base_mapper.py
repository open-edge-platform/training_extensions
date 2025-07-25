from abc import ABC, abstractmethod
from typing import Any

from app.db.schema import Base


class BaseMapper(ABC):
    """Abstract base class for all mappers."""

    @abstractmethod
    def to_schema(self, model: Base) -> Any:
        """Convert database model to Pydantic schema."""

    @abstractmethod
    def from_schema(self, schema: Any) -> Any:
        """Convert Pydantic schema to database model."""
