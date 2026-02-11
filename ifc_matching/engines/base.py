"""Base types and interfaces for matching engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class MatchResult:
    """A single candidate match with scoring metadata."""

    candidate: str
    score: float
    method: str = ""
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"MatchResult({self.candidate!r}, score={self.score:.4f}, method={self.method!r})"


class BaseMatcher(ABC):
    """Interface every matching engine must implement."""

    @abstractmethod
    def match(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
        *,
        element_type: str | None = None,
    ) -> list[MatchResult]:
        """Return up to *top_k* candidates ranked by relevance.

        Parameters
        ----------
        query:
            The IFC product / material name to match.
        candidates:
            The database entries to compare against.
        top_k:
            Maximum number of results to return.
        element_type:
            Optional IFC element type (e.g. ``IfcWall``) to give the
            reranker physical context.
        """
