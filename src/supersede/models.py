"""Minimal bi-temporal fact model for the supersede environment.

Trimmed from the Engram / Vrin ``Fact`` model down to what the
supersession environment needs: a ``(subject, predicate, object)`` triple
with a validity window and a supersession pointer. Anything richer
(provenance, derivation chains, embeddings) is out of scope here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class Fact(BaseModel):
    """A ``(subject, predicate, object)`` triple with validity and supersession.

    Bi-temporal: ``valid_from`` / ``valid_to`` describe when the fact is
    true in the world; ``recorded_at`` describes when the environment
    introduced it. A superseded fact keeps its row and points at its
    successor via ``superseded_by`` so the full history (and therefore the
    "which version was stale at turn t" ground truth) is preserved.
    """

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(default_factory=lambda: _new_id("fact"))]
    subject: str
    predicate: str
    object: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    recorded_at: Annotated[datetime, Field(default_factory=datetime.now)]
    superseded_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """A fact is active while nothing has superseded it."""
        return self.superseded_by is None

    def statement(self) -> str:
        """Render the triple as a short natural-language clause."""
        return f"{self.subject} {self.predicate} {self.object}".strip()


__all__ = ["Fact"]
