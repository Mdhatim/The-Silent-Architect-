from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class Intent(str, Enum):
    create_file = "create_file"
    write_code = "write_code"
    summarize_text = "summarize_text"
    general_chat = "general_chat"


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    rationale: str = ""

    # For file/code intents
    path: str | None = None
    content: str | None = None

    # For summarize intent
    text: str | None = None

