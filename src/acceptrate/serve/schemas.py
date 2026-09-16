"""Request and response bodies for the OpenAI-compatible endpoint (Pydantic v2).

Only the subset OpenAI clients actually send is validated; unknown fields
(top_p, n, presence_penalty, ...) are ignored so Open WebUI / Zed /
continue.dev work with a base-URL change and nothing else.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MAX_TOKENS = 512
MAX_TOKENS_LIMIT = 8192
"""Hard ceiling per request; the 8B target's KV cache is sized for 8k context."""

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length"]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=MAX_TOKENS_LIMIT)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=MAX_TOKENS_LIMIT)
    stream: bool = False
    temperature: float = Field(default=0.0, ge=0.0)

    @property
    def completion_budget(self) -> int:
        """`max_completion_tokens` is OpenAI's newer name; it wins when both are given."""
        return self.max_completion_tokens or self.max_tokens


class Delta(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = 0
    delta: Delta
    finish_reason: FinishReason | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]


class Choice(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = 0
    message: Message
    finish_reason: FinishReason


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ModelCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "acceptrate"


class ModelList(BaseModel):
    model_config = ConfigDict(frozen=True)

    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    type: str
    code: int


class ErrorResponse(BaseModel):
    """OpenAI's error envelope, so clients surface the message instead of a bare status."""

    model_config = ConfigDict(frozen=True)

    error: ErrorBody


class Health(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    busy: bool
