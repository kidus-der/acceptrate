"""Load and validate design/tokens.json (Pydantic at the boundary).

The palette is fixed-field on purpose: both themes must define exactly the
same colour names so that every semantic alias resolves in both, and so the
generated Go and CSS stay symmetric. Accept/reject are never encoded by
colour alone, which is why glyphs are tokens too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

HEX_PATTERN = r"^#[0-9A-F]{6}$"
"""Six-digit uppercase hex, as written in the brief. Uppercase keeps codegen canonical."""

HexColor = Annotated[str, StringConstraints(pattern=HEX_PATTERN)]
Glyph = Annotated[str, StringConstraints(min_length=1, max_length=1)]
FontName = Annotated[str, StringConstraints(min_length=1)]

DEFAULT_TOKENS_PATH = Path("design") / "tokens.json"
"""Relative to the repo root; the codegen entry point resolves it against `--root`."""


class Palette(BaseModel):
    """One colour set. Field order is the emission order in every generated file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ground: HexColor
    surface: HexColor
    surface_2: HexColor
    surface_3: HexColor
    ink: HexColor
    ink_2: HexColor
    muted: HexColor
    line: HexColor
    line_2: HexColor
    teal: HexColor
    amber: HexColor
    green: HexColor
    red: HexColor

    def items(self) -> tuple[tuple[str, str], ...]:
        return tuple((name, getattr(self, name)) for name in PALETTE_NAMES)


PALETTE_NAMES: tuple[str, ...] = tuple(Palette.model_fields)


class ColorSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dark: Palette
    light: Palette


class Typography(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ui: FontName
    data: FontName
    capture: FontName


class Spring(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stiffness: float = Field(gt=0)
    damping: float = Field(gt=0)


class Motion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spring: Spring
    reduced_motion: bool


class Glyphs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: Glyph
    rejected: Glyph
    bonus: Glyph


class Tokens(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    color: ColorSet
    type: Typography
    motion: Motion
    semantic: dict[str, str]
    glyph: Glyphs

    @model_validator(mode="after")
    def _semantic_aliases_resolve(self) -> Tokens:
        unknown = {role: name for role, name in self.semantic.items() if name not in PALETTE_NAMES}
        if unknown:
            raise ValueError(f"semantic aliases name undefined colours: {unknown}")
        return self


def load_tokens(path: Path) -> Tokens:
    """Read and validate a tokens file; any structural or hex error raises ValidationError."""
    return Tokens.model_validate(json.loads(path.read_text(encoding="utf-8")))
