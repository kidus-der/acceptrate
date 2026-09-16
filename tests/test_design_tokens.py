"""design/tokens.json is the single source of truth and is validated at load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from acceptrate.design.tokens import PALETTE_NAMES, Tokens, load_tokens

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENS_PATH = REPO_ROOT / "design" / "tokens.json"


def _tampered(tmp_path: Path, mutate) -> Path:  # noqa: ANN001
    data = json.loads(TOKENS_PATH.read_text())
    mutate(data)
    target = tmp_path / "tokens.json"
    target.write_text(json.dumps(data))
    return target


def test_committed_tokens_load_and_carry_the_instrument_palette() -> None:
    tokens = load_tokens(TOKENS_PATH)

    assert tokens.color.dark.ground == "#0E1319"
    assert tokens.color.dark.teal == "#52B9C8"
    assert tokens.color.light.ground == "#E9ECF0"
    assert tokens.color.light.red == "#AE3448"
    assert tokens.type.ui == "Geist"
    assert tokens.glyph.accepted == "●"


def test_bad_hex_fails_validation(tmp_path: Path) -> None:
    def mutate(data: dict) -> None:
        data["color"]["dark"]["teal"] = "#52B9C"

    with pytest.raises(ValidationError):
        load_tokens(_tampered(tmp_path, mutate))


def test_unknown_colour_name_is_rejected(tmp_path: Path) -> None:
    def mutate(data: dict) -> None:
        data["color"]["light"]["magenta"] = "#FF00FF"

    with pytest.raises(ValidationError):
        load_tokens(_tampered(tmp_path, mutate))


def test_semantic_alias_must_name_a_defined_colour(tmp_path: Path) -> None:
    def mutate(data: dict) -> None:
        data["semantic"]["accepted"] = "chartreuse"

    with pytest.raises(ValidationError):
        load_tokens(_tampered(tmp_path, mutate))


def test_every_committed_semantic_alias_resolves() -> None:
    tokens = load_tokens(TOKENS_PATH)

    for role, colour in tokens.semantic.items():
        assert colour in PALETTE_NAMES, f"semantic {role!r} -> {colour!r} is not a palette colour"


def test_accept_and_reject_carry_distinct_glyphs_not_just_colour() -> None:
    tokens = load_tokens(TOKENS_PATH)

    glyphs = (tokens.glyph.accepted, tokens.glyph.rejected, tokens.glyph.bonus)
    assert len(set(glyphs)) == 3
    assert all(len(g) == 1 for g in glyphs)


def test_glyph_must_be_a_single_character(tmp_path: Path) -> None:
    def mutate(data: dict) -> None:
        data["glyph"]["accepted"] = "ok"

    with pytest.raises(ValidationError):
        load_tokens(_tampered(tmp_path, mutate))


def test_light_and_dark_palettes_share_one_key_set() -> None:
    tokens = load_tokens(TOKENS_PATH)

    assert tuple(tokens.color.dark.model_dump()) == PALETTE_NAMES
    assert tuple(tokens.color.light.model_dump()) == PALETTE_NAMES


def test_tokens_are_frozen() -> None:
    tokens = load_tokens(TOKENS_PATH)

    with pytest.raises(ValidationError):
        tokens.color.dark.teal = "#000000"  # type: ignore[misc]


def test_tokens_model_rejects_missing_sections() -> None:
    with pytest.raises(ValidationError):
        Tokens.model_validate({"color": {}})
