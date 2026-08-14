"""
VYOMAAV Base Model Engine
Test Suite: tests/test_tokenizer.py

Expanded Pytest suite validating lexical analysis, source offsets, escape decoding,
malformed comments, empty files, and diagnostic recovery.
"""

import pytest
from world_language.tokenizer import (
    WLTokenizer, TokenType, Keyword, LexicalError, DiagnosticSeverity
)


def test_component_keyword_isolation():
    source = "Component company entity_component"
    tokenizer = WLTokenizer(source)
    tokens, diags = tokenizer.tokenize()

    assert len(diags) == 0
    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[0].value == "Component"


def test_source_offsets():
    source = "world_state entity"
    tokenizer = WLTokenizer(source)
    tokens, _ = tokenizer.tokenize()

    assert tokens[0].start_offset == 0 and tokens[0].end_offset == 11
    assert tokens[1].start_offset == 12 and tokens[1].end_offset == 18


def test_basic_tokenization():
    source = 'world_state "Room_01" { confidence: 0.95; is_static: true; mass_kg: 10; }'
    tokenizer = WLTokenizer(source)
    tokens, diags = tokenizer.tokenize()

    assert len(diags) == 0
    assert tokens[0].type == TokenType.KEYWORD
    assert tokens[0].keyword == Keyword.WORLD_STATE
    assert tokens[1].type == TokenType.STRING
    assert tokens[1].value == "Room_01"
    assert tokens[9].type == TokenType.BOOLEAN
    assert tokens[9].value is True


def test_string_escape_sequences():
    source = r'"Line1\nLine2\tTabbed \"Quotes\" \x41 \u0042"'
    tokenizer = WLTokenizer(source)
    tokens, _ = tokenizer.tokenize()

    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == 'Line1\nLine2\tTabbed "Quotes" A B'


def test_malformed_escape_sequences():
    source = r'"Bad hex \xZZ and bad unicode \u12"'
    tokenizer = WLTokenizer(source, strict=False)
    tokens, diags = tokenizer.tokenize()

    assert len(diags) == 2
    assert "Malformed hex escape" in diags[0].message
    assert "Malformed unicode escape" in diags[1].message
    assert tokens[0].type == TokenType.STRING


def test_empty_file():
    tokenizer = WLTokenizer("")
    tokens, diags = tokenizer.tokenize()

    assert len(diags) == 0
    assert len(tokens) == 1
    assert tokens[0].type == TokenType.EOF
    assert tokens[0].start_offset == 0 and tokens[0].end_offset == 0


def test_large_file_tokenization():
    source = ("entity \"item\" { mass_kg: 1.0; }\n" * 1000)
    tokenizer = WLTokenizer(source)
    tokens, diags = tokenizer.tokenize()

    assert len(diags) == 0
    assert len(tokens) == (1000 * 8) + 1  # 9 tokens per line + EOF
    assert tokens[-1].type == TokenType.EOF


def test_line_and_column_tracking():
    source = "world_state\n   \"Test\"\n   {"
    tokenizer = WLTokenizer(source)
    tokens, _ = tokenizer.tokenize()

    assert tokens[0].line == 1 and tokens[0].column == 1
    assert tokens[1].line == 2 and tokens[1].column == 4
    assert tokens[2].line == 3 and tokens[2].column == 4


def test_comment_stripping():
    source = """
    // Single line comment
    world_state /* multi
    line comment */ "Scene"
    """
    tokenizer = WLTokenizer(source)
    tokens, _ = tokenizer.tokenize()

    assert len(tokens) == 3
    assert tokens[0].type == TokenType.KEYWORD
    assert tokens[1].value == "Scene"


def test_strict_mode_error():
    source = "world_state @invalid"
    tokenizer = WLTokenizer(source, strict=True)
    with pytest.raises(LexicalError) as exc_info:
        tokenizer.tokenize()
    assert "Unexpected character '@'" in str(exc_info.value)


def test_non_strict_error_recovery():
    source = "world_state @invalid"
    tokenizer = WLTokenizer(source, strict=False)
    tokens, diags = tokenizer.tokenize()

    assert len(diags) == 1
    assert diags[0].severity == DiagnosticSeverity.ERROR
    assert tokens[1].type == TokenType.INVALID
    assert tokens[1].value == "@"
    assert tokens[2].type == TokenType.IDENTIFIER
    assert tokens[2].value == "invalid"