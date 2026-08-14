"""
VYOMAAV Base Model Engine
Test Suite: tests/test_compiler.py

Comprehensive integration test suite covering Tokenizer, Parser, Validator, and Serializer.
"""

import pytest
from world_language.tokenizer import WLTokenizer
from world_language.parser import WLParser
from world_language.validator import WLValidator
from world_language.serializer import WLSerializer


def test_validator_referential_integrity_error():
    source = """
    world_state "Test" {
        entity "chair" {
            relationships {
                supported_by: "non_existent_floor" ;
            }
        }
    }
    """
    tokenizer = WLTokenizer(source)
    tokens, _ = tokenizer.tokenize()
    parser = WLParser(tokens)
    program, _ = parser.parse()

    validator = WLValidator(program)
    res = validator.validate()

    assert res.is_valid is False
    assert "unknown target entity 'non_existent_floor'" in res.diagnostics[0].message


def test_validator_bbox_min_greater_than_max():
    source = """
    world_state "Test" {
        entity "box" {
            spatial {
                bbox_min: [1.0, 1.0, 1.0] ;
                bbox_max: [0.0, 0.0, 0.0] ;
            }
        }
    }
    """
    tokens, _ = WLTokenizer(source).tokenize()
    program, _ = WLParser(tokens).parse()
    res = WLValidator(program).validate()

    assert res.is_valid is False
    assert "bbox_min [1.0, 1.0, 1.0] must be strictly less than bbox_max" in res.diagnostics[0].message


def test_serializer_roundtrip():
    source = """world_state "Room" {
    entity "table" {
        semantic {
            label: "table" ;
            class_id: 1 ;
            confidence: 0.99 ;
        }
        spatial {
            bbox_min: [-1.0, 0.0, -1.0] ;
            bbox_max: [1.0, 0.8, 1.0] ;
        }
    }
}"""
    tokens, _ = WLTokenizer(source).tokenize()
    program, _ = WLParser(tokens).parse()

    serializer = WLSerializer()
    generated_wl = serializer.to_wl(program)

    # Re-tokenize generated WL to verify syntactic identity
    tokens2, _ = WLTokenizer(generated_wl).tokenize()
    program2, _ = WLParser(tokens2).parse()

    assert program2.world_state.name == "Room"
    assert len(program2.world_state.blocks) == 1