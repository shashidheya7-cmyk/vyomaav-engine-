"""
VYOMAAV Base Model Engine
Module: world_language.tokenizer

Production-grade lexical analyzer for World Language (WL) v1.0.
Decouples core lexical categories from language keywords, tracks character offset spans
for IDE/diagnostic tools, supports non-destructive error recovery, and exposes a versioned API.
"""

from dataclasses import dataclass
import enum
import re
from typing import Any, Dict, List, Optional, Tuple


class WLVersion(enum.Enum):
    V1_0 = "1.0"


class TokenType(enum.Enum):
    """Core lexical categories emitted by the scanner."""
    KEYWORD = "KEYWORD"
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"

    # Symbols
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    COLON = ":"
    SEMICOLON = ";"
    COMMA = ","

    # Diagnostics & Lifecycle
    INVALID = "INVALID"
    EOF = "EOF"


class Keyword(enum.Enum):
    """WL v1.0 Language Keywords."""
    WORLD_STATE = "world_state"
    CAMERA_TRAJECTORY = "camera_trajectory"
    FRAME = "frame"
    ENTITY = "entity"
    ENVIRONMENT = "environment"

    # Block Attributes
    POSE_SE3 = "pose_se3"
    INTRINSICS_K = "intrinsics_k"
    DISTORTION = "distortion"
    FOV = "fov"
    SEMANTIC = "semantic"
    LABEL = "label"
    CLASS_ID = "class_id"
    CONFIDENCE = "confidence"
    SPATIAL = "spatial"
    TRANSFORM_MATRIX = "transform_matrix"
    BBOX_MIN = "bbox_min"
    BBOX_MAX = "bbox_max"
    SDF_LATENT_REF = "sdf_latent_ref"
    MATERIAL = "material"
    BASE_TYPE = "base_type"
    ALBEDO_RGB = "albedo_rgb"
    ROUGHNESS = "roughness"
    METALLIC = "metallic"
    NORMAL_MAP_REF = "normal_map_ref"
    PHYSICS = "physics"
    MASS_KG = "mass_kg"
    FRICTION = "friction"
    RESTITUTION = "restitution"
    IS_STATIC = "is_static"
    AFFORDANCES = "affordances"
    ACTIONS = "actions"
    MAX_LOAD_KG = "max_load_kg"
    RELATIONSHIPS = "relationships"
    UNCERTAINTY = "uncertainty"
    ALEATORIC_NOISE = "aleatoric_noise"
    EPISTEMIC_RISK = "epistemic_risk"
    IS_INFERRED = "is_inferred"
    DYNAMICS = "dynamics"
    LINEAR_VELOCITY = "linear_velocity"
    ANGULAR_VELOCITY = "angular_velocity"
    HDRI_REF = "hdri_ref"
    AMBIENT_INTENSITY = "ambient_intensity"

    # Relationships
    SUPPORTED_BY = "supported_by"
    CONTAINS = "contains"
    ADJACENT_TO = "adjacent_to"
    BLOCKS_PATH = "blocks_path"
    ATTACHED_TO = "attached_to"


# Explicit Version Keyword Registries
KEYWORDS_V1_0: Dict[str, Keyword] = {kw.value: kw for kw in Keyword}


class DiagnosticSeverity(enum.Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Diagnostic:
    """Compiler diagnostic emitted during scanning with character offset spans."""
    message: str
    line: int
    column: int
    start_offset: int
    end_offset: int
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR


@dataclass(frozen=True)
class Token:
    """Immutable lexical token with line, column, and absolute source offset spans."""
    type: TokenType
    value: Any
    line: int
    column: int
    start_offset: int
    end_offset: int
    keyword: Optional[Keyword] = None

    def __repr__(self) -> str:
        if self.keyword:
            return f"Token(KEYWORD:{self.keyword.name}, L{self.line}:C{self.column}, [{self.start_offset}:{self.end_offset}])"
        return f"Token({self.type.name}, {repr(self.value)}, L{self.line}:C{self.column}, [{self.start_offset}:{self.end_offset}])"


class LexicalError(SyntaxError):
    """Raised when strict tokenization encounters an invalid sequence."""
    def __init__(self, message: str, line: int, column: int, start_offset: int, end_offset: int):
        super().__init__(f"Lexical Error [L{line}:C{column}, offset {start_offset}:{end_offset}]: {message}")
        self.line = line
        self.column = column
        self.start_offset = start_offset
        self.end_offset = end_offset


class WLTokenizer:
    """Versioned lexical analyzer for World Language."""

    SPECIFICATION = [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI",  r"/\*[\s\S]*?\*/"),
        ("FLOAT",          r"-?\d+\.\d+([eE][+-]?\d+)?"),
        ("INTEGER",        r"-?\d+"),
        ("STRING",         r'"([^"\\]|\\.)*"'),
        ("SYMBOL",         r"[{}\[\]:;,]"),
        ("IDENTIFIER",     r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("NEWLINE",        r"\n"),
        ("SKIP",           r"[ \t\r]+"),
        ("MISMATCH",       r"."),
    ]

    MASTER_REGEX = re.compile("|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in SPECIFICATION))

    SYMBOL_MAP = {
        "{": TokenType.LBRACE,
        "}": TokenType.RBRACE,
        "[": TokenType.LBRACKET,
        "]": TokenType.RBRACKET,
        ":": TokenType.COLON,
        ";": TokenType.SEMICOLON,
        ",": TokenType.COMMA,
    }

    def __init__(self, source_code: str, version: str = "1.0", strict: bool = True):
        self.source_code = source_code
        self.version = WLVersion(version)
        self.strict = strict
        self.diagnostics: List[Diagnostic] = []

        if self.version == WLVersion.V1_0:
            self.keyword_table = KEYWORDS_V1_0
        else:
            raise ValueError(f"Unsupported WL version: {version}")

    def _decode_string_escapes(
        self, raw_str: str, line: int, column: int, start_offset: int, end_offset: int
    ) -> str:
        """Decodes standard and hex/unicode escape sequences while capturing malformed escapes."""
        idx = 0
        length = len(raw_str)
        out = []
        while idx < length:
            char = raw_str[idx]
            if char == '\\' and idx + 1 < length:
                nxt = raw_str[idx + 1]
                if nxt == 'n': out.append('\n'); idx += 2; continue
                elif nxt == 't': out.append('\t'); idx += 2; continue
                elif nxt == 'r': out.append('\r'); idx += 2; continue
                elif nxt == '"': out.append('"'); idx += 2; continue
                elif nxt == '\\': out.append('\\'); idx += 2; continue
                elif nxt == 'x':
                    if idx + 3 < length:
                        try:
                            out.append(chr(int(raw_str[idx+2:idx+4], 16)))
                            idx += 4
                            continue
                        except ValueError:
                            pass
                    self.diagnostics.append(
                        Diagnostic(
                            f"Malformed hex escape sequence '{raw_str[idx:idx+4]}'",
                            line, column, start_offset, end_offset, DiagnosticSeverity.WARNING
                        )
                    )
                    out.append(raw_str[idx:idx+2])
                    idx += 2
                    continue
                elif nxt == 'u':
                    if idx + 5 < length:
                        try:
                            out.append(chr(int(raw_str[idx+2:idx+6], 16)))
                            idx += 6
                            continue
                        except ValueError:
                            pass
                    self.diagnostics.append(
                        Diagnostic(
                            f"Malformed unicode escape sequence '{raw_str[idx:idx+6]}'",
                            line, column, start_offset, end_offset, DiagnosticSeverity.WARNING
                        )
                    )
                    out.append(raw_str[idx:idx+2])
                    idx += 2
                    continue
            out.append(char)
            idx += 1
        return "".join(out)

    def tokenize(self) -> Tuple[List[Token], List[Diagnostic]]:
        """Scans source_code and returns (tokens, diagnostics)."""
        tokens: List[Token] = []
        line_num = 1
        line_start = 0

        for match in self.MASTER_REGEX.finditer(self.source_code):
            kind = match.lastgroup
            value = match.group()
            start_off = match.start()
            end_off = match.end()
            column = start_off - line_start + 1

            if kind in ("SKIP", "COMMENT_SINGLE"):
                continue

            elif kind == "COMMENT_MULTI":
                newline_count = value.count("\n")
                if newline_count > 0:
                    line_num += newline_count
                    line_start = start_off + value.rfind("\n") + 1
                continue

            elif kind == "NEWLINE":
                line_num += 1
                line_start = end_off
                continue

            elif kind == "FLOAT":
                tokens.append(Token(TokenType.FLOAT, float(value), line_num, column, start_off, end_off))

            elif kind == "INTEGER":
                tokens.append(Token(TokenType.INTEGER, int(value), line_num, column, start_off, end_off))

            elif kind == "STRING":
                cleaned = self._decode_string_escapes(value[1:-1], line_num, column, start_off, end_off)
                tokens.append(Token(TokenType.STRING, cleaned, line_num, column, start_off, end_off))

            elif kind == "SYMBOL":
                tokens.append(Token(self.SYMBOL_MAP[value], value, line_num, column, start_off, end_off))

            elif kind == "IDENTIFIER":
                if value == "true":
                    tokens.append(Token(TokenType.BOOLEAN, True, line_num, column, start_off, end_off))
                elif value == "false":
                    tokens.append(Token(TokenType.BOOLEAN, False, line_num, column, start_off, end_off))
                elif value in self.keyword_table:
                    kw_enum = self.keyword_table[value]
                    tokens.append(
                        Token(TokenType.KEYWORD, value, line_num, column, start_off, end_off, keyword=kw_enum)
                    )
                else:
                    tokens.append(Token(TokenType.IDENTIFIER, value, line_num, column, start_off, end_off))

            elif kind == "MISMATCH":
                diag = Diagnostic(f"Unexpected character {repr(value)}", line_num, column, start_off, end_off)
                self.diagnostics.append(diag)
                if self.strict:
                    raise LexicalError(diag.message, line_num, column, start_off, end_off)
                tokens.append(Token(TokenType.INVALID, value, line_num, column, start_off, end_off))

        eof_offset = len(self.source_code)
        eof_column = eof_offset - line_start + 1
        tokens.append(
            Token(TokenType.EOF, "", line_num, eof_column, eof_offset, eof_offset)
        )
        return tokens, self.diagnostics