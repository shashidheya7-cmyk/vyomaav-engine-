"""
VYOMAAV Base Model Engine
Module: world_language.parser

Production-grade recursive-descent parser for World Language (WL) v1.0 (V2 Refactor).
Enforces immutable AST node creation, abstracts component block parsing,
and provides token synchronization recovery for non-strict error handling.
"""

from typing import List, Optional, Tuple, Any, Dict, Callable
from world_language.tokenizer import (
    Token, TokenType, Keyword, Diagnostic, DiagnosticSeverity
)
from world_language.ast import (
    ProgramNode, WorldStateNode, CameraTrajectoryNode, CameraFrameNode,
    EntityNode, SemanticComponentNode, SpatialComponentNode, MaterialComponentNode,
    PhysicsComponentNode, AffordanceComponentNode, RelationshipComponentNode,
    RelationPairNode, UncertaintyComponentNode, DynamicsComponentNode,
    EnvironmentNode, ASTNode
)


class ParseError(SyntaxError):
    """Raised when strict parsing encounters an unrecoverable syntax error."""
    def __init__(self, message: str, token: Token):
        super().__init__(f"Parse Error [L{token.line}:C{token.column}]: {message}")
        self.token = token


class WLParser:
    """Versioned recursive-descent parser for World Language."""

    SYNC_KEYWORDS = {
        Keyword.WORLD_STATE, Keyword.CAMERA_TRAJECTORY,
        Keyword.ENTITY, Keyword.ENVIRONMENT, Keyword.FRAME
    }

    def __init__(self, tokens: List[Token], strict: bool = True):
        self.tokens = tokens
        self.strict = strict
        self.pos = 0
        self.diagnostics: List[Diagnostic] = []

    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF Token

    def advance(self) -> Token:
        tok = self.current()
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def match_keyword(self, keyword: Keyword) -> bool:
        tok = self.current()
        return tok.type == TokenType.KEYWORD and tok.keyword == keyword

    def match_type(self, token_type: TokenType) -> bool:
        return self.current().type == token_type

    def consume_type(self, token_type: TokenType, expected_msg: str) -> Token:
        tok = self.current()
        if tok.type == token_type:
            return self.advance()
        self._report_error(f"Expected {expected_msg}, got '{tok.value}'", tok)
        return tok

    def consume_keyword(self, keyword: Keyword) -> Token:
        tok = self.current()
        if tok.type == TokenType.KEYWORD and tok.keyword == keyword:
            return self.advance()
        self._report_error(f"Expected keyword '{keyword.value}', got '{tok.value}'", tok)
        return tok

    def _report_error(self, message: str, token: Token):
        diag = Diagnostic(message, token.line, token.column, token.start_offset, token.end_offset)
        self.diagnostics.append(diag)
        if self.strict:
            raise ParseError(message, token)

    def _synchronize(self):
        """Skip tokens until reaching a block boundary or top-level keyword."""
        self.advance()
        while not self.match_type(TokenType.EOF):
            if self.match_type(TokenType.SEMICOLON) or self.match_type(TokenType.RBRACE):
                self.advance()
                return
            if self.current().type == TokenType.KEYWORD and self.current().keyword in self.SYNC_KEYWORDS:
                return
            self.advance()

    def parse(self) -> Tuple[Optional[ProgramNode], List[Diagnostic]]:
        """Main entry point: parses Token stream into a ProgramNode AST."""
        if self.tokens[0].type == TokenType.EOF:
            return None, self.diagnostics

        start_tok = self.current()
        self.consume_keyword(Keyword.WORLD_STATE)
        name_tok = self.consume_type(TokenType.STRING, "world_state string identifier")
        self.consume_type(TokenType.LBRACE, "'{' to start world_state block")

        blocks: List[ASTNode] = []
        while not self.match_type(TokenType.RBRACE) and not self.match_type(TokenType.EOF):
            try:
                block = self._parse_top_level_block()
                if block:
                    blocks.append(block)
            except ParseError:
                if not self.strict:
                    self._synchronize()
                else:
                    raise

        end_tok = self.consume_type(TokenType.RBRACE, "'}' to close world_state block")

        world_node = WorldStateNode(
            line=start_tok.line, column=start_tok.column,
            start_offset=start_tok.start_offset, end_offset=end_tok.end_offset,
            name=name_tok.value, blocks=blocks
        )

        program = ProgramNode(
            line=start_tok.line, column=start_tok.column,
            start_offset=start_tok.start_offset, end_offset=end_tok.end_offset,
            world_state=world_node
        )

        return program, self.diagnostics

    def _parse_top_level_block(self) -> Optional[ASTNode]:
        tok = self.current()
        if tok.type == TokenType.KEYWORD:
            if tok.keyword == Keyword.CAMERA_TRAJECTORY:
                return self._parse_camera_trajectory()
            elif tok.keyword == Keyword.ENTITY:
                return self._parse_entity()
            elif tok.keyword == Keyword.ENVIRONMENT:
                return self._parse_environment()

        self._report_error(f"Unexpected top-level keyword '{tok.value}'", tok)
        self._synchronize()
        return None

    # --- Component Parsing Abstraction ---

    def _parse_component_block(self, block_keyword: Keyword, attr_handlers: Dict[Keyword, Callable[[], None]]) -> Token:
        """Generic helper for parsing '{ attribute: value; ... }' blocks."""
        start_tok = self.consume_keyword(block_keyword)
        self.consume_type(TokenType.LBRACE, f"'{{' to open {block_keyword.value}")

        while not self.match_type(TokenType.RBRACE) and not self.match_type(TokenType.EOF):
            tok = self.current()
            if tok.type == TokenType.KEYWORD and tok.keyword in attr_handlers:
                self.advance()
                self.consume_type(TokenType.COLON, f"':' after {tok.value}")
                attr_handlers[tok.keyword]()
                self.consume_type(TokenType.SEMICOLON, f"';' after {tok.value}")
            else:
                self._report_error(f"Unexpected attribute in {block_keyword.value}: '{tok.value}'", tok)
                self.advance()

        end_tok = self.consume_type(TokenType.RBRACE, f"'}}' to close {block_keyword.value}")
        return end_tok

    # --- Block Parsers ---

    def _parse_camera_trajectory(self) -> CameraTrajectoryNode:
        start_tok = self.consume_keyword(Keyword.CAMERA_TRAJECTORY)
        self.consume_type(TokenType.LBRACE, "'{' to open camera_trajectory")

        frames: List[CameraFrameNode] = []
        while not self.match_type(TokenType.RBRACE) and not self.match_type(TokenType.EOF):
            if self.match_keyword(Keyword.FRAME):
                frames.append(self._parse_camera_frame())
            else:
                self._report_error(f"Expected 'frame' block in trajectory, got '{self.current().value}'", self.current())
                self.advance()

        end_tok = self.consume_type(TokenType.RBRACE, "'}' to close camera_trajectory")
        return CameraTrajectoryNode(start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset, frames)

    def _parse_camera_frame(self) -> CameraFrameNode:
        start_tok = self.consume_keyword(Keyword.FRAME)
        frame_id_tok = self.consume_type(TokenType.STRING, "frame ID string")

        pose_se3, intrinsics_k, distortion, fov = [], [], None, None

        def handle_pose(): nonlocal pose_se3; pose_se3 = self._parse_float_vector()
        def handle_k(): nonlocal intrinsics_k; intrinsics_k = self._parse_float_vector()
        def handle_dist(): nonlocal distortion; distortion = self._parse_float_vector()
        def handle_fov(): nonlocal fov; fov = self._parse_number_value()

        handlers = {
            Keyword.POSE_SE3: handle_pose, Keyword.INTRINSICS_K: handle_k,
            Keyword.DISTORTION: handle_dist, Keyword.FOV: handle_fov
        }

        self.consume_type(TokenType.LBRACE, "'{' to open frame")
        while not self.match_type(TokenType.RBRACE) and not self.match_type(TokenType.EOF):
            tok = self.current()
            if tok.type == TokenType.KEYWORD and tok.keyword in handlers:
                self.advance(); self.consume_type(TokenType.COLON, f"':' after {tok.value}")
                handlers[tok.keyword]()
                self.consume_type(TokenType.SEMICOLON, f"';' after {tok.value}")
            else:
                self.advance()

        end_tok = self.consume_type(TokenType.RBRACE, "'}' to close frame")
        return CameraFrameNode(
            start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset,
            frame_id_tok.value, pose_se3, intrinsics_k, distortion, fov
        )

    def _parse_entity(self) -> EntityNode:
        start_tok = self.consume_keyword(Keyword.ENTITY)
        entity_id_tok = self.consume_type(TokenType.STRING, "entity ID string")
        self.consume_type(TokenType.LBRACE, "'{' to open entity block")

        semantic, spatial, material, physics = None, None, None, None
        affordances, relationships, uncertainty, dynamics = None, None, None, None

        while not self.match_type(TokenType.RBRACE) and not self.match_type(TokenType.EOF):
            tok = self.current()
            if tok.type == TokenType.KEYWORD:
                if tok.keyword == Keyword.SEMANTIC: semantic = self._parse_semantic_comp()
                elif tok.keyword == Keyword.SPATIAL: spatial = self._parse_spatial_comp()
                elif tok.keyword == Keyword.MATERIAL: material = self._parse_material_comp()
                elif tok.keyword == Keyword.PHYSICS: physics = self._parse_physics_comp()
                elif tok.keyword == Keyword.AFFORDANCES: affordances = self._parse_affordance_comp()
                elif tok.keyword == Keyword.RELATIONSHIPS: relationships = self._parse_relationship_comp()
                elif tok.keyword == Keyword.UNCERTAINTY: uncertainty = self._parse_uncertainty_comp()
                elif tok.keyword == Keyword.DYNAMICS: dynamics = self._parse_dynamics_comp()
                else: self.advance()
            else:
                self.advance()

        end_tok = self.consume_type(TokenType.RBRACE, "'}' to close entity block")

        # Immutable Node Instantiation
        return EntityNode(
            line=start_tok.line, column=start_tok.column,
            start_offset=start_tok.start_offset, end_offset=end_tok.end_offset,
            entity_id=entity_id_tok.value, semantic=semantic, spatial=spatial,
            material=material, physics=physics, affordances=affordances,
            relationships=relationships, uncertainty=uncertainty, dynamics=dynamics
        )

    # --- Component Parsers using Refactored Helper ---

    def _parse_semantic_comp(self) -> SemanticComponentNode:
        start_tok = self.current()
        label, class_id, confidence = "", 0, 1.0

        def h_label(): nonlocal label; label = self.consume_type(TokenType.STRING, "label").value
        def h_cid(): nonlocal class_id; class_id = self.consume_type(TokenType.INTEGER, "class_id").value
        def h_conf(): nonlocal confidence; confidence = self._parse_number_value()

        end_tok = self._parse_component_block(
            Keyword.SEMANTIC, {Keyword.LABEL: h_label, Keyword.CLASS_ID: h_cid, Keyword.CONFIDENCE: h_conf}
        )
        return SemanticComponentNode(start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset, label, class_id, confidence)

    def _parse_spatial_comp(self) -> SpatialComponentNode:
        start_tok = self.current()
        bbox_min, bbox_max, transform_matrix, sdf_ref = [], [], None, None

        def h_min(): nonlocal bbox_min; bbox_min = self._parse_float_vector()
        def h_max(): nonlocal bbox_max; bbox_max = self._parse_float_vector()
        def h_mat(): nonlocal transform_matrix; transform_matrix = self._parse_float_vector()
        def h_sdf(): nonlocal sdf_ref; sdf_ref = self.consume_type(TokenType.STRING, "sdf_ref").value

        end_tok = self._parse_component_block(
            Keyword.SPATIAL, {Keyword.BBOX_MIN: h_min, Keyword.BBOX_MAX: h_max, Keyword.TRANSFORM_MATRIX: h_mat, Keyword.SDF_LATENT_REF: h_sdf}
        )
        return SpatialComponentNode(start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset, bbox_min, bbox_max, transform_matrix, sdf_ref)

    def _parse_material_comp(self) -> MaterialComponentNode:
        start_tok = self.current()
        base_type, roughness, metallic, albedo_rgb, normal_ref = "generic", 0.5, 0.0, None, None

        def h_base(): nonlocal base_type; base_type = self.consume_type(TokenType.STRING, "base_type").value
        def h_rough(): nonlocal roughness; roughness = self._parse_number_value()
        def h_metal(): nonlocal metallic; metallic = self._parse_number_value()
        def h_alb(): nonlocal albedo_rgb; albedo_rgb = self._parse_float_vector()
        def h_norm(): nonlocal normal_ref; normal_ref = self.consume_type(TokenType.STRING, "normal_ref").value

        end_tok = self._parse_component_block(
            Keyword.MATERIAL, {Keyword.BASE_TYPE: h_base, Keyword.ROUGHNESS: h_rough, Keyword.METALLIC: h_metal, Keyword.ALBEDO_RGB: h_alb, Keyword.NORMAL_MAP_REF: h_norm}
        )
        return MaterialComponentNode(start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset, base_type, roughness, metallic, albedo_rgb, normal_ref)

    def _parse_physics_comp(self) -> PhysicsComponentNode:
        start_tok = self.current()
        mass_kg, friction, is_static, restitution = 1.0, 0.5, False, None

        def h_mass(): nonlocal mass_kg; mass_kg = self._parse_number_value()
        def h_fric(): nonlocal friction; friction = self._parse_number_value()
        def h_static(): nonlocal is_static; is_static = self.consume_type(TokenType.BOOLEAN, "boolean").value
        def h_rest(): nonlocal restitution; restitution = self._parse_number_value()

        end_tok = self._parse_component_block(
            Keyword.PHYSICS, {Keyword.MASS_KG: h_mass, Keyword.FRICTION: h_fric, Keyword.IS_STATIC: h_static, Keyword.RESTITUTION: h_rest}
        )
        return PhysicsComponentNode(start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset, mass_kg, friction, is_static, restitution)

    def _parse_affordance_comp(self) -> AffordanceComponentNode:
        start_tok = self.current()
        actions, max_load_kg = [], None

        def h_actions(): nonlocal actions; actions = self._parse_string_list()
        def h_load(): nonlocal max_load_kg; max_load_kg = self._parse_number_value()

        end_tok = self._parse_component_block(
            Keyword.AFFORDANCES, {Keyword.ACTIONS: h_actions, Keyword.MAX_LOAD_KG: h_load}
        )
        return AffordanceComponentNode(start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset, actions, max_load_kg)

    def _parse_relationship_comp(self) -> RelationshipComponentNode:
        start_tok = self.consume_keyword(Keyword.RELATIONSHIPS)
        self.consume_type(TokenType.LBRACE, "'{'")
        relations: List[RelationPairNode] = []

        while not self.match_type(TokenType.RBRACE) and not self.match_type(TokenType.EOF):
            tok = self.current()
            if tok.type == TokenType.KEYWORD and tok.keyword in {
                Keyword.SUPPORTED_BY, Keyword.CONTAINS, Keyword.ADJACENT_TO, Keyword.BLOCKS_PATH, Keyword.ATTACHED_TO
            }:
                rel_type = tok.keyword
                rel_tok = self.advance()
                self.consume_type(TokenType.COLON, "':'")
                target_tok = self.consume_type(TokenType.STRING, "target entity string")
                self.consume_type(TokenType.SEMICOLON, "';'")
                relations.append(RelationPairNode(rel_tok.line, rel_tok.column, rel_tok.start_offset, target_tok.end_offset, rel_type, target_tok.value))
            else:
                self.advance()

        end_tok = self.consume_type(TokenType.RBRACE, "'}'")
        return RelationshipComponentNode(start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset, relations)

    def _parse_uncertainty_comp(self) -> UncertaintyComponentNode:
        start_tok = self.current()
        aleatoric, epistemic, is_inferred = 0.0, 0.0, None

        def h_aleat(): nonlocal aleatoric; aleatoric = self._parse_number_value()
        def h_epist(): nonlocal epistemic; epistemic = self._parse_number_value()
        def h_inf(): nonlocal is_inferred; is_inferred = self.consume_type(TokenType.BOOLEAN, "boolean").value

        end_tok = self._parse_component_block(
            Keyword.UNCERTAINTY, {Keyword.ALEATORIC_NOISE: h_aleat, Keyword.EPISTEMIC_RISK: h_epist, Keyword.IS_INFERRED: h_inf}
        )
        return UncertaintyComponentNode(start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset, aleatoric, epistemic, is_inferred)

    def _parse_dynamics_comp(self) -> DynamicsComponentNode:
        start_tok = self.current()
        linear_vel, angular_vel = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        def h_lin(): nonlocal linear_vel; linear_vel = self._parse_float_vector()
        def h_ang(): nonlocal angular_vel; angular_vel = self._parse_float_vector()

        end_tok = self._parse_component_block(
            Keyword.DYNAMICS, {Keyword.LINEAR_VELOCITY: h_lin, Keyword.ANGULAR_VELOCITY: h_ang}
        )
        return DynamicsComponentNode(start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset, linear_vel, angular_vel)

    def _parse_environment(self) -> EnvironmentNode:
        start_tok = self.current()
        hdri_ref, ambient_intensity = "", 1.0

        def h_hdri(): nonlocal hdri_ref; hdri_ref = self.consume_type(TokenType.STRING, "hdri_ref string").value
        def h_amb(): nonlocal ambient_intensity; ambient_intensity = self._parse_number_value()

        end_tok = self._parse_component_block(
            Keyword.ENVIRONMENT, {Keyword.HDRI_REF: h_hdri, Keyword.AMBIENT_INTENSITY: h_amb}
        )
        return EnvironmentNode(start_tok.line, start_tok.column, start_tok.start_offset, end_tok.end_offset, hdri_ref, ambient_intensity)

    # --- Utility Vector & Primitive Parsers ---

    def _parse_number_value(self) -> float:
        tok = self.current()
        if tok.type in (TokenType.FLOAT, TokenType.INTEGER):
            self.advance()
            return float(tok.value)
        self._report_error(f"Expected number (float/int), got '{tok.value}'", tok)
        return 0.0

    def _parse_float_vector(self) -> List[float]:
        self.consume_type(TokenType.LBRACKET, "'[' to open vector")
        elements: List[float] = []

        if not self.match_type(TokenType.RBRACKET):
            elements.append(self._parse_number_value())
            while self.match_type(TokenType.COMMA):
                self.advance()
                elements.append(self._parse_number_value())

        self.consume_type(TokenType.RBRACKET, "']' to close vector")
        return elements

    def _parse_string_list(self) -> List[str]:
        self.consume_type(TokenType.LBRACKET, "'[' to open list")
        elements: List[str] = []

        if not self.match_type(TokenType.RBRACKET):
            elements.append(self.consume_type(TokenType.STRING, "string element").value)
            while self.match_type(TokenType.COMMA):
                self.advance()
                elements.append(self.consume_type(TokenType.STRING, "string element").value)

        self.consume_type(TokenType.RBRACKET, "']' to close list")
        return elements