"""AST extraction module for parsing Ruby codebases via tree-sitter."""

from __future__ import annotations

import logging
from typing import Any
import tree_sitter
import tree_sitter_ruby

from utils.telemetry import get_json_logger

logger: logging.Logger = get_json_logger("graph_builder.ast_parser")


class RubyASTExtractor:
    """Extracts structural entities (classes, methods, and require dependencies) from Ruby source code."""

    def __init__(self) -> None:
        self.language: tree_sitter.Language = tree_sitter.Language(tree_sitter_ruby.language())
        self.parser: tree_sitter.Parser = tree_sitter.Parser(self.language)

    def extract_entities(self, source_code: str, max_depth: int = 20) -> dict[str, list[str]]:
        """Parses Ruby source code and extracts classes, methods, and requires.

        Guards against syntax errors and deep recursion.
        """
        entities: dict[str, list[str]] = {
            "classes": [],
            "methods": [],
            "requires": [],
        }

        if not source_code or not source_code.strip():
            return entities

        source_bytes: bytes = source_code.encode("utf-8", errors="replace")

        try:
            tree = self.parser.parse(source_bytes)
        except Exception as exc:
            logger.error(
                "Tree-sitter parser failed unexpectedly.",
                extra={"extra_payload": {"error": str(exc)}},
            )
            return entities

        if tree.root_node is not None:
            self._traverse(tree.root_node, entities, current_depth=0, max_depth=max_depth)

        return entities

    def _traverse(
        self,
        node: Any,
        entities: dict[str, list[str]],
        current_depth: int,
        max_depth: int,
    ) -> None:
        """Recursively traverses AST nodes up to max_depth, recovering gracefully from ERROR nodes."""
        if current_depth > max_depth:
            logger.warning(
                f"AST traversal reached maximum recursion depth limit ({max_depth}); skipping deeper nodes.",
                extra={
                    "extra_payload": {
                        "event": "MAX_RECURSION_DEPTH_EXCEEDED",
                        "current_depth": current_depth,
                        "max_depth": max_depth,
                        "node_type": getattr(node, "type", "unknown"),
                    }
                },
            )
            return

        node_type: str = getattr(node, "type", "")

        # Gracefully log syntax error nodes and continue traversing recoverable children
        if node_type == "ERROR":
            logger.warning(
                "AST syntax ERROR node detected; traversing recoverable child nodes.",
                extra={
                    "extra_payload": {
                        "event": "SYNTAX_ERROR_NODE_DETECTED",
                        "start_byte": getattr(node, "start_byte", None),
                        "end_byte": getattr(node, "end_byte", None),
                    }
                },
            )
            for child in getattr(node, "children", []):
                self._traverse(child, entities, current_depth=current_depth + 1, max_depth=max_depth)
            return

        # Extract Class definitions
        if node_type == "class":
            class_name = self._extract_class_name(node)
            if class_name and class_name not in entities["classes"]:
                entities["classes"].append(class_name)

        # Extract Method definitions (standard and singleton)
        elif node_type in ("method", "singleton_method"):
            method_name = self._extract_method_name(node)
            if method_name and method_name not in entities["methods"]:
                entities["methods"].append(method_name)

        # Extract Require / Require_relative calls
        elif node_type in ("call", "command"):
            require_targets = self._extract_require_targets(node)
            for target in require_targets:
                if target and target not in entities["requires"]:
                    entities["requires"].append(target)

        # Recurse children
        children = getattr(node, "children", [])
        for child in children:
            self._traverse(child, entities, current_depth=current_depth + 1, max_depth=max_depth)

    def _extract_class_name(self, node: Any) -> str | None:
        """Extracts the class name from a class node."""
        if hasattr(node, "child_by_field_name"):
            name_node = node.child_by_field_name("name")
            if name_node is not None and getattr(name_node, "text", None):
                return name_node.text.decode("utf-8", errors="ignore").strip()

        for child in getattr(node, "children", []):
            if child.type in ("constant", "scope_resolution"):
                if getattr(child, "text", None):
                    return child.text.decode("utf-8", errors="ignore").strip()
        return None

    def _extract_method_name(self, node: Any) -> str | None:
        """Extracts the method name from a method or singleton_method node."""
        if hasattr(node, "child_by_field_name"):
            name_node = node.child_by_field_name("name")
            if name_node is not None and getattr(name_node, "text", None):
                return name_node.text.decode("utf-8", errors="ignore").strip()

        for child in getattr(node, "children", []):
            if child.type in ("identifier", "setter", "operator"):
                if getattr(child, "text", None):
                    return child.text.decode("utf-8", errors="ignore").strip()
        return None

    def _extract_require_targets(self, node: Any) -> list[str]:
        """Extracts imported files from require / require_relative invocations."""
        func_name: str | None = None
        for child in getattr(node, "children", []):
            if child.type == "identifier":
                func_name = child.text.decode("utf-8", errors="ignore").strip()
                break

        if func_name not in ("require", "require_relative"):
            return []

        args_node = None
        if hasattr(node, "child_by_field_name"):
            args_node = node.child_by_field_name("arguments") or node.child_by_field_name("argument_list")

        if args_node is None:
            for child in getattr(node, "children", []):
                if child.type in ("argument_list", "arguments"):
                    args_node = child
                    break

        targets: list[str] = []
        if args_node is not None:
            for arg in getattr(args_node, "children", []):
                if arg.type in ("string", "simple_symbol", "bare_string", "string_array"):
                    raw_val = arg.text.decode("utf-8", errors="ignore").strip("'\"`")
                    if raw_val:
                        targets.append(raw_val)
                elif arg.type not in ("(", ")", ",", "argument_list", "arguments"):
                    raw_val = arg.text.decode("utf-8", errors="ignore").strip("'\"`")
                    if raw_val:
                        targets.append(raw_val)
        return targets
