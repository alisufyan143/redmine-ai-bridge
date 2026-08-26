"""Unit tests for Phase 2: Ruby AST Parsing & Extraction."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock
import pytest

from graph_builder.ast_parser import RubyASTExtractor


@pytest.fixture
def extractor() -> RubyASTExtractor:
    """Fixture providing a clean RubyASTExtractor instance."""
    return RubyASTExtractor()


def test_extract_entities_valid_ruby(extractor: RubyASTExtractor) -> None:
    """Test 1: Valid Ruby string with class, method, and require statements."""
    ruby_code = """
    require 'json'
    require_relative 'models/base_issue'

    class IssueNotifier < ApplicationController
      def send_notification(user_id, message)
        puts "#{user_id}: #{message}"
      end

      def self.batch_notify(users)
        users.each { |u| puts u }
      end
    end
    """
    result = extractor.extract_entities(ruby_code)

    assert "IssueNotifier" in result["classes"]
    assert "send_notification" in result["methods"]
    assert "batch_notify" in result["methods"]
    assert "json" in result["requires"]
    assert "models/base_issue" in result["requires"]


def test_extract_entities_syntax_error_recovery(
    extractor: RubyASTExtractor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test 2: Syntactically invalid Ruby code recovers and extracts valid entities."""
    broken_ruby_code = """
    require 'securerandom'

    class ValidHeaderClass
      def valid_method
        1 + 1
      end
    end

    # Corrupted syntax block missing parentheses and end tags
    class CorruptedClass
      def broken_method(
        unclosed_expression = [1, 2,

    class RecoveredClass
      def recovered_method
        true
      end
    end
    """
    with caplog.at_level(logging.WARNING):
        result = extractor.extract_entities(broken_ruby_code)

    assert "securerandom" in result["requires"]
    assert "ValidHeaderClass" in result["classes"]
    assert "valid_method" in result["methods"]
    assert "RecoveredClass" in result["classes"]
    assert "recovered_method" in result["methods"]

    # Verify that a warning was logged for skipping syntax ERROR node
    warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("SYNTAX_ERROR_NODE_SKIPPED" in str(getattr(r, "extra_payload", {})) or "ERROR" in r.message for r in warning_logs)


def test_extract_entities_recursion_depth_limit_mock(
    extractor: RubyASTExtractor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test 3: Mocked AST node tree exceeding recursion depth halts safely."""
    # Construct a chain of mock AST nodes exceeding max_depth (e.g. 25 levels deep)
    root_mock = MagicMock()
    root_mock.type = "program"
    root_mock.start_byte = 0
    root_mock.end_byte = 100

    current = root_mock
    for level in range(25):
        child = MagicMock()
        child.type = "class" if level % 2 == 0 else "method"
        name_mock = MagicMock()
        name_mock.text = f"Entity_{level}".encode("utf-8")
        child.child_by_field_name.return_value = name_mock
        child.children = []
        current.children = [child]
        current = child

    entities: dict[str, list[str]] = {
        "classes": [],
        "methods": [],
        "requires": [],
    }

    with caplog.at_level(logging.WARNING):
        extractor._traverse(root_mock, entities, current_depth=0, max_depth=5)

    # Traversal should have halted at max_depth=5 and logged warning
    assert len(entities["classes"]) + len(entities["methods"]) <= 6

    depth_warnings = [
        r for r in caplog.records
        if "MAX_RECURSION_DEPTH_EXCEEDED" in str(getattr(r, "extra_payload", {}))
        or "maximum recursion depth" in r.message
    ]
    assert len(depth_warnings) > 0


def test_extract_entities_empty_input(extractor: RubyASTExtractor) -> None:
    """Test empty source code returns empty structures gracefully."""
    result = extractor.extract_entities("")
    assert result == {"classes": [], "methods": [], "requires": []}

    result_whitespace = extractor.extract_entities("   \n\t  ")
    assert result_whitespace == {"classes": [], "methods": [], "requires": []}
