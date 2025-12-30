import re

import pytest

from logic_utils import (
    check_balance,
    auto_fix,
    postprocess,
    normalize_expression,
    verify_equivalence,
)


def test_check_balance_basic():
    assert check_balance("(p & q) -> r") is True
    assert check_balance("((p)") is False
    assert check_balance(")p(") is False


def test_auto_fix_balances_parentheses():
    fixed = auto_fix("(p & q -> r")
    assert check_balance(fixed)
    # 不应引入重复空格
    assert "  " not in fixed


def test_normalize_expression_spacing():
    s = normalize_expression(" ~ ( p|q )  ->r ")
    # 箭头两侧单空格
    assert " -> " in s
    # & | 两侧单空格（此处为 |）
    assert "|" in s
    assert "  " not in s
    # 括号内外不应有额外空格
    assert "( " not in s and " )" not in s


def test_postprocess_makes_balanced_and_clean():
    expr = "((p|q) -> r) )"
    out = postprocess(expr)
    assert check_balance(out)
    assert "  " not in out


def test_verify_equivalence_contrapositive():
    assert verify_equivalence("p -> q", "~q -> ~p") is True
    assert verify_equivalence("(p & q) -> r", "~r -> ~(p & q)") is True
