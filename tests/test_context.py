# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for scoped FLContext (parent/child)."""
from erdos_fl import FLContext, RunEngine


def test_child_inherits_from_parent():
    parent = FLContext()
    parent.set_prop("a", 1)
    child = parent.new_child()
    assert child.get_prop("a") == 1
    assert "a" in child


def test_child_write_does_not_leak_to_parent():
    parent = FLContext()
    parent.set_prop(FLContext.CURRENT_CLIENT, "site-1")
    child = parent.new_child()
    child.set_prop(FLContext.CURRENT_CLIENT, "site-2")
    assert child.get_prop(FLContext.CURRENT_CLIENT) == "site-2"
    assert parent.get_prop(FLContext.CURRENT_CLIENT) == "site-1"


def test_sibling_scopes_are_isolated():
    parent = FLContext()
    a = parent.new_child()
    b = parent.new_child()
    a.set_prop("x", "a")
    b.set_prop("x", "b")
    assert a.get_prop("x") == "a"
    assert b.get_prop("x") == "b"
    assert parent.get_prop("x") is None


def test_engine_accessor():
    ctx = FLContext()
    assert ctx.get_engine() is None
    engine = RunEngine()
    engine.attach(ctx)
    assert ctx.get_engine() is engine
    assert ctx.new_child().get_engine() is engine
