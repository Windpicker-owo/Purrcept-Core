"""Exercise static result inference for ``perform`` and direct effect yields."""

from collections.abc import Generator
from typing import assert_type

from purrcept_core import AgentFlow, Effect, perform


class ReadCount(Effect[int]):
    """Typed effect whose successful result is an integer."""

    pass


class ReadName(Effect[str]):
    """Typed effect whose successful result is text."""

    pass


def typed_flow() -> AgentFlow[tuple[int, str]]:
    """Assert that ``perform`` preserves each effect's result parameter."""

    count = yield from perform(ReadCount())
    name = yield from perform(ReadName())
    assert_type(count, int)
    assert_type(name, str)
    return count, name


def direct_yield_flow() -> Generator[Effect[object], object, None]:
    """Show the wider send type inferred for a direct generator yield."""

    yield ReadCount()
