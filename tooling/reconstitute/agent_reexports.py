"""Single place that re-exports the gemmi agent's internal harness for the lift.

The phase-2 lift agent deliberately reuses the phase-1 gemmi agent's compile/run/
write tool handlers, block extraction, dep-resolution, and tool schemas rather than
duplicating hundreds of lines. Those live as module-private (`_`-prefixed) names in
`tooling.gemmi.agent`; importing them across the package is a conscious coupling, so
it is isolated here behind clean names. If the gemmi agent's internals change, this
is the only file to update.
"""
from __future__ import annotations

from ..gemmi.agent import (
    _make_tool_handlers as make_tool_handlers,
    _extract_blocks as extract_blocks,
    _dep_extra_includes as dep_extra_includes,
    _dep_extra_sources as dep_extra_sources,
    _COMPILE_TOOL,
    _RUN_TOOL,
    _GET_ERRORS_TOOL,
    _WRITE_FILE_TOOL,
    _READ_GEMMI_FILE_TOOL,
    _PATCH_GEMMI_FILE_TOOL,
)

# Tools the lift needs: write/compile/run + read/patch/errors. No DB lookups —
# the gemmi code already resolved every API, so the lift never needs to discover one.
LIFT_TOOLS = [
    _WRITE_FILE_TOOL,
    _COMPILE_TOOL,
    _RUN_TOOL,
    _GET_ERRORS_TOOL,
    _READ_GEMMI_FILE_TOOL,
    _PATCH_GEMMI_FILE_TOOL,
]

__all__ = [
    "make_tool_handlers",
    "extract_blocks",
    "dep_extra_includes",
    "dep_extra_sources",
    "LIFT_TOOLS",
]
