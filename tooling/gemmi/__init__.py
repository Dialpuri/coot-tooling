"""Single-session MMDB → gemmi port of a function and its test.

The agent emits `function.hh` (+ optional `function.cc`) and `test.cc`
together in one session so the ported function's signature and the test's
call site agree by construction. The test's EXPECT_* assertions stay frozen
— they are the correctness oracle carried over from the MMDB test.
"""
