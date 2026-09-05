"""The timing gate must not mistake a non-entering JIT for a working one."""

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "grain_jit_check", Path(__file__).resolve().parents[1] / "grain-jit-check.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
INTEGER = "tight integer loop"


def run(fields):
    return CHECK.Run("vm=jit jit=consulted jitcodes=10", {}, {INTEGER: fields}, "")


class CompiledEntryTests(unittest.TestCase):
    def test_compiling_and_entering_satisfies_the_milestone(self):
        self.assertEqual(
            CHECK.compiled_entry_failures(run({"compiled": "1", "entries": "24"}), [INTEGER]), []
        )

    def test_compilation_without_entry_is_not_enough(self):
        failures = CHECK.compiled_entry_failures(run({"compiled": "1", "entries": "0"}), [INTEGER])
        self.assertEqual(len(failures), 1)
        self.assertIn("entries=0", failures[0])

    def test_entry_does_not_replace_a_compilation_observation(self):
        failures = CHECK.compiled_entry_failures(run({"compiled": "0", "entries": "24"}), [INTEGER])
        self.assertEqual(len(failures), 1)
        self.assertIn("compiled=0", failures[0])

    def test_missing_and_malformed_counters_fail_closed(self):
        for fields in ({}, {"compiled": "bad", "entries": "-1"}):
            with self.subTest(fields=fields):
                self.assertEqual(len(CHECK.compiled_entry_failures(run(fields), [INTEGER])), 2)

    def test_another_case_cannot_supply_the_integer_loop_counters(self):
        observed = CHECK.Run("jitcodes=10", {}, {"float arithmetic": {"compiled": "1", "entries": "24"}}, "")
        self.assertEqual(len(CHECK.compiled_entry_failures(observed, [INTEGER])), 2)

    def test_an_explicit_other_case_does_not_claim_the_integer_milestone(self):
        self.assertEqual(CHECK.compiled_entry_failures(run({}), ["float arithmetic"]), [])

    def test_counter_parser_retains_actual_entry_and_compilation(self):
        observed = CHECK.Run.parse(
            "build: vm=jit jit=consulted jitcodes=10\n"
            "[jit] tight integer loop: compiled=1 entries=24 aborted=0 reasons=-\n"
        )
        self.assertEqual(observed.jitcodes, 10)
        self.assertEqual(CHECK.compiled_entry_failures(observed, [INTEGER]), [])

    def test_check_fails_even_when_non_entering_jit_is_fast_and_stable(self):
        plain = CHECK.Run("vm=plain jit=absent", {}, {}, "")
        jit = run({"compiled": "0", "entries": "0"})
        for observed in (plain, jit):
            observed.cases[INTEGER] = {"walker": 10.0, "vm": 5.0, "speedup": 2.0, "spread": 0.0}
        with (
            mock.patch.object(CHECK, "measure", side_effect=[plain, jit]),
            mock.patch("sys.argv", ["grain-jit-check.py", "--check", "--no-build", "--case", INTEGER]),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as stopped,
        ):
            CHECK.main()
        self.assertEqual(stopped.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
