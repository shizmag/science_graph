"""Offline regression tests for the R1 post-run audit primitives."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_r1_mlx.py"
spec = importlib.util.spec_from_file_location("audit_r1_mlx", SCRIPT)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditMathTests(unittest.TestCase):
    def test_exact_mcnemar_balanced_discordance(self):
        result = audit.exact_mcnemar([True, False, True, False], [False, True, True, False])
        self.assertEqual(result["on_success_off_failure"], 1)
        self.assertEqual(result["on_failure_off_success"], 1)
        self.assertEqual(result["discordant"], 2)
        self.assertEqual(result["exact_two_sided_p"], 1.0)

    def test_exact_mcnemar_zero_discordance(self):
        result = audit.exact_mcnemar([True, True], [True, True])
        self.assertEqual(result["discordant"], 0)
        self.assertEqual(result["exact_two_sided_p"], 1.0)

    def test_bootstrap_is_fixed_seed_and_degenerate_ci_is_exact(self):
        values = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(audit.bootstrap(values, seed=123, n=1000), audit.bootstrap(values, seed=123, n=1000))
        self.assertEqual(audit.bootstrap([0.0, 0.0], seed=123, n=1000), (0.0, 0.0))
