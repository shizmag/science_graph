import sys
import unittest
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from eligibility_registry import build_registry, classify_record, paired_eligibility, registry_from_historical_availability  # noqa: E402
from paired_protocol import make_pair, paired_call_budget, real_judge_call, score_with_fake_judge, validate_pair  # noqa: E402


class EligibilityTests(unittest.TestCase):
    def test_archived_and_verified_reconstruction(self):
        archived = classify_record({"query_id": "q1", "paper_baseline": "B3",
                                    "exact_archived_prompt": "archive/prompt.txt"})
        rebuilt = classify_record({"query_id": "q1", "paper_baseline": "B5",
                                  "reconstruction_verified": True, "query": "q", "golden_answer": "g",
                                  "candidate_answer": "a", "rubric": "r", "rendered_context": "c"})
        self.assertEqual(archived.status, "exact_archived")
        self.assertEqual(rebuilt.status, "verified_reconstructed")
        self.assertEqual((archived.code_baseline, rebuilt.code_baseline), ("B4", "B6"))

    def test_chunks_or_scores_are_never_complete_prompt(self):
        for extra in ({"retrieved_chunks": ["chunk"]}, {"historical_score": 0.8},
                      {"retrieved_chunks": ["chunk"], "query": "q", "candidate_answer": "a"}):
            row = classify_record({"query_id": "q1", "paper_baseline": "B3", **extra})
            self.assertEqual(row.status, "partial")

    def test_unavailable_and_paired_counts(self):
        rows = build_registry([
            {"query_id": "q1", "paper_baseline": "B3", "exact_archived_prompt": True},
            {"query_id": "q1", "paper_baseline": "B5", "reconstruction_verified": True,
             "query": "q", "golden_answer": "g", "candidate_answer": "a", "rubric": "r", "rendered_context": "c"},
            {"query_id": "q2", "paper_baseline": "B3"},
        ])
        summary = paired_eligibility(rows)
        self.assertEqual(summary["q1"]["paired_eligible"], "true")
        self.assertEqual(summary["q2"]["paired_eligible"], "false")
        self.assertEqual(summary["q2"]["B3"], "unavailable")

    def test_saved_score_table_is_partial_not_prompt_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "availability.csv"
            path.write_text("query_id,paper_baseline,code_baseline,historical_eval_detail_present,historical_score_value_present,metric\n"
                            "q1,B3,B4,True,True,semantic_accuracy\n"
                            "q1,B5,B6,False,True,semantic_accuracy\n", encoding="utf-8")
            rows = registry_from_historical_availability(path)
        self.assertEqual([row.status for row in rows], ["partial", "partial"])
        self.assertTrue(all("complete rendered prompt" in row.reason for row in rows))

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            build_registry([{"query_id": "q", "paper_baseline": "B3"}] * 2)


class PairedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.pair = make_pair(query_id="q1", query="same query", golden_answer="same ref",
                              generator_visible_evidence="same full generator context", rubric="same rubric",
                              b3_answer="answer 3", b5_answer="answer 5", token_budget=1024,
                              evidence_status="verified_reconstructed")

    def test_shared_fields_and_budget_are_identical(self):
        left, right = self.pair
        validate_pair(self.pair)
        self.assertEqual(left.query_id, right.query_id)
        self.assertEqual(left.token_budget, right.token_budget)
        self.assertEqual((left.query, left.golden_answer, left.generator_visible_evidence, left.rubric),
                         (right.query, right.golden_answer, right.generator_visible_evidence, right.rubric))
        self.assertNotEqual(left.candidate_answer, right.candidate_answer)

    def test_fake_judge_receives_both_sides(self):
        seen = []
        result = score_with_fake_judge(self.pair, lambda item: seen.append(item.paper_baseline) or 0.5)
        self.assertEqual(seen, ["B3", "B5"])
        self.assertEqual(result, {"B3": 0.5, "B5": 0.5})

    def test_real_call_is_hard_disabled(self):
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            real_judge_call(self.pair)

    def test_partial_retrieved_context_cannot_be_judged(self):
        with self.assertRaisesRegex(ValueError, "requires exact archived"):
            make_pair(query_id="q2", query="q", golden_answer="g", generator_visible_evidence="chunks only",
                      rubric="r", b3_answer="a", b5_answer="b", token_budget=100, evidence_status="partial")

    def test_two_judges_get_fixed_four_evaluation_budget(self):
        budget = paired_call_budget({"q1": self.pair}, judges=("judge_a", "judge_b"), output_token_budget_per_call=128)
        self.assertEqual(budget, {"paired_query_ids": 1, "judge_count": 2, "evaluations": 4,
                                  "input_token_budget": 4096, "output_token_budget": 512})

    def test_pair_mismatch_rejected(self):
        left, right = self.pair
        with self.assertRaises(ValueError):
            validate_pair((left, right.__class__(**{**right.__dict__, "token_budget": 5})))


if __name__ == "__main__":
    unittest.main()
