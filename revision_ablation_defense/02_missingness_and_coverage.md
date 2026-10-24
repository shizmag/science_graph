# Missingness, availability, and service utility

## Coverage by arm

All numbers below are rebuilt from per-query records; complete 14-arm values are in `tables/coverage_by_arm.csv`.

| Arm | Success / 375 | Failed | Overall coverage | Answerable success / 300 | Designated-negative success / 75 | Shared successes with B6 | Success only in arm | Failed in both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B6 full | 296 | 79 | 78.93% | 248 | 48 | 296 | 0 | 79 |
| minus dynamic GraphExpander | 375 | 0 | 100% | 300 | 75 | 296 | 79 | 0 |
| minus parent deduplication | 347 | 28 | 92.53% | 293 | 54 | 296 | 51 | 28 |

B6 generation coverage is higher on answerable than designated-negative cases (82.67% vs 64.00%); no-expander coverage is 100% in both strata. Parent-dedup-off coverage is 97.67% answerable and 72.00% designated-negative. These are exact ID-derived counts, not just manifest totals.

## Estimands kept separate

1. **Conditional answer quality:** judge scores only among the same query IDs where both B6 and arm generated. Principal matched-quality table: N=296 for both large interventions. It says nothing directly about failed-case judge outcomes.
2. **Generation availability:** valid output / all 375 planned queries. No-expander changes availability from 78.93% to 100%; parent-dedup-off from 78.93% to 92.53%.
3. **Service-level utility (policy composite):** `U_q = I(generation available) × Semantic Accuracy of delivered answer`; an unavailable result contributes zero to service utility *by definition of this operational policy*. It is not a Semantic Accuracy score for a nonexistent answer, not a judge observation, and not a clinical utility measure. Means are 0.5302 for B6, 0.7601 no-expander (paired all-query delta +0.2299; query bootstrap 95% CI +0.1748 to +0.2849), and 0.7435 parent-dedup-off (+0.2133; CI +0.1682 to +0.2575). The conditional-quality means remain reported separately.

## Missingness sensitivity and interpretation

All 985 missing generations are attributable to serialized prompt overflow beyond the 12,000-token cap. None were converted to judge zeros. The paired complete-case comparison is informative about the same 296 cases that both arms served; paired resampling preserves query identity and does not treat the two arms as independent. It is not representative by construction of all 375. Failures are arm-dependent, so MCAR is not plausible: B6 fails 79 cases, while no-expander fails none and parent-dedup-off fails 28. Within B6, answerability coverage differs by 18.67 pp (82.67% vs 64%). Failure prompt-token counts average 33,132 in B6 and 33,324 in parent-dedup-off; successful prompts average 5,833 and 2,802 respectively. Mean question character counts show only a modest B6 difference (fail 140.9 vs success 117.4), while designated-negative prompts/questions are substantially longer overall. Character length is a coarse proxy, not a causal failure predictor.

Without imputing judge scores, worst/best potential-score bounds assume each unavailable potential Semantic Accuracy lies anywhere in [0,1]. All-query `arm − B6` bounds are +0.27 to +28.80 pp (parent-dedup-off) and +1.93 to +22.99 pp (no-expander). The bounds are deliberately wide and are not confidence intervals. Both remain above zero only under the observed frozen cohort and this simple [0,1] score-scale bound; they do not remove judge error or establish population generalization.

## Conclusion

The missingness limitation is transparent and useful within observed support without rerunning the large experiment. Report matched-success quality, full-cohort availability, and the explicitly policy-defined service utility separately. Do not claim missing answers were scored zero, do not call failures abstentions, and do not state that the quality estimand covers all 375 cases.
