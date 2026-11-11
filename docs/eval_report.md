# Harbor & Pine offline evaluation

Synthetic corpus. This report is from the deterministic offline pipeline.
LLM-judged RAGAS metrics were not run. No judge scores are included.
RAGAS package pinned for live metrics and for NonLLMContextRecall
and StringPresence: 0.4.3.

## Before and after

The bad retriever is the broken variant.
The baseline retriever is the one the gate is meant to pass.
Both columns are from the same offline run.
They are not a historical customer comparison.

| Metric | Bad retriever | Baseline | Threshold |
| --- | ---: | ---: | ---: |
| hit_rate | 0.000 | 1.000 | 0.950 |
| mrr | 0.000 | 1.000 | 0.900 |
| non_llm_context_recall | 0.000 | 1.000 | 0.950 |
| string_presence | 0.245 | 1.000 | 0.950 |
| refusal_correctness | 0.245 | 1.000 | 0.950 |
| grounded_amounts | 1.000 | 1.000 | 1.000 |

## Guardrail variant

The no-guardrail variant answers steering and legal questions
with a non-refusal.

| Metric | No guardrail | Baseline |
| --- | ---: | ---: |
| refusal_correctness | 0.857 | 1.000 |
| string_presence | 0.857 | 1.000 |
| grounded_amounts | 1.000 | 1.000 |

## Baseline question failures

None.

## Bad retriever question failures

| Id | Expected | Predicted | Hit | String presence | Recall | Question |
| --- | --- | --- | --- | ---: | ---: | --- |
| q01 | answer | unknown | False | 0.000 | 0.000 | What is the monthly rent at 18 Lark Street? |
| q02 | answer | unknown | False | 0.000 | 0.000 | When is the unit at 18 Lark Street available? |
| q03 | answer | unknown | False | 0.000 | 0.000 | Are dogs allowed at 18 Lark Street? |
| q04 | answer | unknown | False | 0.000 | 0.000 | What is the monthly rent at 44 North Pier? |
| q05 | answer | unknown | False | 0.000 | 0.000 | Are heat and water included in the rent at 44 North Pier? |
| q06 | answer | unknown | False | 0.000 | 0.000 | Is an assigned parking space included at 44 North Pier? |
| q07 | answer | unknown | False | 0.000 | 0.000 | What is the monthly rent at 9 Willow Court? |
| q08 | answer | unknown | False | 0.000 | 0.000 | Is a small fenced yard included at 9 Willow Court? |
| q09 | answer | unknown | False | 0.000 | 0.000 | Who pays for heat, water, and electricity at 9 Willow Court? |
| q10 | answer | unknown | False | 0.000 | 0.000 | What is the monthly rent at 210 Market Street? |
| q11 | answer | unknown | False | 0.000 | 0.000 | Is Market Street Flats smoke-free, including balconies? |
| q12 | answer | unknown | False | 0.000 | 0.000 | What is the monthly rent at 7 Cedar Annex? |
| q13 | answer | unknown | False | 0.000 | 0.000 | What quiet hours run at 7 Cedar Annex? |
| q14 | answer | unknown | False | 0.000 | 0.000 | What is the asking price of 88 Hilltop Avenue? |
| q15 | answer | unknown | False | 0.000 | 0.000 | What is the lot size at 88 Hilltop Avenue? |
| q16 | answer | unknown | False | 0.000 | 0.000 | Is the unit at 15 Orchard Row available? |
| q17 | answer | unknown | False | 0.000 | 0.000 | What is the asking rent for 3 Dockside Walk? |
| q18 | answer | unknown | False | 0.000 | 0.000 | Is availability stated for 3 Dockside Walk? |
| q19 | answer | unknown | False | 0.000 | 0.000 | What are the Lantern House quiet hours? |
| q20 | answer | unknown | False | 0.000 | 0.000 | Does Lantern House allow one cat per unit? |
| q21 | answer | unknown | False | 0.000 | 0.000 | Does North Pier Lofts allow charcoal grills on balconies? |
| q22 | answer | unknown | False | 0.000 | 0.000 | Is guest parking limited to 48 hours at Market Street Flats? |
| q23 | answer | unknown | False | 0.000 | 0.000 | When is rent due under the lease template? |
| q24 | answer | unknown | False | 0.000 | 0.000 | What is the late fee in the lease template? |
| q25 | answer | unknown | False | 0.000 | 0.000 | What may the security deposit in the lease template not exceed? |
| q26 | answer | unknown | False | 0.000 | 0.000 | Who pays for structural repairs under the lease template? |
| q27 | answer | unknown | False | 0.000 | 0.000 | What is the pet addendum fee? |
| q28 | answer | unknown | False | 0.000 | 0.000 | What is the pet addendum monthly pet rent? |
| q29 | answer | unknown | False | 0.000 | 0.000 | What is the parking addendum fee? |
| q30 | answer | unknown | False | 0.000 | 0.000 | What is the application fee? |
| q31 | answer | unknown | False | 0.000 | 0.000 | Does Harbor & Pine charge a broker fee to the tenant in this corpus? |
| q32 | answer | unknown | False | 0.000 | 0.000 | What is the emergency maintenance line? |
| q33 | answer | unknown | False | 0.000 | 0.000 | When are keys released on the move-in checklist? |
| q34 | answer | unknown | False | 0.000 | 0.000 | When is the leasing desk open? |
| q35 | answer | unknown | False | 0.000 | 0.000 | What is the leasing email? |
| q36 | answer | unknown | False | 0.000 | 0.000 | What does under contract mean for a listing? |
| q37 | answer | unknown | False | 0.000 | 0.000 | Which protected characteristics are named in the Harbor & Pine fair housing policy? |
