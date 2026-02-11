# Universal Migration Planner: Test Prompts

These prompts demonstrate the capabilities of the Universal Solver to generate complex constraints from Natural Language.

## 1. Global Capacity & Period Limits
*   **Prompt:** "Limit total pods between Month 18 and 29 to 14000."
*   **CDL Feature:** `GET_POD_COUNT_FOR_PERIOD`, Iterator (if simple limits), Aggregation.

## 2. Mixed Cohort Sequencing (Modulo)
*   **Prompt:** "If Cohort C (Soak), it must start in a month where (Month % 3 == 2)."
*   **CDL Feature:** Arithmetic (`%`), Logic (`IMPLIES`).

## 3. Prioritization (Objectives)
*   **Prompt:** "Prioritize small families (Size < 2TB) to start as early as possible."
*   **CDL Feature:** `OBJECTIVE`, `MINIMIZE`, `filter` scope.

## 4. Exclusion & Blackout
*   **Prompt:** "Do not schedule any pods in Month 20, 24, or 28."
*   **CDL Feature:** `NOT IN`, List constants.

## 5. Conditional Logic
*   **Prompt:** "If Region is PHX, pods must start after Month 15."
*   **CDL Feature:** `IMPLIES`, `Filter`.

## 6. Iterative Ramp-up
*   **Prompt:** "Limit pods to 200 per month for every month."
*   **CDL Feature:** `iterator` (`for_each_month`).
