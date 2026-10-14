# FreightTiger AI & Data Engineering Case Study
## Smart Assistant for Shipping Cost Anomaly Detection & Evidence-Grounded AI Explanations

---

## 1. Problem Statement & Objectives

Transportation costs can quietly creep up on freight routes over time. Sometimes, cost increases are driven by legitimate real-world operational disruptions (e.g., severe flooding, regional festival surcharges, nationwide fuel price hikes). Other times, cost spikes occur without valid justification and warrant immediate commercial review.

This repository implements a **production-ready, deterministic freight-cost anomaly detection and explanation system** satisfying all official FreightTiger 24-Hour Case Study requirements.

### Core Objectives:
1. **Deterministic Data Pipeline:** Group shipment records into Monday-to-Sunday weeks, derive standardized route keys (`Origin-Destination`), and compute weekly freight cost per tonne-kilometre ($\text{₹}/\text{tonne-km}$) using full internal precision.
2. **Precision Baseline Calculations:**
   - **Own History Baseline:** Trailing 8-week rolling average cost per tonne-km strictly prior to the current week (no look-ahead).
   - **Similar Routes (Peer) Baseline:** Average cost per tonne-km across all *other* routes sharing the same `route_type` (`Short`, `Medium`, `Long`) in that same week (strictly excluding the route itself).
3. **Deterministic Anomaly Detection:** Flag candidate routes where cost increases $\ge 20\%$ over historical or peer baselines.
4. **Anti-Hallucination Context Retrieval & Guardrails:** Match candidates against official context notes, strictly verifying corridor applicability and temporal windows, while rejecting non-justifications and distractors (e.g., maintenance notes with unaffected costs, or stable demand reports).
5. **Auditable Explanations & Accounting:** Deliver plain-English explanations adhering to the exact 8-column contract, with zero-cost offline determinism and auditable token/cost logging (`run_metrics.json`).

---

## 2. Input Files & Expected Locations

The system expects input CSV files located in `data/raw/`:

| File | Path | Description |
| :--- | :--- | :--- |
| `shipment_records.csv` | `data/raw/shipment_records.csv` | 2,940 shipment records across 7 routes over 104 weeks (Jan 2024 – Dec 2025). |
| `context_notes.csv` | `data/raw/context_notes.csv` | 10 real-world context notes containing positive disruptions and distractors. |
| `sample_output_format_v2.csv` | `data/raw/sample_output_format_v2.csv` | The contract schema reference and sample regression cases. |

*Note: Raw inputs are strictly treated as read-only. Original files are never modified or overwritten.*

---

## 3. Mathematical Formulations & Baselines

### A. Weekly Cost per Tonne-Kilometre ($\text{₹}/\text{tonne-km}$)
For a given route $r$ and week $w$:
$$\text{cost\_per\_tonne\_km}_{r, w} = \frac{\sum_{i \in (r, w)} \text{freight\_cost\_inr}_i}{\sum_{i \in (r, w)} (\text{quantity\_tonnes}_i \times \text{distance\_km}_i)}$$
- Internal precision: 64-bit floating point (`float64`).
- Presentation: Rounded to 2 decimal places in the final report.

### B. Trailing 8-Week Historical Baseline (`vs_own_history`)
- The rolling average of up to 8 prior weeks strictly before the current week ($w-8$ to $w-1$):
  $$\text{trailing\_hist}_{r, w} = \frac{1}{K} \sum_{k=1}^{K} \text{cost\_per\_tonne\_km}_{r, w-k} \quad (1 \le K \le 8)$$
- **Strict No Look-Ahead:** Current week $w$ and future weeks are strictly excluded.
- **Early Weeks ($K < 8$):** When fewer than 8 prior weeks exist, all available prior weeks are averaged without artificial padding or extrapolation. Week 1 is recorded as `N/A (no prior history)`.
- **Contract Formatting:** `+XX.X% vs this route's past average` (signed float, 1 decimal place).

### C. Same-Route-Type Peer Baseline (`vs_similar_routes`)
- The average cost per tonne-km in that *same week* across all *other* routes sharing the same `route_type` category:
  $$\text{peer\_avg}_{r, w} = \frac{1}{|\mathcal{P}_{r, w}|} \sum_{p \in \mathcal{P}_{r, w}} \text{cost\_per\_tonne\_km}_{p, w}$$
  where $\mathcal{P}_{r, w} = \{ r' \mid \text{route\_type}(r') = \text{route\_type}(r) \text{ and } r' \ne r \}$.
- **Strict Self-Exclusion:** The candidate route itself is never included in its own peer average.
- **Contract Formatting:** `+XX.X% vs similar-length routes this week` (signed float, 1 decimal place).

---

## 4. Configurable Anomaly Thresholds

A route-week qualifies as an **anomaly candidate** if:
$$\Delta_{\text{history}} \ge \text{threshold}_{\text{history}} \quad \text{OR} \quad \Delta_{\text{peer}} \ge \text{threshold}_{\text{peer}}$$

- **Default Values:** `--history-threshold 20.0` and `--peer-threshold 20.0`.
- **Derivation & Documentation:** Inferred from official sample case `Mumbai-Pune` (2025-09-15), which exhibits a peer deviation of **+23.6%** despite a historical increase of only **+9.2%**, yet is flagged as a candidate.
- **Missing Baseline Handling:** Missing baselines (e.g. Week 1 history) evaluate to `NaN` and are excluded from triggering decisions (never coerced to 0.0).

---

## 5. RAG Retrieval & Anti-Hallucination Guardrails

To prevent hallucination, candidate retrieval is separated from evidence validation:

### Guardrail Rules:
1. **Route Applicability:** Note must explicitly match candidate route corridor or specify `All Routes`.
2. **Temporal Window Boundaries:**
   - **`N001` (Chennai–Bangalore Flooding):** Documented disruption runs from **Feb 24 to Mar 8, 2025**. Roads were passable again by March 9. Only weeks `2025-02-24` and `2025-03-03` are valid. Week `2025-03-10` is after reopening and is marked unexplained.
   - **`N002` (Ahmedabad–Mumbai Festival Surcharge):** Valid strictly for the documented festival week `2025-01-20`.
   - **`N003` (Diesel Price Increase):** Applicable starting week `2025-05-05`.
3. **Strict Distractor Rejection:**
   - `N004`: Rejected (*"affected routes are not part of this dataset"*).
   - `N005`: Rejected (*"costs were not significantly affected"*).
   - `N006`: Rejected (*"stable demand... no major disruptions"*).
   - `N007`: Rejected (*"improved road conditions after resurfacing completed"*).
   - `N008`: Rejected (*"freight movement remained normal"*).
   - `N009`: Rejected (*"returned to normal conditions"*).
   - `N010`: Rejected (*"costs were absorbed without rate changes"*).
4. **Deterministic Verdicts:**
   - Valid supporting note found $\rightarrow$ `flagged = "No (justified)"`, `matched_note_id = "<NOTE_ID>"`.
   - No valid supporting note $\rightarrow$ `flagged = "Yes"`, `matched_note_id = ""`.

---

## 6. Execution Modes & LLM Configuration

The pipeline supports dual execution modes via a modular provider abstraction:

### A. Offline Mode (Default & Recommended)
- Executes 100% locally with **0 external API calls** and **$0.00 spend**.
- Generates deterministic, template-based explanations adhering to sample output formatting.
- Guarantee: **100% bit-for-bit reproducibility** across repeat runs.

### B. Optional Live LLM Mode
- Configurable for OpenAI (`gpt-4o-mini`), Groq (`llama-3.1-8b`), or OpenAI-compatible endpoints.
- Enforces `temperature=0.0` and structured JSON responses (`{"explanation": "..."}`).
- Guardrail response validator automatically rejects malformed output, invalid note IDs, or verdict contradictions, falling back safely to deterministic explanations.
- API keys are read securely via environment variables (`.env`). Secrets are never printed or logged.

---

## 7. Setup & Run Commands

### Installation
```powershell
# Clone or navigate to the repository
cd d:\OneDrive\Desktop\AI_Tiger

# Install pinned dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline
```powershell
# 1. Standard run (Generates final submission CSV with 20 candidate anomalies)
python run_pipeline.py

# 2. Optional: Run with all 728 route-weeks exported (normal weeks labeled 'No')
python run_pipeline.py --include-all-weeks

# 3. Optional: Run with custom anomaly thresholds
python run_pipeline.py --history-threshold 25.0 --peer-threshold 25.0

# 4. Optional: Run with Live LLM provider (requires API key in .env)
python run_pipeline.py --provider openai --model gpt-4o-mini
```

### Running the Diagnostic Verification Scripts
```powershell
# Phase 1 Diagnostics (Weekly aggregation & baseline checks)
python diagnose_phase1.py

# Phase 2 Diagnostics (Candidate anomalies & triggers)
python diagnose_phase2.py

# Phase 3 Diagnostics (Evidence validation & distractor rejections)
python diagnose_phase3.py

# Phase 4 Diagnostics (Explanations & token/cost accounting)
python diagnose_phase4.py
```

### Running Automated Tests
```powershell
# Run the complete test suite (59 unit + integration tests)
pytest -v
```

---

## 8. Output Contract & Sample Interpretation

The generated submission CSV is written to:
`data/output/freight_anomaly_report.csv`

### Exact Eight-Column Contract:
1. `route` (e.g., `Delhi-Jaipur`)
2. `week_of` (e.g., `2024-11-11`)
3. `cost_per_tonne_km` (e.g., `4.17`)
4. `vs_own_history` (e.g., `+35.5% vs this route's past average`)
5. `vs_similar_routes` (e.g., `+21.0% vs similar-length routes this week`)
6. `flagged` (`Yes` or `No (justified)`)
7. `matched_note_id` (e.g., `N002`, or empty `""` when unexplained)
8. `reason` (Plain-English explanation)

### Official Sample Rows Verification:

| Route | Week Of | Cost/t-km | vs_own_history | vs_similar_routes | Flagged | Note ID | Explanation Excerpt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Delhi-Jaipur** | `2024-11-11` | `4.17` | `+35.5%` | `+21.0%` | `Yes` | *blank* | *"No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."* |
| **Ahmedabad-Mumbai** | `2025-01-20` | `3.29` | `+29.5%` | `+22.5%` | `No (justified)` | `N002` | *"Matches note N002 dated 2025-01-20: a regional festival week drove a temporary surcharge on this corridor..."* |
| **Mumbai-Pune** | `2025-09-15` | `3.98` | `+9.2%` | `+23.6%` | `Yes` | *blank* | *"The closest note (N006, 2025-09-22) mentions stable demand with no major disruptions -- it does not describe a reason..."* |

---

## 9. RFC 4180 Quoting Safeguard

In standard CSV format, explanations containing commas (such as `"(N006, 2025-09-22)"`) must be enclosed in double quotes. Our exporter strictly applies `csv.QUOTE_MINIMAL` to guarantee that the output parses cleanly into exactly 8 columns across all standard CSV parsers (resolving the tokenization error present in the provided sample format).

---

## 10. Test Suite & Coverage

The test suite contains **59 automated tests** across 8 modules:
- `tests/test_loader.py`: Schema validation, positive numeric checks, invalid date handling, and duplicate row/ID detection.
- `tests/test_weekly_metrics.py`: Monday boundaries, cost per tonne-km formula, and zero-division protection.
- `tests/test_baselines.py`: Trailing 8-week history, strict no-look-ahead isolation, and peer route exclusion.
- `tests/test_anomaly_detector.py`: Threshold boundaries, single vs dual triggers, negative rates, and missing baselines.
- `tests/test_guardrails.py`: Valid window checks (N001, N002, N003), distractor rejections (N004–N010), and route mismatches.
- `tests/test_llm_explainer.py`: Explanation formatting, hallucination rejection, provider error fallbacks, and secret redaction.
- `tests/test_sample_ground_truth.py`: Precision checks against official sample numerical baselines.
- `tests/test_pipeline_integration.py`: End-to-end execution, RFC 4180 parsing, and 3-run byte-for-byte reproducibility.

---

## 11. Known Structural Observations

1. **Short Route Peer Comparison:** The `Short` category contains only two routes (`Delhi-Jaipur` and `Mumbai-Pune`). Consequently, for short routes, the peer baseline is a 1-to-1 comparison against the single other short route. Shorter distances naturally have higher fixed loading costs per tonne-km.
2. **Context Note N010 in Explanations:** In later 2025 weeks for Mumbai-Pune, note N010 is cited in explanations as an *irrelevant nearby note* that was rejected (because costs were absorbed without rate changes), not as causal evidence. This follows the exact pattern demonstrated in official sample row 4 for note N006.
