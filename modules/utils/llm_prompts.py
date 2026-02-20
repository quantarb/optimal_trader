# Notebook-parity LLM prompt builders.
# Source-of-truth: notebook_refactored.ipynb
import pandas as pd

def build_llm_guardrail_prompt_from_results(
    *,
    as_of_date,
    results_df: pd.DataFrame,
    confidence_cutoff: float,
    llm_top_k: int = 15,
    known_events: str = "NONE",
    round_decimals: int = 2,
) -> str | None:

    if results_df is None or len(results_df) == 0:
        return None

    top_df = results_df.sort_values("combined_score", ascending=False).head(int(llm_top_k)).copy()

    def r(x):
        if pd.isna(x):
            return "None"
        return f"{float(x):.{int(round_decimals)}f}"

    lines = []
    for sym, row in top_df.iterrows():
        lines.append(
            f"{sym:<8} | "
            f"cls={r(row['pred_rf_cls_proba'])} | "
            f"reg={r(row['pred_rf_reg'])} | "
            f"ae={r(row['ae_familiarity'])} | "
            f"score={r(row['combined_score'])}"
        )

    model_table = "\n".join(lines)
    known_events_block = known_events.strip() if known_events else "NONE"

    prompt = f"""
You are an independent investment analyst.

DATE:
{as_of_date}

Below are stock candidates selected by a quantitative model.
The model used technical indicators, fundamental numeric features, and macro features.

You are NOT constrained by the model.
You may use:
- valuation reasoning
- business quality assessment
- sector and macro trends
- earnings timing
- breaking news
- sentiment
- geopolitical risk
- any other relevant information

Your job is to independently evaluate whether these look like GOOD investment/trade candidates right now.

You may:
- Agree with the model
- Partially agree
- Completely disagree
- Identify better relative opportunities among the list

Provide your own conviction ranking.

OUTPUT STRICT JSON ONLY (no extra commentary):

{{
  "as_of": "{as_of_date}",
  "overall_view": "1-3 sentence market commentary",
  "top_conviction": ["TICKER1","TICKER2"],
  "avoid": ["TICKER3","TICKER4"],
  "decisions": [
    {{
      "symbol": "TICKER",
      "verdict": "STRONG_BUY|BUY|NEUTRAL|AVOID",
      "confidence": 0.0,
      "reasoning": "2-4 sentence independent investment thesis"
    }}
  ]
}}

MODEL OUTPUTS (Top {llm_top_k} by combined score):
symbol   | cls_prob | reg_score | familiarity | combined_score
-------------------------------------------------------------
{model_table}

USER-KNOWN CONTEXT:
{known_events_block}
""".strip()

    return prompt



# ============================================================
# 1) MAIN PREDICTION FUNCTION
# ============================================================
