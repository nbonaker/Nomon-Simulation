"""Historical QuickClick data/analysis extracted from evaluation_oneclick_phrase_audit.

No simulator execution. Saved result semantics are preserved.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from User_Simulation.evaluation.historical_oneclick.data_utils import TEXT_DATA_ROOT

def load_real_phrase_candidates(data_root: Path = TEXT_DATA_ROOT) -> pd.DataFrame:
    rows = []
    for phrase_path in sorted(data_root.glob("user_*_text_phrase_data_clean.csv")):
        user_id = phrase_path.name.removeprefix("user_").removesuffix("_text_phrase_data_clean.csv")
        phrase_df = pd.read_csv(phrase_path)
        for phrase_text in phrase_df["Phrase Text"].dropna().astype(str):
            rows.append({"Phrase Text": phrase_text.strip(), "Source User ID": user_id})
    candidates = pd.DataFrame(rows)
    grouped = (
        candidates.groupby("Phrase Text", sort=True)["Source User ID"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .reset_index()
    )
    grouped.insert(0, "Phrase ID", [f"real_{index:03d}" for index in range(1, len(grouped) + 1)])
    return grouped
