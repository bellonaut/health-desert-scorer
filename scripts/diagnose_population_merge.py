"""Diagnostics for population-to-LGA merge issues (phase 1, read-only)."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import geopandas as gpd
import pandas as pd


FEATURES_PATH = Path("data/processed/lga_features.csv")
POP_PATH = Path("data/raw/population_lga.csv")
BOUNDARY_PATH = Path("data/raw/lga_boundaries.geojson")
OUTPUT_PATH = Path("logs/population_merge_diagnostics.csv")


def _rename_from_candidates(df: pd.DataFrame, target: str, candidates: List[str]) -> pd.DataFrame:
    """Rename the first matching column (case-insensitive) to target."""

    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return df.rename(columns={lower_map[cand.lower()]: target})
    raise SystemExit(f"Missing {target} column; tried candidates {candidates}")


def normalize(series: pd.Series) -> pd.Series:
    """Lowercase, strip, drop punctuation, collapse whitespace."""

    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def top_matches(
    target: str,
    candidates: Iterable[Tuple[str, str, str]],
    n: int = 3,
) -> List[Tuple[str, str, float]]:
    """
    Return top N fuzzy matches for a target string.

    Parameters
    ----------
    target: normalized target string
    candidates: iterable of (state_name, lga_name, lga_norm)
    """

    scores = []
    for state, lga_raw, lga_norm in candidates:
        score = difflib.SequenceMatcher(None, target, lga_norm).ratio()
        scores.append((state, lga_raw, score))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:n]


CARDINAL = {"north", "south", "east", "west", "northeast", "southeast", "southwest", "northwest"}
PUNCT_PATTERN = re.compile(r"[-/]")


def _has_direction_tokens(name: str) -> bool:
    tokens = set(name.split())
    return bool(tokens & CARDINAL)


def _looks_truncation(a: str, b: str) -> bool:
    a_compact = a.replace(" ", "")
    b_compact = b.replace(" ", "")
    if a_compact.startswith(b_compact) or b_compact.startswith(a_compact):
        return True
    # off-by-one endings (e.g., awkanort vs awka north)
    return abs(len(a_compact) - len(b_compact)) <= 2 and (
        a_compact.startswith(b_compact[:-1]) or b_compact.startswith(a_compact[:-1])
    )


def _looks_abbreviation(a: str, b: str) -> bool:
    # Abbreviation if one side is much shorter but shares prefix
    a_compact = a.replace(" ", "")
    b_compact = b.replace(" ", "")
    shorter, longer = sorted([a_compact, b_compact], key=len)
    return len(shorter) <= 4 and longer.startswith(shorter)


def classify_cause(feature_state: str, feature_lga: str, candidate_state: str | None, candidate_lga: str | None) -> str:
    """Classify likely mismatch cause for reporting purposes."""

    if not candidate_lga:
        return "spelling error"

    f_norm = normalize(pd.Series([feature_lga])).iat[0]
    c_norm = normalize(pd.Series([candidate_lga])).iat[0]
    f_raw = feature_lga.lower()
    c_raw = candidate_lga.lower()

    # Missing cardinal suffix/prefix
    if _has_direction_tokens(f_norm) != _has_direction_tokens(c_norm):
        return "missing suffix/prefix"

    # Punctuation loss
    if PUNCT_PATTERN.search(feature_lga) or PUNCT_PATTERN.search(candidate_lga):
        compact_f = PUNCT_PATTERN.sub(" ", f_raw)
        compact_c = PUNCT_PATTERN.sub(" ", c_raw)
        if difflib.SequenceMatcher(None, compact_f, compact_c).ratio() >= 0.8:
            return "punctuation loss"

    # Truncation
    if _looks_truncation(f_norm, c_norm):
        return "truncation"

    # Abbreviation (state or LGA)
    if _looks_abbreviation(f_norm, c_norm) or (
        candidate_state and _looks_abbreviation(normalize(pd.Series([feature_state])).iat[0], normalize(pd.Series([candidate_state])).iat[0])
    ):
        return "abbreviation"

    return "spelling error"


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    population = pd.read_csv(POP_PATH)
    boundaries_raw = gpd.read_file(BOUNDARY_PATH)
    boundaries = _rename_from_candidates(
        boundaries_raw,
        "lga_name",
        ["lga_name", "lganame", "lga", "adm2_name", "adm2", "name", "lg_name"],
    )
    boundaries = _rename_from_candidates(
        boundaries,
        "state_name",
        ["state_name", "statename", "state", "adm1_name", "adm1", "name_1"],
    )

    for col in ["lga_name", "state_name"]:
        if col not in features:
            raise SystemExit(f"Missing {col} in features")
    for col in ["lga_name", "state_name", "population"]:
        if col not in population:
            raise SystemExit(f"Missing {col} in population file")

    # Normalize
    for df in (features, population, boundaries):
        df["state_norm"] = normalize(df["state_name"])
        df["lga_norm"] = normalize(df["lga_name"])
        df["key"] = df["state_norm"] + "__" + df["lga_norm"]

    # Basic coverage numbers
    keys_pop = set(population["key"])
    keys_feat = set(features["key"])
    match_rate = features["key"].isin(keys_pop).mean()

    print(f"Match rate (state+lga): {match_rate:.2%}")
    print(f"Keys only in FEATURES: {len(keys_feat - keys_pop)}")
    print(f"Keys only in POPULATION: {len(keys_pop - keys_feat)}")
    print(f"Boundary polygons: {len(boundaries)}")

    unmatched = features.loc[~features["key"].isin(keys_pop)].copy()
    pop_candidates_by_state = population.groupby("state_norm")

    diag_rows = []
    for _, row in unmatched.head(50).iterrows():
        state_norm = row["state_norm"]
        feature_state = row["state_name"]
        feature_lga = row["lga_name"]

        if state_norm in pop_candidates_by_state.groups:
            pop_group = pop_candidates_by_state.get_group(state_norm)
        else:
            # fallback to fuzzy state matches
            pop_group = population

        matches = top_matches(
            row["lga_norm"],
            pop_group[["state_name", "lga_name", "lga_norm"]].itertuples(index=False, name=None),
            n=3,
        )
        state_candidates = sorted(pop_group["state_name"].unique())
        best_state, best_lga, best_score = matches[0] if matches else (None, None, 0.0)
        cause = classify_cause(feature_state, feature_lga, best_state, best_lga)
        diag_rows.append(
            {
                "feature_state": feature_state,
                "feature_lga": feature_lga,
                "population_state_candidates": "; ".join(state_candidates[:3]),
                "top_matches": "; ".join([f"{s} | {l} ({score:.2f})" for s, l, score in matches]),
                "best_match_score": best_score,
                "cause_bucket": cause,
            }
        )

    diag_df = pd.DataFrame(diag_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(OUTPUT_PATH, index=False)

    print("\nMismatch sample (first 5 rows):")
    print(diag_df.head().to_string(index=False))
    print(f"\nDiagnostic CSV written to {OUTPUT_PATH}")
    print("Cause bucket counts:")
    print(diag_df["cause_bucket"].value_counts())


if __name__ == "__main__":
    main()
