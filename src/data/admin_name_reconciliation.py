"""Deterministic reconciliation of population rows to canonical LGA boundaries."""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from src.data.spatial_ops import normalize_admin_name

LOGGER = logging.getLogger(__name__)

# Known state aliases observed in population extracts
STATE_ALIASES: Dict[str, str] = {
    "Federal Capital Territory": "FCT",
    "Nassarawa": "Nasarawa",
}

# Manual LGA name corrections (population value -> canonical boundary value)
_MANUAL_OVERRIDES_RAW: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("Abia", "Oboma Ngwa"): ("Abia", "Obi Nwga"),
    ("Abia", "Ohafia Abia"): ("Abia", "Ohafia"),
    ("Adamawa", "Girie"): ("Adamawa", "Girei"),
    ("Adamawa", "Teungo"): ("Adamawa", "Toungo"),
    ("Akwa Ibom", "UrueOffo"): ("Akwa Ibom", "Urue Offong/Oruko"),
    ("Anambra", "AwkaNort"): ("Anambra", "Awka North"),
    ("Anambra", "AwkaSout"): ("Anambra", "Awka South"),
    ("Anambra", "NnewiSou"): ("Anambra", "Nnewi South"),
    ("Anambra", "OrumbaNo"): ("Anambra", "Orumba North"),
    ("Anambra", "OrumbaSo"): ("Anambra", "Orumba South"),
    ("Bauchi", "Damban"): ("Bauchi", "Dambam"),
    ("Bauchi", "Gamjuwa"): ("Bauchi", "Ganjuwa"),
    ("Bauchi", "Tafawa-B"): ("Bauchi", "Tafawa-Balewa"),
    ("Benue", "Katsina (Benue)"): ("Benue", "Katsina-Ala"),
    ("Cross River", "Calabar"): ("Cross River", "Calabar Municipal"),
    ("Cross River", "Yala Cross"): ("Cross River", "Yala"),
    ("Delta", "AniochaN"): ("Delta", "Aniocha North"),
    ("Delta", "AniochaS"): ("Delta", "Aniocha South"),
    ("Delta", "EthiopeE"): ("Delta", "Ethiope East"),
    ("Delta", "IkaNorth"): ("Delta", "Ika North East"),
    ("Delta", "IsokoNor"): ("Delta", "Isoko North"),
    ("Delta", "IsokoSou"): ("Delta", "Isoko South"),
    ("Ebonyi", "Afikpo"): ("Ebonyi", "Afikpo North"),
    ("Ebonyi", "AfikpoSo"): ("Ebonyi", "Afikpo South"),
    ("Edo", "EsanCent"): ("Edo", "Esan Central"),
    ("Edo", "EsanNort"): ("Edo", "Esan North-East"),
    ("Edo", "EsanSout"): ("Edo", "Esan South-East"),
    ("Edo", "EtsakoEa"): ("Edo", "Etsako East"),
    ("Edo", "EtsakoWe"): ("Edo", "Etsako West"),
    ("Edo", "Oredo Edo"): ("Edo", "Oredo"),
    ("Edo", "OviaNort"): ("Edo", "Ovia North-East"),
    ("Ekiti", "EkitiEas"): ("Ekiti", "Ekiti East"),
    ("Ekiti", "Emure/Ise/Orun"): ("Ekiti", "Emure"),
    ("Enugu", "EnuguSou"): ("Enugu", "Enugu South"),
    ("FCT", "AbujaMun"): ("FCT", "Municipal Area Council"),
    ("Gombe", "Yamaltu"): ("Gombe", "Yamaltu/Deba"),
    ("Imo", "Ahizu-Mb"): ("Imo", "Ahiazu-Mbaise"),
    ("Imo", "IdeatoNo"): ("Imo", "Ideato North"),
    ("Imo", "IsialaMb"): ("Imo", "Isiala Mbano"),
    ("Imo", "Unuimo"): ("Imo", "Onuimo"),
    ("Jigawa", "BirninKu"): ("Jigawa", "Birnin Kudu"),
    ("Jigawa", "KafinHau"): ("Jigawa", "Kafin Hausa"),
    ("Jigawa", "KiriKasa"): ("Jigawa", "Kiri Kasama"),
    ("Jigawa", "MalamMad"): ("Jigawa", "Malam Madori"),
    ("Jigawa", "Sule-Tan"): ("Jigawa", "Sule Tankarkar"),
    ("Kaduna", "ZangonKa"): ("Kaduna", "Zangon Kataf"),
    ("Kano", "DawakinK"): ("Kano", "Dawakin Kudu"),
    ("Kano", "DawakinT"): ("Kano", "Dawakin Tofa"),
    ("Kano", "Garum Mallam"): ("Kano", "Garun Malam"),
    ("Kano", "Kano"): ("Kano", "Kano Municipal"),
    ("Kano", "RiminGad"): ("Kano", "Rimin Gado"),
    ("Katsina", "Katsina (K)"): ("Katsina", "Katsina"),
    ("Kebbi", "Arewa"): ("Kebbi", "Arewa Dandi"),
    ("Kebbi", "Bagudo"): ("Kebbi", "Bagudu"),
    ("Kebbi", "BirninKe"): ("Kebbi", "Birnin Kebbi"),
    ("Kogi", "Kotonkar"): ("Kogi", "Kogi"),
    ("Kwara", "IlorinWe"): ("Kwara", "Ilorin West"),
    ("Lagos", "Mainland"): ("Lagos", "Lagos Mainland"),
    ("Niger", "Kontogur"): ("Niger", "Kontagora"),
    ("Niger", "Muya"): ("Niger", "Munya"),
    ("Ogun", "EgbadoNorth"): ("Ogun", "Yewa North"),
    ("Ogun", "EgbadoSouth"): ("Ogun", "Yewa South"),
    ("Ondo", "IlajeEseodo"): ("Ondo", "Ilaje"),
    ("Osun", "Odo0tin"): ("Osun", "Odo Otin"),
    ("Plateau", "Qua'anpa"): ("Plateau", "Qua'An Pan"),
    ("Rivers", "Akukutor"): ("Rivers", "Akuku Toru"),
    ("Rivers", "Ogba/Egbe"): ("Rivers", "Ogba/Egbema/Ndoni"),
    ("Sokoto", "Tambawal"): ("Sokoto", "Tambuwal"),
    ("Yobe", "Borsari"): ("Yobe", "Bursari"),
}


def _normalize_scalar(value: str) -> str:
    return normalize_admin_name(pd.Series([value])).iat[0]


def _normalize_series(series: pd.Series) -> pd.Series:
    return normalize_admin_name(series)


def _norm_key(state: str, lga: str) -> str:
    return f"{_normalize_scalar(state)}__{_normalize_scalar(lga)}"


MANUAL_OVERRIDES: Dict[str, str] = {
    _norm_key(src_state, src_lga): _norm_key(dst_state, dst_lga)
    for (src_state, src_lga), (dst_state, dst_lga) in _MANUAL_OVERRIDES_RAW.items()
}

MATCH_PRIORITY = {"exact": 1, "fuzzy": 2, "manual": 3}


def _similarity(a: str, b: str) -> float:
    """Fuzzy similarity for LGA names (state-constrained)."""

    seq = difflib.SequenceMatcher(None, a, b).ratio()
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    prefix = difflib.SequenceMatcher(None, shorter, longer[: len(shorter)]).ratio() if seq >= 0.8 else 0.0
    tokens_a = a.split()
    tokens_b = b.split()
    token_ratio = len(set(tokens_a) & set(tokens_b)) / max(len(tokens_a), len(tokens_b)) if tokens_a and tokens_b else 0
    return max(seq, prefix, token_ratio)


def _apply_state_alias(name: str) -> str:
    return STATE_ALIASES.get(name, name)


def reconcile_population_to_lgas(population_df: pd.DataFrame, lga_df: pd.DataFrame, *, min_similarity: float = 0.90) -> pd.DataFrame:
    """
    Align population rows to canonical LGA names.

    Parameters
    ----------
    population_df : DataFrame with columns ['lga_name', 'state_name', 'population']
    lga_df : DataFrame/GeoDataFrame with canonical ['lga_name', 'state_name']
    min_similarity : minimum fuzzy similarity required for automated matches
    """

    required_pop = {"lga_name", "state_name", "population"}
    required_lga = {"lga_name", "state_name"}
    if not required_pop.issubset(population_df.columns):
        missing = required_pop - set(population_df.columns)
        raise ValueError(f"Population data missing columns: {missing}")
    if not required_lga.issubset(lga_df.columns):
        missing = required_lga - set(lga_df.columns)
        raise ValueError(f"LGA data missing columns: {missing}")

    pop = population_df.copy()
    pop["state_name"] = pop["state_name"].apply(_apply_state_alias)
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    pop["state_norm"] = _normalize_series(pop["state_name"])
    pop["lga_norm"] = _normalize_series(pop["lga_name"])
    pop["norm_key"] = pop["state_norm"] + "__" + pop["lga_norm"]

    lgas = lga_df.copy()
    lgas["state_name"] = lgas["state_name"].apply(_apply_state_alias)
    lgas["state_norm"] = _normalize_series(lgas["state_name"])
    lgas["lga_norm"] = _normalize_series(lgas["lga_name"])
    lgas["norm_key"] = lgas["state_norm"] + "__" + lgas["lga_norm"]

    canonical_lookup = lgas.set_index("norm_key")[["state_name", "lga_name"]].to_dict(orient="index")
    matches = []
    unmatched_rows = []

    for _, row in pop.iterrows():
        pop_key = row["norm_key"]
        state_norm = row["state_norm"]

        if pop_key in canonical_lookup:
            matches.append({"target_key": pop_key, "population": row["population"], "source_match_type": "exact"})
            continue

        best_key = None
        best_score = 0.0
        candidates = lgas[lgas["state_norm"] == state_norm]
        for _, cand in candidates.iterrows():
            score = _similarity(row["lga_norm"], cand["lga_norm"])
            if score > best_score:
                best_score = score
                best_key = cand["norm_key"]

        if best_key and best_score >= min_similarity:
            matches.append(
                {
                    "target_key": best_key,
                    "population": row["population"],
                    "source_match_type": "fuzzy",
                }
            )
            continue

        if pop_key in MANUAL_OVERRIDES:
            matches.append(
                {
                    "target_key": MANUAL_OVERRIDES[pop_key],
                    "population": row["population"],
                    "source_match_type": "manual",
                }
            )
            continue

        unmatched_rows.append((row["state_name"], row["lga_name"], row["population"], best_score))

    if unmatched_rows:
        LOGGER.warning("Unmatched population rows (kept out): %s", unmatched_rows[:10])

    aggregated: Dict[str, Dict[str, float | str]] = {}
    for match in matches:
        key = match["target_key"]
        pop_val = match["population"]
        mtype = match["source_match_type"]
        if key not in aggregated:
            aggregated[key] = {"population": 0.0, "source_match_type": mtype}
        aggregated[key]["population"] = aggregated[key]["population"] + (pop_val if pd.notna(pop_val) else 0.0)
        if MATCH_PRIORITY[mtype] > MATCH_PRIORITY[aggregated[key]["source_match_type"]]:
            aggregated[key]["source_match_type"] = mtype

    records = []
    for key, vals in aggregated.items():
        meta = canonical_lookup[key]
        records.append(
            {
                "lga_name": meta["lga_name"],
                "state_name": meta["state_name"],
                "population": vals["population"],
                "source_match_type": vals["source_match_type"],
            }
        )

    result = pd.DataFrame(records).sort_values(["state_name", "lga_name"]).reset_index(drop=True)
    return result


def _cli(population_path: Path, lga_path: Path, output_path: Path, min_similarity: float) -> None:
    import geopandas as gpd  # local import to avoid hard dependency when unused

    population_df = pd.read_csv(population_path)
    lga_df = gpd.read_file(lga_path).rename(columns={"lganame": "lga_name", "statename": "state_name"})
    reconciled = reconcile_population_to_lgas(population_df, lga_df, min_similarity=min_similarity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reconciled.to_csv(output_path, index=False)
    LOGGER.info(
        "Wrote %d reconciled LGAs to %s (coverage %.2f%%)",
        len(reconciled),
        output_path,
        len(reconciled) / len(lga_df) * 100 if len(lga_df) else 0.0,
    )


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _cli(
        population_path=Path("data/raw/population_lga.csv"),
        lga_path=Path("data/raw/lga_boundaries.geojson"),
        output_path=Path("data/processed/population_lga_canonical.csv"),
        min_similarity=0.90,
    )
