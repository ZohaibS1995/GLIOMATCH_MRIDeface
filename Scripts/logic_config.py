import re

# logic_config.py

"""
Scientific Configuration: Heuristics and Identification Logic.
Edit this file to control how T1s are found and which sequences are defaced.
"""

LOGIC = {
    # -------------------------------------------------------------------------
    # T1 Identification Settings
    # -------------------------------------------------------------------------
    "t1_identification": {
        "keywords": [
            "t1", "mprage", "mpr", "spgr", "bravo", "vibe", "t1w", "t1-weighted"
        ],
        "priority_keywords": [
            "mprage", "bravo", "spgr"
        ],
        # Exact SeriesDescription preferences for selecting the mideface T1 input.
        # These are matched case-insensitively after removing spaces, underscores,
        # hyphens, and other punctuation.
        "preferred_series_descriptions": [
            "t1mpragetraiso",
            "t1mpr3dtraiso",
        ],
        "fallback_series_descriptions": [
            "t1mpragetraisokm",
            "t1mpr3dtraisokm",
        ],
        "avoid_contains": [
            "acpc", "kmacpc", "mprcor", "mprsag",
            "adc", "tracew", "calcbval",
        ],
        "avoid_prefixes": [
            "posdisp",
            "aaheadscout",
        ],
        "preferred_bonus": 200,
        "fallback_bonus": 140,
        "avoid_penalty": 80,
    },

    # -------------------------------------------------------------------------
    # Defacing Logic Settings
    # -------------------------------------------------------------------------
    "defacing_logic": {
        # WHITELIST: Deface these structural sequences.
        "structural_keywords": [
            "t1", "t2", "tse", "fse", "space", "cube", "vista", "flair",
            "tof", "angio", "swi", "swan", "susceptibility", "pd", "proton",
            "anat", "structural", "scout"
        ],

        # BLACKLIST: Do NOT deface these.
        "skip_keywords": [
            "dwi", "dti", "diffusion", "b-value", "adc", "trace", "fa_map",
            "fmri", "bold", "func", "task", "rest", "epi", "asl", "perfusion",
            "cbf", "localizer", "topogram", "screen", "loc", "phase",
            "field_map", "fieldmap", "mag", "screen save", "screenshot", "derived"
        ],

        # Minimum matrix size (Rows/Columns) required to trigger defacing.
        "min_matrix_size": 128
    }
}


def normalize_sequence_text(value):
    value = str(value or "").lower()
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip()


def normalize_series_label(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def matched_keywords(text, keywords):
    return [keyword for keyword in keywords if keyword in text]


def numeric_slice_count(value):
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def t1_preference_matches(series_description):
    conf = LOGIC.get("t1_identification", {})
    label = normalize_series_label(series_description)

    preferred = [normalize_series_label(item) for item in conf.get("preferred_series_descriptions", [])]
    fallback = [normalize_series_label(item) for item in conf.get("fallback_series_descriptions", [])]
    avoid_contains = [normalize_series_label(item) for item in conf.get("avoid_contains", [])]
    avoid_prefixes = [normalize_series_label(item) for item in conf.get("avoid_prefixes", [])]

    return {
        "PreferredT1SeriesDescriptionMatch": label if label in preferred else "",
        "FallbackT1SeriesDescriptionMatch": label if label in fallback else "",
        "AvoidedT1SeriesDescriptionMatches": [
            item for item in avoid_contains if item and item in label
        ] + [
            item for item in avoid_prefixes if item and label.startswith(item)
        ],
    }


def score_t1_candidate(meta: dict) -> int:
    """Scores a series to find the best native T1 for mideface input."""
    series_description = meta.get("SeriesDescription") or ""
    protocol_name = meta.get("ProtocolName") or ""
    computed_slice_count = numeric_slice_count(meta.get("computed_slice_count"))

    parsed_series_description = normalize_sequence_text(series_description)
    parsed_protocol_name = normalize_sequence_text(protocol_name)
    sequence_text = f"{parsed_series_description} {parsed_protocol_name}".strip()

    conf = LOGIC.get("t1_identification", {})
    keywords = [keyword.lower() for keyword in conf.get("keywords", [])]
    priority_keywords = [keyword.lower() for keyword in conf.get("priority_keywords", [])]
    preference = t1_preference_matches(series_description)

    score = 0
    if matched_keywords(sequence_text, keywords):
        score += 50
    if matched_keywords(sequence_text, priority_keywords):
        score += 20
    score += min(computed_slice_count or 0, 300) // 5
    if "head" in sequence_text or "brain" in sequence_text:
        score += 5

    if "kmacpc" in sequence_text:
        score -= 10
    
    if preference["PreferredT1SeriesDescriptionMatch"]:
        score += conf.get("preferred_bonus", 200)
    elif preference["FallbackT1SeriesDescriptionMatch"]:
        score += conf.get("fallback_bonus", 140)

    if preference["AvoidedT1SeriesDescriptionMatches"]:
        score -= conf.get("avoid_penalty", 80)

    return score
