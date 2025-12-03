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
        ]
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