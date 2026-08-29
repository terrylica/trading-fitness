"""
Hand-crafted NAV edge cases with pre-calculated expected results.

Each case tests a specific behavior of long/short epoch detection.
Expected values will be calibrated by running existing long algorithm first.

SR&ED: Experimental validation datasets for Bear ITH algorithm development.
SRED-Type: experimental-development
SRED-Claim: BEAR-ITH
"""

import numpy as np

# ============================================================
# CASE 1: Pure Decline (Shorts should profit)
# ============================================================
# NAV drops 20% with realistic volatility (2-3% daily swings)
CASE_1_PURE_DECLINE = {
    "name": "pure_decline",
    "description": "20% decline with realistic daily volatility",
    "nav": np.array(
        [
            100.0,  # Day 0: Start
            98.5,  # Day 1: -1.5%
            97.0,  # Day 2: -1.5%
            99.0,  # Day 3: +2.1% relief rally
            96.5,  # Day 4: -2.5%
            94.0,  # Day 5: -2.6%
            95.5,  # Day 6: +1.6% bounce
            92.0,  # Day 7: -3.7% sharp drop
            90.0,  # Day 8: -2.2%
            88.0,  # Day 9: -2.2%
            85.0,  # Day 10: -3.4%
            83.0,  # Day 11: -2.4%
            80.0,  # Day 12: -3.6% (total -20%)
        ],
        dtype=np.float64,
    ),
    "tmaeg": 0.05,  # 5% threshold
    "tmaer": 0.05,  # 5% threshold
    # Calibrated: Long algorithm finds NO epochs (no new highs in decline)
    "expected_bull_epochs": [],
    # Calibrated: Bear algorithm finds epochs at [5, 9, 11] (decline gains > 5%)
    "expected_bear_epochs": [5, 9, 11],
}

# ============================================================
# CASE 2: Pure Rally (Longs should profit, shorts hurt)
# ============================================================
CASE_2_PURE_RALLY = {
    "name": "pure_rally",
    "description": "20% rally with pullbacks - longs win, shorts lose",
    "nav": np.array(
        [
            100.0,  # Day 0
            102.0,  # Day 1: +2%
            104.5,  # Day 2: +2.5%
            103.0,  # Day 3: -1.4% pullback
            106.0,  # Day 4: +2.9%
            108.5,  # Day 5: +2.4%
            107.0,  # Day 6: -1.4% pullback
            110.0,  # Day 7: +2.8%
            113.0,  # Day 8: +2.7%
            111.5,  # Day 9: -1.3% pullback
            115.0,  # Day 10: +3.1%
            118.0,  # Day 11: +2.6%
            120.0,  # Day 12: +1.7% (total +20%)
        ],
        dtype=np.float64,
    ),
    "tmaeg": 0.05,  # 5% threshold
    "tmaer": 0.05,
    # Calibrated: Long algorithm finds epochs at [4, 8, 12] (breakouts after pullbacks)
    "expected_bull_epochs": [4, 8, 12],
    # Calibrated: Bear algorithm finds NO epochs (runup adverse for shorts)
    "expected_bear_epochs": [],
}

# ============================================================
# CASE 3: V-Recovery (Tests runup killing short gains)
# ============================================================
CASE_3_V_RECOVERY = {
    "name": "v_recovery",
    "description": "Sharp drop then full recovery - tests runup adverse effect",
    "nav": np.array(
        [
            100.0,  # Day 0
            95.0,  # Day 1: -5%
            90.0,  # Day 2: -5.3%
            85.0,  # Day 3: -5.6% (trough, short gain = 15%)
            88.0,  # Day 4: +3.5% bounce
            92.0,  # Day 5: +4.5% rally
            96.0,  # Day 6: +4.3%
            100.0,  # Day 7: +4.2% (full recovery, runup wipes short gain)
            103.0,  # Day 8: +3% new high
        ],
        dtype=np.float64,
    ),
    "tmaeg": 0.05,
    "tmaer": 0.05,
    # Calibrated: Long algorithm finds NO epochs (recovery doesn't exceed 5% from start)
    "expected_bull_epochs": [],
    # Calibrated: Bear algorithm finds epochs at [1, 2, 3] (decline phase)
    "expected_bear_epochs": [1, 2, 3],
}

# ============================================================
# CASE 4: Zigzag Down (Multiple short epochs)
# ============================================================
CASE_4_ZIGZAG_DOWN = {
    "name": "zigzag_down",
    "description": "Stair-step decline with relief rallies",
    "nav": np.array(
        [
            100.0,  # Day 0
            97.0,  # Day 1: -3%
            94.0,  # Day 2: -3.1%
            96.0,  # Day 3: +2.1% relief
            93.0,  # Day 4: -3.1% lower low
            90.0,  # Day 5: -3.2%
            92.5,  # Day 6: +2.8% relief
            89.0,  # Day 7: -3.8% lower low
            86.0,  # Day 8: -3.4%
            88.0,  # Day 9: +2.3% relief
            84.0,  # Day 10: -4.5% lower low
            80.0,  # Day 11: -4.8% (total -20%)
        ],
        dtype=np.float64,
    ),
    "tmaeg": 0.05,
    "tmaer": 0.05,
    # Calibrated: Long algorithm finds NO epochs (downtrend, no new highs)
    "expected_bull_epochs": [],
    # Calibrated: Bear algorithm finds epochs at [2, 7, 10, 11] (lower lows)
    # Note: Algorithm now finds extra epoch at day 11 due to symmetric formula fix
    "expected_bear_epochs": [2, 7, 10, 11],
}

# ============================================================
# CASE 5: Zigzag Up (Multiple long epochs)
# ============================================================
CASE_5_ZIGZAG_UP = {
    "name": "zigzag_up",
    "description": "Stair-step rally with pullbacks",
    "nav": np.array(
        [
            100.0,  # Day 0
            103.0,  # Day 1: +3%
            106.0,  # Day 2: +2.9%
            104.0,  # Day 3: -1.9% pullback
            107.0,  # Day 4: +2.9% higher high
            110.0,  # Day 5: +2.8%
            108.0,  # Day 6: -1.8% pullback
            112.0,  # Day 7: +3.7% higher high
            116.0,  # Day 8: +3.6%
            114.0,  # Day 9: -1.7% pullback
            118.0,  # Day 10: +3.5% higher high
            122.0,  # Day 11: +3.4% (total +22%)
        ],
        dtype=np.float64,
    ),
    "tmaeg": 0.05,
    "tmaer": 0.05,
    # Calibrated: Long algorithm finds epochs at [2, 7, 10] (higher-highs)
    "expected_bull_epochs": [2, 7, 10],
    # Calibrated: Bear algorithm finds NO epochs (uptrend adverse for shorts)
    "expected_bear_epochs": [],
}

# ============================================================
# CASE 6: Flat Market (No epochs either direction)
# ============================================================
CASE_6_FLAT = {
    "name": "flat_market",
    "description": "Sideways chop - no meaningful trends",
    "nav": np.array(
        [
            100.0,
            101.0,
            99.5,
            100.5,
            99.0,
            100.2,
            99.8,
            100.3,
            99.7,
            100.1,
            99.9,
            100.0,
        ],
        dtype=np.float64,
    ),
    "tmaeg": 0.05,
    "tmaer": 0.05,
    # Expected values: no epochs in flat market
    "expected_bull_epochs": [],
    "expected_bear_epochs": [],
}

# ============================================================
# CASE 7: Crash then Dead Cat Bounce (Short epoch mid-crash)
# ============================================================
CASE_7_CRASH_DCB = {
    "name": "crash_dead_cat_bounce",
    "description": "Sharp crash, dead cat bounce, then continuation",
    "nav": np.array(
        [
            100.0,  # Day 0
            92.0,  # Day 1: -8% crash
            85.0,  # Day 2: -7.6%
            80.0,  # Day 3: -5.9% (panic low)
            86.0,  # Day 4: +7.5% dead cat bounce
            88.0,  # Day 5: +2.3% bounce continues
            82.0,  # Day 6: -6.8% bounce fails
            75.0,  # Day 7: -8.5% new lows
            70.0,  # Day 8: -6.7% capitulation
        ],
        dtype=np.float64,
    ),
    "tmaeg": 0.05,
    "tmaer": 0.05,
    # Calibrated: Long algorithm finds NO epochs (crash, no new highs)
    "expected_bull_epochs": [],
    # Calibrated: Bear algorithm finds epochs at [1, 2, 3, 8] (crash points)
    "expected_bear_epochs": [1, 2, 3, 8],
}

# ============================================================
# CASE 8: Real Data Sample (from sample_account.csv volatility pattern)
# ============================================================
# Mimics actual PnL swings: 0.5-0.6% daily volatility
CASE_8_REALISTIC = {
    "name": "realistic_volatility",
    "description": "Mimics real trading data volatility from sample_account.csv",
    "nav": np.array(
        [
            1000000.0,  # Day 0: $1M start
            999243.71,  # Day 1: -0.08%
            995245.88,  # Day 2: -0.40%
            999479.32,  # Day 3: +0.43%
            1002259.82,  # Day 4: +0.28%
            1001696.80,  # Day 5: -0.06%
            1004082.15,  # Day 6: +0.24%
            1005438.37,  # Day 7: +0.14%
            1009837.56,  # Day 8: +0.44%
            1004579.17,  # Day 9: -0.52% (realistic drawdown)
            1005350.70,  # Day 10: +0.08%
            1010882.14,  # Day 11: +0.55%
            1013760.97,  # Day 12: +0.28%
            1013422.41,  # Day 13: -0.03%
            1015956.16,  # Day 14: +0.25%
            1020303.82,  # Day 15: +0.43%
            1017128.50,  # Day 16: -0.31%
            1023367.31,  # Day 17: +0.61%
            1023868.23,  # Day 18: +0.05%
            1022348.99,  # Day 19: -0.15%
        ],
        dtype=np.float64,
    ),
    "tmaeg": 0.02,  # 2% threshold (realistic for this vol)
    "tmaer": 0.02,
    # Calibrated: Long algorithm finds epoch at [15] (2% threshold exceeded)
    "expected_bull_epochs": [15],
    # Calibrated: Bear algorithm finds NO epochs (uptrend, shorts lose)
    "expected_bear_epochs": [],
}

# ============================================================
# CASE 9: Symmetry Test - Base NAV for inversion testing
# ============================================================
# Key property: If we invert NAV (mirror around mean), long/short epochs should swap
CASE_9_SYMMETRY_BASE = {
    "name": "symmetry_base",
    "description": "Base NAV for symmetry property testing",
    "nav": np.array([100, 105, 103, 108, 106, 112, 110, 115], dtype=np.float64),
    "tmaeg": 0.05,
    "tmaer": 0.05,
    # Calibrated: Long algorithm finds epochs at [1, 5] (5% gains)
    "expected_bull_epochs": [1, 5],
    # Calibrated: Bear algorithm finds NO epochs (uptrend adverse for shorts)
    "expected_bear_epochs": [],
}

# Inverted version: Mirror around midpoint (107.5)
# Inverted NAV = [115, 110, 112, 107, 109, 103, 105, 100]
# This is a downtrend with relief rallies - symmetric to base uptrend
CASE_9_SYMMETRY_INVERTED = {
    "name": "symmetry_inverted",
    "description": "Inverted NAV - long/short epochs should swap",
    "nav": 215 - CASE_9_SYMMETRY_BASE["nav"],  # Mirror around 107.5
    "tmaeg": 0.05,
    "tmaer": 0.05,
    # Symmetry: base had bear_epochs=[], so inverted has bull_epochs=[]
    "expected_bull_epochs": [],
    # Symmetry: base had bull_epochs=[1, 5], so inverted has bear_epochs=[1, 5]
    "expected_bear_epochs": [1, 5],
}

# ============================================================
# CASE 10: Boundary Condition - Exactly at Threshold
# ============================================================
CASE_10_BOUNDARY = {
    "name": "boundary_threshold",
    "description": "Gains exactly at threshold - tests >= vs > logic",
    "nav": np.array([100.0, 105.0, 104.0, 109.2], dtype=np.float64),  # 5% gain exactly
    "tmaeg": 0.05,
    "tmaer": 0.05,
    # Calibrated: Long algorithm finds epoch at [1] (uses > comparison, 5% triggers)
    "expected_bull_epochs": [1],
    # Calibrated: Bear algorithm finds NO epochs (uptrend)
    "expected_bear_epochs": [],
}

# All cases for parametrized testing (excluding symmetry inverted which is derived)
ALL_EDGE_CASES = [
    CASE_1_PURE_DECLINE,
    CASE_2_PURE_RALLY,
    CASE_3_V_RECOVERY,
    CASE_4_ZIGZAG_DOWN,
    CASE_5_ZIGZAG_UP,
    CASE_6_FLAT,
    CASE_7_CRASH_DCB,
    CASE_8_REALISTIC,
    CASE_9_SYMMETRY_BASE,
    CASE_10_BOUNDARY,
]

# Pair for symmetry testing
SYMMETRY_PAIRS = [
    (CASE_9_SYMMETRY_BASE, CASE_9_SYMMETRY_INVERTED),
]
