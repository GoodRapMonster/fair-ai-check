import pytest
import pandas as pd
import numpy as np
from core.bias_engine import BiasEngine

@pytest.fixture
def bias_engine():
    return BiasEngine()

@pytest.fixture
def binary_biased_df():
    # 100 rows, sex: 0(unpriv), 1(priv)
    # Selection rates: unpriv=0.2, priv=0.8
    data = {
        'sex': [0]*50 + [1]*50,
        'outcome': [0]*40 + [1]*10 + [0]*10 + [1]*40
    }
    return pd.DataFrame(data)

@pytest.fixture
def multi_group_df():
    # A=0.8, B=0.4, C=0.2
    data = {
        'race': ['A']*50 + ['B']*50 + ['C']*50,
        'outcome': [1]*40 + [0]*10 + [1]*20 + [0]*30 + [1]*10 + [0]*40
    }
    return pd.DataFrame(data)

def test_disparate_impact_binary(bias_engine, binary_biased_df):
    # unpriv/priv = 0.2 / 0.8 = 0.25
    score = bias_engine.compute_disparate_impact(binary_biased_df, 'outcome', 'sex', 1)
    assert score == pytest.approx(0.25)

def test_disparate_impact_multi_group(bias_engine, multi_group_df):
    # min(A,B,C) / A = 0.2 / 0.8 = 0.25
    score = bias_engine.compute_disparate_impact(multi_group_df, 'outcome', 'race', 'A')
    assert score == pytest.approx(0.25)

def test_statistical_parity_binary(bias_engine, binary_biased_df):
    # 1 - |0.2 - 0.8| = 0.4
    score = bias_engine.compute_statistical_parity_diff(binary_biased_df, 'outcome', 'sex', 1)
    assert score == pytest.approx(0.4)

def test_statistical_parity_multi_group(bias_engine, multi_group_df):
    # priv=A=0.8. Gaps: B(0.4), C(0.6). Max gap = 0.6. 
    # Result = 1 - 0.6 = 0.4
    score = bias_engine.compute_statistical_parity_diff(multi_group_df, 'outcome', 'race', 'A')
    assert score == pytest.approx(0.4)

def test_raw_dp_gap(bias_engine, binary_biased_df):
    # 0.2 - 0.8 = -0.6
    gap = bias_engine.compute_raw_dp_gap(binary_biased_df, 'outcome', 'sex', 1)
    assert gap == pytest.approx(-0.6)

def test_bootstrap_ci_structure(bias_engine, binary_biased_df):
    # Just check it returns the 4-tuple and values are reasonable
    result = bias_engine.compute_disparate_impact(binary_biased_df, 'outcome', 'sex', 1, compute_stats=True)
    assert isinstance(result, tuple)
    assert len(result) == 4
    score, lower, upper, p_val = result
    assert 0 <= score <= 1
    if lower is not None:
        assert lower <= score <= upper
