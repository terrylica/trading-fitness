import numpy as np
import pandas as pd
from scipy import stats
from numba import njit, prange
from numba.typed import List
from typing import NamedTuple

class ExcessGainLossResult(NamedTuple):
    excess_gains: np.ndarray
    excess_losses: np.ndarray
    num_of_ith_epochs: int
    ith_epochs: np.ndarray
    ith_intervals_cv: float

@njit
def _excess_gain_excess_loss_numba_original(nav, hurdle):
    excess_gain = excess_loss = 0.0  # Initialize as float
    excess_gains = List([0.0])
    excess_losses = List([0.0])
    excess_gains_at_ith_epoch = List([0.0])
    last_reset_state = False
    ith_epochs = List([False] * len(nav))
    endorsing_crest = endorsing_nadir = candidate_crest = candidate_nadir = nav[0]
    for i, (equity, next_equity) in enumerate(zip(nav[:-1], nav[1:])):
        if next_equity > candidate_crest:
            excess_gain = next_equity / endorsing_crest - 1 if endorsing_crest != 0 else 0.0
            candidate_crest = next_equity
        if next_equity < candidate_nadir:
            excess_loss = 1 - next_equity / endorsing_crest
            candidate_nadir = next_equity
        reset_candidate_nadir_excess_gain_and_excess_loss = excess_gain > abs(excess_loss) and excess_gain > hurdle and candidate_crest >= endorsing_crest
        if reset_candidate_nadir_excess_gain_and_excess_loss:
            endorsing_crest = candidate_crest
            endorsing_nadir = candidate_nadir = equity
            excess_gains_at_ith_epoch.append(excess_gain if not last_reset_state else 0.0)
        else:
            endorsing_nadir = min(endorsing_nadir, equity)
            excess_gains_at_ith_epoch.append(0.0)
        last_reset_state = reset_candidate_nadir_excess_gain_and_excess_loss
        excess_gains.append(excess_gain)
        excess_losses.append(excess_loss)
        if reset_candidate_nadir_excess_gain_and_excess_loss:
            excess_gain = excess_loss = 0.0
        ith_epoch_condition = len(excess_gains) > 1 and excess_gains[-1] > excess_losses[-1] and excess_gains[-1] > hurdle
        ith_epochs[i + 1] = ith_epoch_condition
    num_of_ith_epochs = sum(ith_epochs)
    ith_interval_separators = [i for i, x in enumerate(ith_epochs) if x]
    ith_interval_separators.insert(0, 0)
    ith_intervals = np.diff(np.array(ith_interval_separators))  # Convert to NumPy array before using np.diff
    ith_intervals_cv = np.std(ith_intervals) / np.mean(ith_intervals) if len(ith_intervals) > 0 else np.nan
    return ExcessGainLossResult(
        excess_gains=np.array(excess_gains),
        excess_losses=np.array(excess_losses),
        num_of_ith_epochs=num_of_ith_epochs,
        ith_epochs=np.array(ith_epochs),
        ith_intervals_cv=ith_intervals_cv
    )

@njit(parallel=True)
def _excess_gain_excess_loss_numba_modified(nav, hurdle):
    excess_gain = excess_loss = 0.0  # Initialize as float
    excess_gains = List([0.0])
    excess_losses = List([0.0])
    excess_gains_at_ith_epoch = List([0.0])
    ith_epochs = np.zeros(len(nav), dtype=np.bool_)
    endorsing_crest = endorsing_nadir = candidate_crest = candidate_nadir = nav[0]
    
    for i in prange(1, len(nav)):
        equity = nav[i - 1]
        next_equity = nav[i]
        
        if next_equity > candidate_crest:
            excess_gain = next_equity / endorsing_crest - 1 if endorsing_crest != 0 else 0.0
            candidate_crest = next_equity
        if next_equity < candidate_nadir:
            excess_loss = 1 - next_equity / endorsing_crest
            candidate_nadir = next_equity
        
        reset_candidate_nadir_excess_gain_and_excess_loss = excess_gain > abs(excess_loss) and excess_gain > hurdle and candidate_crest >= endorsing_crest
        if reset_candidate_nadir_excess_gain_and_excess_loss:
            endorsing_crest = candidate_crest
            endorsing_nadir = candidate_nadir = equity
            excess_gains_at_ith_epoch.append(excess_gain)
        else:
            endorsing_nadir = min(endorsing_nadir, equity)
            excess_gains_at_ith_epoch.append(0.0)
        
        excess_gains.append(excess_gain)
        excess_losses.append(excess_loss)
        
        if reset_candidate_nadir_excess_gain_and_excess_loss:
            excess_gain = excess_loss = 0.0
        
        ith_epoch_condition = len(excess_gains) > 1 and excess_gains[-1] > excess_losses[-1] and excess_gains[-1] > hurdle
        ith_epochs[i] = ith_epoch_condition
    
    num_of_ith_epochs = np.sum(ith_epochs)
    ith_interval_separators = np.where(ith_epochs)[0]
    ith_interval_separators = np.insert(ith_interval_separators, 0, 0)
    ith_intervals = np.diff(ith_interval_separators)
    ith_intervals_cv = np.std(ith_intervals) / np.mean(ith_intervals) if len(ith_intervals) > 0 else np.nan
    
    return ExcessGainLossResult(
        excess_gains=np.array(excess_gains),
        excess_losses=np.array(excess_losses),
        num_of_ith_epochs=num_of_ith_epochs,
        ith_epochs=ith_epochs,
        ith_intervals_cv=ith_intervals_cv
    )

def generate_synthetic_nav(start_date='2020-01-01', end_date='2023-01-01', avg_daily_return=0.0001, daily_return_volatility=0.01, df=5):
    dates = pd.date_range(start_date, end_date)
    walk = stats.t.rvs(df, loc=avg_daily_return, scale=daily_return_volatility, size=len(dates))
    walk = np.cumsum(walk)
    walk = walk - walk[0] + 1  # Normalize the series so that it starts with 1
    nav = pd.DataFrame(data=walk, index=dates, columns=['NAV'])
    nav.index.name = 'Date'
    return nav

def compare_results(nav, hurdle):
    result_original = _excess_gain_excess_loss_numba_original(nav.values, hurdle)
    result_modified = _excess_gain_excess_loss_numba_modified(nav.values, hurdle)
    
    assert np.allclose(result_original.excess_gains, result_modified.excess_gains), "Excess gains do not match!"
    assert np.allclose(result_original.excess_losses, result_modified.excess_losses), "Excess losses do not match!"
    assert result_original.num_of_ith_epochs == result_modified.num_of_ith_epochs, "Number of ITH epochs do not match!"
    assert np.all(result_original.ith_epochs == result_modified.ith_epochs), "ITH epochs do not match!"
    assert np.isclose(result_original.ith_intervals_cv, result_modified.ith_intervals_cv, equal_nan=True), "ITH intervals CV do not match!"
    
    print("All checks passed. The results are identical.")

if __name__ == "__main__":
    # Generate synthetic NAV data
    nav = generate_synthetic_nav()
    hurdle = 0.05  # Example hurdle rate
    
    try:
        compare_results(nav, hurdle)
    except AssertionError as e:
        print(f"Test failed: {e}")