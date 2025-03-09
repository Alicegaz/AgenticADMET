# copypaste from https://github.com/asapdiscovery/asap-polaris-blind-challenge-examples/blob/a613051bac57060f686d9993e201ecaa15e51009/evaulation.py
# with a log-transform fix according to this issue https://github.com/asapdiscovery/asap-polaris-blind-challenge-examples/issues/14

from collections import defaultdict
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


TARGET_COLUMNS = ["LogHLM", "LogMLM", "LogD", "LogKSOL", "LogMDR1-MDCKII"]


def mask_nan(y_true, y_pred):
    mask = ~np.isnan(y_true)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]
    return y_true, y_pred

def eval_admet(preds: dict[str, list], refs: dict[str, list], target_columns: list[str] | None = None) \
    -> Tuple[dict[str, float], np.ndarray]:
    """
    Eval ADMET targets with MAE for pre-log10 transformed targets (LogD) and MALE  (MAE on log10 transformed dataset) on non-transformed data

    This provides a "relative" error metric that will not be as sensitive to the large outliers with huge errors. This is sometimes known as MALE.

    Parameters
    ----------
    preds : dict[str, list]
        Dictionary of predicted ADMET values.
    refs : dict[str, list]
        Dictionary of reference ADMET values.

    Returns
    -------
    dict[str, float]
        Returns a dictonary of summary statistics
    """
    keys = {
        "MLM",
        "HLM",
        "KSOL",
        "LogD",
        "MDR1-MDCKII",
    } if target_columns is None else target_columns
    # will be treated as is
    logscale_endpts = {"LogD"}

    collect = defaultdict(dict)

    for k in keys:
        if k not in preds.keys() or k not in refs.keys():
            raise ValueError("required key not present")

        ref, pred = mask_nan(refs[k], preds[k])

        if k in logscale_endpts:
            # already log10scaled
            mae = mean_absolute_error(ref, pred)
            r2 = r2_score(ref, pred)
        else:
            # clip to a detection limit
            # epsilon = 1e-8
            # pred = np.clip(pred, a_min=epsilon, a_max=None)
            # ref = np.clip(ref, a_min=epsilon, a_max=None)

            # transform both log10scale
            pred_log10s = np.log10(pred + 1.)
            ref_log10s = np.log10(ref + 1.)

            # compute MALE and R2 in log space
            mae = mean_absolute_error(ref_log10s, pred_log10s)
            r2 = r2_score(ref_log10s, pred_log10s)

        collect[k]["mean_absolute_error"] = mae
        collect[k]["r2"] = r2

    # compute macro average MAE
    macro_mae = np.mean([collect[k]["mean_absolute_error"] for k in keys])
    collect["aggregated"]["macro_mean_absolute_error"] = macro_mae

    # compute macro average R2
    macro_r2 = np.mean([collect[k]["r2"] for k in keys])
    collect["aggregated"]["macro_r2"] = macro_r2

    return collect


def extract_preds(preds: pd.DataFrame, target_columns: list[str] = TARGET_COLUMNS):
    preds_dict = {}
    for t in target_columns:
        if t in ["LogHLM", "LogMLM", "LogKSOL", "LogMDR1-MDCKII"]:
            # transform back to non-log scale
            preds_dict[t[3:]] = np.power(10, preds.iloc[:, preds.columns.get_loc(f"pred_{t}")].values) - 1.
        else:
            preds_dict[t] = preds.iloc[:, preds.columns.get_loc(f"pred_{t}")].values
    
    return preds_dict

def extract_refs(refs: pd.DataFrame, target_columns: list[str] = TARGET_COLUMNS):
    refs_dict = {}
    for t in target_columns:
        if t in ["LogHLM", "LogMLM", "LogKSOL", "LogMDR1-MDCKII"]:
            refs_dict[t[3:]] = refs.iloc[:, refs.columns.get_loc(t[3:])].values
        else:
            refs_dict[t] = refs.iloc[:, refs.columns.get_loc(t)].values
    
    return refs_dict
