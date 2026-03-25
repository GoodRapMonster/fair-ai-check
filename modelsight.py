#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║  ModelSight CLI — Model Bias Detection Tool                      ║
║  6-Layer fairness audit for trained ML models                    ║
║                                                                   ║
║  Companion to FairSight (dataset bias).                          ║
║  This tool evaluates a TRAINED MODEL's predictions for bias.     ║
╚═══════════════════════════════════════════════════════════════════╝

Usage:
    python modelsight.py --model model.pkl --csv test_data.csv --target label
    python modelsight.py --model model.joblib --csv test.csv --target income --protected gender race
"""

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from colorama import Fore, Style, init

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import pickle
    HAS_PICKLE = True
except ImportError:
    HAS_PICKLE = False

try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROTECTED_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "age",
    "religion", "nationality", "disability", "marital",
]

DI_THRESHOLD = 0.8          # Disparate Impact
DP_THRESHOLD = 0.1          # Demographic Parity Gap
EO_THRESHOLD = 0.1          # Equal Opportunity Difference
PP_THRESHOLD = 0.1          # Predictive Parity Difference
FPR_THRESHOLD = 0.05        # False Positive Rate gap
FNR_THRESHOLD = 0.05        # False Negative Rate gap
CALIBRATION_THRESHOLD = 0.1 # Calibration gap
COUNTERFACTUAL_THRESHOLD = 0.05  # Counterfactual flip rate


def dynamic_min_size(n):
    """Scale minimum subgroup size to dataset size."""
    if n <= 20:   return 2
    if n <= 50:   return 5
    if n <= 200:  return 10
    if n <= 1000: return 20
    return 30


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON ENCODER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if math.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DISPLAY HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def banner():
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   🔍 ModelSight CLI — Model Bias Detection Tool      ║")
    print("  ║      Comprehensive 6-Layer Model Fairness Audit      ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)


def shdr(n, t):
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'━' * 62}")
    print(f"  LAYER {n} — {t}")
    print(f"{'━' * 62}{Style.RESET_ALL}\n")


def p_pass(t):
    print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}  {t}")


def p_fail(t):
    print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL}  {t}")


def p_warn(t):
    print(f"  {Fore.YELLOW}⚠ WARN{Style.RESET_ALL}  {t}")


def p_info(t):
    print(f"  {Fore.CYAN}ℹ INFO{Style.RESET_ALL}  {t}")


def p_skip(t):
    print(f"  {Fore.CYAN}⊘ SKIP{Style.RESET_ALL}  {t}")


def p_rec(t):
    print(f"  {Fore.MAGENTA}→ REC {Style.RESET_ALL}  {t}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITY FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_protected(df):
    """Auto-detect protected attribute columns by keyword matching."""
    return [c for c in df.columns if any(k in c.lower() for k in PROTECTED_KEYWORDS)]


def bin_numeric_protected(df, attrs):
    """Bin numeric protected attributes into meaningful groups."""
    df = df.copy()
    new_attrs = []
    info = {}
    for attr in attrs:
        col = df[attr]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > 10:
            if "age" in attr.lower():
                df[attr + "_group"] = pd.cut(
                    col, [0, 25, 35, 45, 55, 65, 120],
                    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
                    right=True,
                ).astype(str)
            else:
                try:
                    df[attr + "_group"] = pd.qcut(col, q=4, duplicates="drop").astype(str)
                except Exception:
                    df[attr + "_group"] = col.astype(str)
            new_attrs.append(attr + "_group")
            info[attr] = True
            p_info(f"Binned numeric '{attr}' → '{attr}_group'")
        else:
            new_attrs.append(attr)
    return df, new_attrs, info


def binarise_target(df, target):
    """Safely binarise any target column type."""
    df = df.copy()
    col = df[target]
    nu = col.nunique()
    if nu <= 1:
        p_warn(f"Target '{target}' has only 1 unique value — analysis will be limited.")
        df[target] = 1
        return df, "1", True
    if nu == 2:
        vals = col.dropna().unique().tolist()
        pos = next(
            (v for v in vals if ">" in str(v) or str(v).strip().lower() in ("1", "yes", "true", "high")),
            sorted(vals, key=str)[-1],
        )
        df[target] = (col == pos).astype(int)
        return df, str(pos), False
    if pd.api.types.is_numeric_dtype(col):
        med = col.median()
        p_warn(f"Multi-value numeric target — binarising at median ({med}).")
        df[target] = (col >= med).astype(int)
        return df, f">={med}", False
    top = col.mode()[0]
    p_warn(f"Multi-value string target — using '{top}' as positive class.")
    df[target] = (col == top).astype(int)
    return df, str(top), False


def safe_di(u, p):
    """NaN/Inf-safe disparate impact."""
    if p is None or math.isnan(p) or p == 0 or u is None or math.isnan(u):
        return None
    return u / p


def load_model(model_path):
    """Load a trained model from .pkl or .joblib file."""
    # Try joblib first (most sklearn models are saved with joblib), then pickle
    if HAS_JOBLIB:
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    print(f"{Fore.RED}Error: Cannot load model from '{model_path}'. "
          f"Supported formats: .pkl, .joblib{Style.RESET_ALL}")
    sys.exit(1)


def prepare_features(df, protected_attrs, target, binned_originals=None, model=None):
    """Prepare feature matrix X for model prediction.
    
    Attempts to detect the model's expected features automatically.
    If the model was trained WITH protected attributes, they stay in.
    If trained WITHOUT them, they are excluded.
    """
    binned_originals = binned_originals or set()

    # Try to get the model's expected feature names
    model_features = None
    if model is not None and hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        p_info(f"Detected model expects {len(model_features)} features: {', '.join(model_features)}")

        # Check which are available in df
        missing = [f for f in model_features if f not in df.columns]
        if missing:
            p_warn(f"Features expected by model but missing from CSV: {', '.join(missing)}")
        feature_cols = [f for f in model_features if f in df.columns]

        # Flag if protected attrs are included
        prot_in_model = [f for f in feature_cols if f in protected_attrs or f in binned_originals
                         or any(k in f.lower() for k in PROTECTED_KEYWORDS)]
        if prot_in_model:
            p_warn(f"Model uses protected attributes as features: {', '.join(prot_in_model)}")
        else:
            p_pass("Model does NOT use protected attributes as direct features.")
    else:
        # Fallback: use all columns except target and binned group columns
        feature_cols = [
            c for c in df.columns
            if c != target and c not in binned_originals
            and not c.endswith("_group")  # exclude derived group columns
        ]
        p_info(f"Using all non-target columns as features: {', '.join(feature_cols)}")

    X = df[feature_cols].copy()
    # Encode categorical columns
    encoders = {}
    for c in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le
    # Fill missing
    X = X.fillna(X.median(numeric_only=True))
    return X, feature_cols, encoders


def get_predictions(model, X):
    """Get binary predictions and probabilities from a model."""
    predictions = model.predict(X)

    # Try to get probabilities
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] >= 2:
                probabilities = proba[:, 1]
            else:
                probabilities = proba[:, 0]
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            probabilities = model.decision_function(X)
        except Exception:
            pass

    return predictions, probabilities


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 1 — PREDICTION FAIRNESS METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer1(df, protected_attrs, target, predictions, min_size):
    """Evaluate standard fairness metrics on model predictions."""
    shdr(1, "Prediction Fairness Metrics")
    results = {"status": "PASS", "metrics": {}, "issues": []}

    actual = df[target].values

    # Model accuracy overview
    if HAS_SKLEARN:
        acc = accuracy_score(actual, predictions)
        p_info(f"Model accuracy on test set: {acc:.1%}")
        results["model_accuracy"] = round(acc, 4)
    print()

    for attr in protected_attrs:
        print(f"  {Style.BRIGHT}Protected: {attr}{Style.RESET_ALL}")
        groups = df[attr].dropna().unique()

        # Warn small groups
        for g in groups:
            cnt = (df[attr] == g).sum()
            if cnt < min_size:
                p_warn(f"Group '{g}' n={cnt} (< {min_size})")

        # Need >= 2 valid groups
        valid = [g for g in groups if (df[attr] == g).sum() >= min_size]
        if len(valid) < 2:
            p_skip(f"'{attr}' has <2 groups with ≥{min_size} samples — skipping")
            results["metrics"][attr] = {"skipped": True, "reason": "insufficient_groups"}
            continue

        # Use model predictions to calculate rates
        sub = df[df[attr].isin(valid)]
        sub_preds = predictions[df[attr].isin(valid)]

        # Positive prediction rates per group
        pred_rates = {}
        for g in valid:
            mask = sub[attr] == g
            pred_rates[g] = float(sub_preds[mask.values].mean())

        priv = max(pred_rates, key=pred_rates.get)
        unpriv = min(pred_rates, key=pred_rates.get)
        pr = pred_rates[priv]
        ur = pred_rates[unpriv]

        priv_mask = (df[attr] == priv).values
        unpriv_mask = (df[attr] == unpriv).values

        am = {}

        # 1. Disparate Impact (on predictions)
        di = safe_di(ur, pr)
        if di is None:
            p_skip(f"DI: cannot compute (priv_rate={pr:.3f})")
            am["disparate_impact"] = {"value": None, "pass": None, "reason": "degenerate"}
        else:
            ok = di >= DI_THRESHOLD
            am["disparate_impact"] = {
                "value": round(di, 4), "threshold": DI_THRESHOLD, "pass": ok,
                "privileged": str(priv), "priv_rate": round(pr, 4), "unpriv_rate": round(ur, 4),
            }
            (p_pass if ok else p_fail)(
                f"Disparate Impact: {di:.4f}  "
                f"({'≥' if ok else '<'} {DI_THRESHOLD})  "
                f"[priv={priv} {pr:.1%} | unpriv={unpriv} {ur:.1%}]"
            )
            if not ok:
                results["status"] = "FAIL"
                results["issues"].append(f"DI={di:.2f} on '{attr}'")

        # 2. Demographic Parity Gap (on predictions)
        dp = ur - pr
        dp = 0.0 if math.isnan(dp) else dp
        dp_ok = abs(dp) <= DP_THRESHOLD
        am["demographic_parity"] = {
            "value": round(dp, 4), "threshold": DP_THRESHOLD, "pass": dp_ok,
        }
        (p_pass if dp_ok else p_fail)(
            f"Demographic Parity Gap: {dp:+.4f}  "
            f"({'|gap| ≤' if dp_ok else '|gap| >'} {DP_THRESHOLD})"
        )
        if not dp_ok:
            results["status"] = "FAIL"
            results["issues"].append(f"DP={dp:+.2f} on '{attr}'")

        # 3. Equal Opportunity Difference
        priv_actual_pos = actual[priv_mask] == 1
        unpriv_actual_pos = actual[unpriv_mask] == 1

        priv_tpr = (
            float(predictions[priv_mask][priv_actual_pos].mean())
            if priv_actual_pos.sum() > 0 else 0.0
        )
        unpriv_tpr = (
            float(predictions[unpriv_mask][unpriv_actual_pos].mean())
            if unpriv_actual_pos.sum() > 0 else 0.0
        )
        eod = unpriv_tpr - priv_tpr
        eod = 0.0 if math.isnan(eod) else eod
        eod_ok = abs(eod) <= EO_THRESHOLD
        am["equal_opportunity"] = {
            "value": round(eod, 4), "threshold": EO_THRESHOLD, "pass": eod_ok,
            "priv_tpr": round(priv_tpr, 4), "unpriv_tpr": round(unpriv_tpr, 4),
        }
        (p_pass if eod_ok else p_fail)(
            f"Equal Opportunity Diff: {eod:+.4f}  "
            f"({'|diff| ≤' if eod_ok else '|diff| >'} {EO_THRESHOLD})"
        )
        if not eod_ok:
            results["status"] = "FAIL"
            results["issues"].append(f"EOD={eod:+.2f} on '{attr}'")

        # 4. Predictive Parity Difference
        priv_pred_pos = predictions[priv_mask] == 1
        unpriv_pred_pos = predictions[unpriv_mask] == 1

        priv_ppv = (
            float(actual[priv_mask][priv_pred_pos].mean())
            if priv_pred_pos.sum() > 0 else 1.0
        )
        unpriv_ppv = (
            float(actual[unpriv_mask][unpriv_pred_pos].mean())
            if unpriv_pred_pos.sum() > 0 else 1.0
        )
        ppd = unpriv_ppv - priv_ppv
        ppd = 0.0 if math.isnan(ppd) else ppd
        ppd_ok = abs(ppd) <= PP_THRESHOLD
        am["predictive_parity"] = {
            "value": round(ppd, 4), "threshold": PP_THRESHOLD, "pass": ppd_ok,
            "priv_ppv": round(priv_ppv, 4), "unpriv_ppv": round(unpriv_ppv, 4),
        }
        (p_pass if ppd_ok else p_fail)(
            f"Predictive Parity Diff: {ppd:+.4f}  "
            f"({'|diff| ≤' if ppd_ok else '|diff| >'} {PP_THRESHOLD})"
        )
        if not ppd_ok:
            results["status"] = "FAIL"
            results["issues"].append(f"PPD={ppd:+.2f} on '{attr}'")

        results["metrics"][attr] = am

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 2 — ERROR RATE DISPARITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer2(df, protected_attrs, target, predictions, min_size):
    """Analyze False Positive Rate and False Negative Rate gaps across groups."""
    shdr(2, "Error Rate Disparity Analysis")
    results = {"status": "PASS", "error_rates": {}, "issues": []}

    actual = df[target].values

    for attr in protected_attrs:
        print(f"\n  {Style.BRIGHT}Protected: {attr}{Style.RESET_ALL}")
        groups = df[attr].dropna().unique()
        valid = [g for g in groups if (df[attr] == g).sum() >= min_size]

        if len(valid) < 2:
            p_skip(f"'{attr}' has <2 groups with ≥{min_size} samples — skipping")
            results["error_rates"][attr] = {"skipped": True}
            continue

        group_rates = {}
        for g in valid:
            mask = (df[attr] == g).values
            g_actual = actual[mask]
            g_preds = predictions[mask]

            # Calculate confusion matrix components
            tp = int(((g_preds == 1) & (g_actual == 1)).sum())
            fp = int(((g_preds == 1) & (g_actual == 0)).sum())
            tn = int(((g_preds == 0) & (g_actual == 0)).sum())
            fn = int(((g_preds == 0) & (g_actual == 1)).sum())

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

            group_rates[str(g)] = {
                "n": int(mask.sum()),
                "fpr": round(fpr, 4),
                "fnr": round(fnr, 4),
                "accuracy": round(acc, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }

        results["error_rates"][attr] = group_rates

        # Find max FPR and FNR gaps
        fprs = {g: r["fpr"] for g, r in group_rates.items()}
        fnrs = {g: r["fnr"] for g, r in group_rates.items()}
        accs = {g: r["accuracy"] for g, r in group_rates.items()}

        fpr_gap = max(fprs.values()) - min(fprs.values())
        fnr_gap = max(fnrs.values()) - min(fnrs.values())
        acc_gap = max(accs.values()) - min(accs.values())

        # Print per-group stats
        for g, r in group_rates.items():
            print(f"    {Style.BRIGHT}{g}:{Style.RESET_ALL} "
                  f"FPR={r['fpr']:.1%}  FNR={r['fnr']:.1%}  "
                  f"Acc={r['accuracy']:.1%}  (n={r['n']})")

        # Check FPR gap
        fpr_ok = fpr_gap <= FPR_THRESHOLD
        (p_pass if fpr_ok else p_fail)(
            f"FPR gap: {fpr_gap:.4f}  "
            f"({'≤' if fpr_ok else '>'} {FPR_THRESHOLD})  "
            f"[highest={max(fprs, key=fprs.get)} {max(fprs.values()):.1%} | "
            f"lowest={min(fprs, key=fprs.get)} {min(fprs.values()):.1%}]"
        )
        if not fpr_ok:
            results["status"] = "FAIL"
            results["issues"].append(
                f"FPR gap={fpr_gap:.2f} on '{attr}' "
                f"({max(fprs, key=fprs.get)} vs {min(fprs, key=fprs.get)})"
            )

        # Check FNR gap
        fnr_ok = fnr_gap <= FNR_THRESHOLD
        (p_pass if fnr_ok else p_fail)(
            f"FNR gap: {fnr_gap:.4f}  "
            f"({'≤' if fnr_ok else '>'} {FNR_THRESHOLD})  "
            f"[highest={max(fnrs, key=fnrs.get)} {max(fnrs.values()):.1%} | "
            f"lowest={min(fnrs, key=fnrs.get)} {min(fnrs.values()):.1%}]"
        )
        if not fnr_ok:
            results["status"] = "FAIL"
            results["issues"].append(
                f"FNR gap={fnr_gap:.2f} on '{attr}' "
                f"({max(fnrs, key=fnrs.get)} vs {min(fnrs, key=fnrs.get)})"
            )

        # Accuracy gap as informational
        if acc_gap > 0.1:
            p_warn(
                f"Accuracy gap: {acc_gap:.1%}  "
                f"[best={max(accs, key=accs.get)} {max(accs.values()):.1%} | "
                f"worst={min(accs, key=accs.get)} {min(accs.values()):.1%}]"
            )

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 3 — CALIBRATION ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer3(df, protected_attrs, target, probabilities, min_size):
    """Analyze whether predicted probabilities are well-calibrated per group."""
    shdr(3, "Prediction Calibration Analysis")
    results = {"status": "PASS", "calibration": {}, "issues": []}

    if probabilities is None:
        p_skip("Model does not provide probability estimates — skipping calibration.")
        results["status"] = "SKIP"
        return results

    actual = df[target].values

    for attr in protected_attrs:
        print(f"\n  {Style.BRIGHT}Protected: {attr}{Style.RESET_ALL}")
        groups = df[attr].dropna().unique()
        valid = [g for g in groups if (df[attr] == g).sum() >= min_size]

        if len(valid) < 2:
            p_skip(f"'{attr}' has <2 groups with ≥{min_size} samples — skipping")
            results["calibration"][attr] = {"skipped": True}
            continue

        group_cal = {}
        for g in valid:
            mask = (df[attr] == g).values
            g_actual = actual[mask]
            g_probs = probabilities[mask]

            # Bin predictions into quintiles and check calibration
            mean_pred = float(g_probs.mean())
            mean_actual = float(g_actual.mean())
            cal_error = abs(mean_pred - mean_actual)

            # Brier score (lower is better)
            brier = float(np.mean((g_probs - g_actual) ** 2))

            group_cal[str(g)] = {
                "n": int(mask.sum()),
                "mean_predicted_prob": round(mean_pred, 4),
                "actual_positive_rate": round(mean_actual, 4),
                "calibration_error": round(cal_error, 4),
                "brier_score": round(brier, 4),
            }

        results["calibration"][attr] = group_cal

        # Print per-group calibration
        for g, c in group_cal.items():
            ce = c["calibration_error"]
            icon = "✓" if ce <= CALIBRATION_THRESHOLD else "✗"
            col = Fore.GREEN if ce <= CALIBRATION_THRESHOLD else Fore.RED
            print(
                f"    {col}{icon}{Style.RESET_ALL} {Style.BRIGHT}{g}:{Style.RESET_ALL} "
                f"predicted={c['mean_predicted_prob']:.1%}  actual={c['actual_positive_rate']:.1%}  "
                f"error={ce:.1%}  brier={c['brier_score']:.4f}  (n={c['n']})"
            )

        # Check max calibration gap between groups
        cal_errors = {g: c["calibration_error"] for g, c in group_cal.items()}
        briers = {g: c["brier_score"] for g, c in group_cal.items()}
        max_ce_gap = max(cal_errors.values()) - min(cal_errors.values())
        max_brier_gap = max(briers.values()) - min(briers.values())

        if max_ce_gap > CALIBRATION_THRESHOLD:
            p_fail(
                f"Calibration error gap: {max_ce_gap:.4f}  (> {CALIBRATION_THRESHOLD})  "
                f"[worst={max(cal_errors, key=cal_errors.get)} | "
                f"best={min(cal_errors, key=cal_errors.get)}]"
            )
            results["status"] = "WARN"
            results["issues"].append(
                f"Calibration gap={max_ce_gap:.2f} on '{attr}'"
            )
        else:
            p_pass(f"Calibration error gap: {max_ce_gap:.4f}  (≤ {CALIBRATION_THRESHOLD})")

        if max_brier_gap > 0.1:
            p_warn(
                f"Brier score gap: {max_brier_gap:.4f}  "
                f"[worst={max(briers, key=briers.get)} {max(briers.values()):.4f} | "
                f"best={min(briers, key=briers.get)} {min(briers.values()):.4f}]"
            )

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 4 — FEATURE CONTRIBUTION BIAS (SHAP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer4(df, model, X, feature_cols, protected_attrs, target, min_size):
    """Analyze SHAP feature contributions for bias across groups."""
    shdr(4, "Feature Contribution Bias (SHAP)")
    results = {"status": "PASS", "shap_analysis": {}, "issues": []}

    if not HAS_SHAP:
        p_skip("SHAP not installed — pip install shap")
        results["status"] = "SKIP"
        return results

    # Calculate SHAP values
    try:
        # Determine explainer type
        model_type = type(model).__name__
        if hasattr(model, "estimators_") or "Gradient" in model_type or "Forest" in model_type or "XGB" in model_type:
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other model types (sample background data)
            sample_size = min(100, len(X))
            background = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X
            explainer = shap.KernelExplainer(model.predict, background)

        shap_values = explainer.shap_values(X)

        # Handle multi-output SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        p_info(f"SHAP values computed ({model_type})")

    except Exception as e:
        p_warn(f"SHAP analysis failed: {e}")
        results["status"] = "SKIP"
        return results

    # Global feature importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_cols, mean_shap), key=lambda x: x[1], reverse=True)
    results["shap_analysis"]["global_ranking"] = [
        {"feature": f, "mean_abs_shap": round(float(s), 4)} for f, s in ranking[:10]
    ]

    print(f"  {Style.BRIGHT}Global SHAP Feature Importance:{Style.RESET_ALL}")
    for i, (f, v) in enumerate(ranking[:8], 1):
        bar = "█" * int(v / max(mean_shap) * 20)
        print(f"    {i:2d}. {f:25s}  SHAP={v:.4f}  {Fore.CYAN}{bar}{Style.RESET_ALL}")

    # Per-group SHAP analysis
    print(f"\n  {Style.BRIGHT}Per-Group Feature Importance Comparison:{Style.RESET_ALL}")

    for attr in protected_attrs:
        groups = df[attr].dropna().unique()
        valid = [g for g in groups if (df[attr] == g).sum() >= min_size]

        if len(valid) < 2:
            p_skip(f"'{attr}': <2 valid groups")
            continue

        print(f"\n    {Style.BRIGHT}Across '{attr}':{Style.RESET_ALL}")
        group_top_features = {}

        for g in valid:
            mask = (df[attr] == g).values
            g_shap = np.abs(shap_values[mask]).mean(axis=0)
            g_ranking = sorted(zip(feature_cols, g_shap), key=lambda x: x[1], reverse=True)
            group_top_features[str(g)] = g_ranking[:5]

        # Check if top features differ significantly between groups
        attr_results = {"groups": {}}
        for g in valid:
            top3 = [f for f, _ in group_top_features[str(g)][:3]]
            vals = [round(float(v), 4) for _, v in group_top_features[str(g)][:3]]
            attr_results["groups"][str(g)] = {
                "top_3_features": top3,
                "top_3_shap": vals,
            }
            print(f"      {Style.BRIGHT}{g}:{Style.RESET_ALL} "
                  f"top → {', '.join(f'{f}({v:.3f})' for f, v in group_top_features[str(g)][:3])}")

        # Compare top feature rankings between groups
        all_tops = [set(f for f, _ in group_top_features[str(g)][:3]) for g in valid]
        overlap = set.intersection(*all_tops) if all_tops else set()
        consistency = len(overlap) / 3.0 if len(all_tops) > 0 else 1.0

        if consistency < 0.33:
            p_warn(f"Feature importance differs significantly across '{attr}' groups "
                   f"(only {len(overlap)}/3 top features shared)")
            if results["status"] == "PASS":
                results["status"] = "WARN"
            results["issues"].append(f"SHAP ranking diverges for '{attr}'")
        else:
            p_pass(f"Feature importance consistent across '{attr}' groups "
                   f"({len(overlap)}/3 top features shared)")

        results["shap_analysis"][attr] = attr_results

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 5 — COUNTERFACTUAL FAIRNESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer5(df, model, X, feature_cols, protected_attrs, target, binned_originals, min_size):
    """Test counterfactual fairness: flip protected attributes and observe prediction changes."""
    shdr(5, "Counterfactual Fairness Test")
    results = {"status": "PASS", "counterfactuals": {}, "issues": []}

    # Check which protected attributes are actually in the feature set
    # (they could be excluded from features)
    prot_in_features = []
    orig_prot = set()
    for attr in protected_attrs:
        orig = attr.replace("_group", "")
        if attr in feature_cols:
            prot_in_features.append(attr)
            orig_prot.add(attr)
        elif orig in feature_cols:
            prot_in_features.append(orig)
            orig_prot.add(orig)

    if not prot_in_features:
        p_pass("Protected attributes are NOT used as input features — counterfactual test not applicable.")
        p_info("This is actually good practice! The model cannot directly use protected attributes.")
        results["status"] = "PASS"
        results["counterfactuals"]["note"] = "Protected attributes excluded from features"
        return results

    p_warn(f"Protected attributes found in features: {', '.join(prot_in_features)}")
    p_info("Testing what happens when we flip these values...")

    original_preds = model.predict(X)

    for prot_feat in prot_in_features:
        attr_name = prot_feat
        unique_vals = X[prot_feat].unique()

        if len(unique_vals) < 2:
            p_skip(f"'{prot_feat}': only 1 unique value in features — cannot flip")
            continue

        print(f"\n  {Style.BRIGHT}Flipping '{prot_feat}':{Style.RESET_ALL}")

        total_flips = 0
        total_tested = 0

        # For each unique value, set ALL rows to that value and see how predictions change
        flip_rates = {}
        for val in unique_vals:
            X_counterfactual = X.copy()
            X_counterfactual[prot_feat] = val

            new_preds = model.predict(X_counterfactual)
            flipped = (new_preds != original_preds).sum()
            n_tested = len(new_preds)
            flip_rate = flipped / n_tested if n_tested > 0 else 0.0

            flip_rates[str(val)] = round(flip_rate, 4)
            total_flips += flipped
            total_tested += n_tested

            print(f"    Set all → '{val}': {flipped}/{n_tested} predictions changed ({flip_rate:.1%})")

        avg_flip = total_flips / total_tested if total_tested > 0 else 0.0
        results["counterfactuals"][attr_name] = {
            "flip_rates_per_value": flip_rates,
            "average_flip_rate": round(avg_flip, 4),
        }

        if avg_flip > COUNTERFACTUAL_THRESHOLD:
            p_fail(
                f"Average flip rate: {avg_flip:.1%}  (> {COUNTERFACTUAL_THRESHOLD:.0%})  "
                f"— model IS sensitive to '{prot_feat}'"
            )
            results["status"] = "FAIL"
            results["issues"].append(
                f"'{prot_feat}' flips {avg_flip:.0%} of predictions when changed"
            )
        else:
            p_pass(
                f"Average flip rate: {avg_flip:.1%}  (≤ {COUNTERFACTUAL_THRESHOLD:.0%})  "
                f"— model is robust to '{prot_feat}' changes"
            )

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 6 — HUMAN-READABLE VERDICT & RECOMMENDATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer6(l1, l2, l3, l4, l5):
    """Generate human-readable explanations and actionable recommendations."""
    shdr(6, "Human-Readable Verdict & Recommendations")
    results = {"recommendations": [], "explanations": []}

    def emit(color, expl, rec):
        print(f"  {color}▸{Style.RESET_ALL} {expl}")
        p_rec(rec)
        print()
        results["explanations"].append(expl)
        results["recommendations"].append(rec)

    # L1 recommendations
    for attr, m in l1.get("metrics", {}).items():
        if m.get("skipped"):
            continue

        di = m.get("disparate_impact", {})
        if di.get("pass") is False:
            emit(
                Fore.RED,
                f"Model's Disparate Impact on '{attr}' = {di['value']:.2f} (threshold {DI_THRESHOLD}).\n"
                f"    The model approves the unprivileged group at only {di['value']:.0%} the rate of "
                f"the privileged group ('{di.get('privileged', '?')}').",
                f"Retrain with fairness constraints, apply post-hoc threshold adjustment for '{attr}', "
                f"or use reweighting during training.",
            )

        dp = m.get("demographic_parity", {})
        if dp.get("pass") is False:
            emit(
                Fore.RED,
                f"Demographic Parity Gap on '{attr}' = {dp['value']:+.4f} (threshold ±{DP_THRESHOLD}).\n"
                f"    The model's positive prediction rates differ significantly between groups.",
                f"Apply group-specific threshold calibration to equalise positive rates for '{attr}'.",
            )

        eod = m.get("equal_opportunity", {})
        if eod.get("pass") is False:
            emit(
                Fore.RED,
                f"Equal Opportunity Diff on '{attr}' = {eod['value']:+.4f} (threshold ±{EO_THRESHOLD}).\n"
                f"    Qualified individuals in the unprivileged group are missed more often.",
                f"Retrain with equalized-odds constraints or post-hoc calibrate TPR for '{attr}'.",
            )

        ppd = m.get("predictive_parity", {})
        if ppd.get("pass") is False:
            emit(
                Fore.RED,
                f"Predictive Parity Diff on '{attr}' = {ppd['value']:+.4f} (threshold ±{PP_THRESHOLD}).\n"
                f"    When the model predicts positive for the unprivileged group, it is less reliable.",
                f"Calibrate predicted probabilities per group or adjust classification thresholds for '{attr}'.",
            )

    # L2 recommendations
    for attr, rates in l2.get("error_rates", {}).items():
        if isinstance(rates, dict) and rates.get("skipped"):
            continue
    for issue in l2.get("issues", []):
        if "FPR" in issue:
            emit(
                Fore.RED,
                f"False Positive Rate disparity: {issue}.\n"
                f"    Some groups are wrongly flagged/approved at higher rates.",
                f"Investigate training data balance and consider cost-sensitive learning "
                f"to equalize false positive rates.",
            )
        elif "FNR" in issue:
            emit(
                Fore.RED,
                f"False Negative Rate disparity: {issue}.\n"
                f"    Some groups' positive cases are being missed more often.",
                f"Apply equalized-odds post-processing or resample training data "
                f"to equalize false negative rates.",
            )

    # L3 recommendations
    for issue in l3.get("issues", []):
        emit(
            Fore.YELLOW,
            f"Calibration issue: {issue}.\n"
            f"    The model's confidence scores are not equally reliable across groups.",
            f"Apply Platt scaling or isotonic regression calibration per group.",
        )

    # L4 recommendations
    for issue in l4.get("issues", []):
        emit(
            Fore.YELLOW,
            f"Feature importance divergence: {issue}.\n"
            f"    The model relies on different features depending on group membership,\n"
            f"    suggesting it may have learned group-specific shortcuts.",
            f"Audit the divergent features and consider feature selection constraints.",
        )

    # L5 recommendations
    for issue in l5.get("issues", []):
        emit(
            Fore.RED,
            f"Counterfactual violation: {issue}.\n"
            f"    Changing the protected attribute directly changes the model's decision,\n"
            f"    which is a strong indicator of direct discrimination.",
            f"Remove the protected attribute from model inputs. If needed, use adversarial "
            f"debiasing to ensure the model cannot reconstruct group membership.",
        )

    if not results["recommendations"]:
        p_pass("No actionable recommendations — all model metrics within bounds.")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY & OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_summary(l1, l2, l3, l4, l5, l6):
    """Print the final summary table."""
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'═' * 62}")
    print(f"  FINAL MODEL FAIRNESS SUMMARY")
    print(f"{'═' * 62}{Style.RESET_ALL}\n")

    layers = [
        ("L1", "Prediction Fairness", l1),
        ("L2", "Error Rate Disparity", l2),
        ("L3", "Calibration", l3),
        ("L4", "Feature Contribution", l4),
        ("L5", "Counterfactual", l5),
    ]

    print(f"  {'Layer':<6}{'Component':<22}{'Status':<12}{'Issues'}")
    print(f"  {'─' * 6}{'─' * 22}{'─' * 12}{'─' * 26}")

    fails = warns = 0
    for tag, name, r in layers:
        st = r.get("status", "PASS")
        iss = r.get("issues", [])
        istr = (
            "; ".join(iss[:2]) + (f" (+{len(iss) - 2} more)" if len(iss) > 2 else "")
        ) if iss else "—"
        col = (
            Fore.RED if st == "FAIL"
            else Fore.YELLOW if st == "WARN"
            else Fore.CYAN if st == "SKIP"
            else Fore.GREEN
        )
        if st == "FAIL":
            fails += 1
        if st == "WARN":
            warns += 1
        print(f"  {tag:<6}{name:<22}{col}{st:<12}{Style.RESET_ALL}{istr}")

    nr = len(l6.get("recommendations", []))
    print(f"  {'L6':<6}{'Recommendations':<22}{Fore.CYAN}{'—':<12}{Style.RESET_ALL}{nr} recommendation(s)")

    print(f"\n  {'─' * 58}")

    if fails > 0:
        verdict, icon, col = "BIASED", "❌", Fore.RED
    elif warns > 0:
        verdict, icon, col = "BORDERLINE", "⚠ ", Fore.YELLOW
    else:
        verdict, icon, col = "FAIR", "✅", Fore.GREEN

    parts = []
    if fails:
        parts.append(f"{fails} layer(s) failed")
    if warns:
        parts.append(f"{warns} warning(s)")
    summ = ", ".join(parts) if parts else "All layers passed"

    print(f"\n  {Style.BRIGHT}OVERALL MODEL VERDICT: {col}{icon}  {verdict}{Style.RESET_ALL}  — {summ}\n")
    return verdict


def save_results(verdict, l1, l2, l3, l4, l5, l6, output_dir):
    """Save all results to JSON."""
    out = os.path.join(output_dir, "modelsight_results.json")
    try:
        with open(out, "w") as f:
            json.dump(
                {
                    "modelsight_version": "1.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "overall_verdict": verdict,
                    "layer1_prediction_fairness": l1,
                    "layer2_error_rate_disparity": l2,
                    "layer3_calibration": l3,
                    "layer4_feature_contribution": l4,
                    "layer5_counterfactual": l5,
                    "layer6_recommendations": l6,
                },
                f,
                indent=2,
                cls=NumpyEncoder,
            )
        p_info(f"Results saved → {out}")
    except Exception as e:
        p_warn(f"Could not save results: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    init(autoreset=False)
    ap = argparse.ArgumentParser(
        description="ModelSight CLI — Model Bias Detection for trained ML models"
    )
    ap.add_argument("--model", required=True, help="Path to trained model file (.pkl, .joblib)")
    ap.add_argument("--csv", required=True, help="Path to test dataset CSV")
    ap.add_argument("--target", required=True, help="Target column name (the column the model predicts)")
    ap.add_argument(
        "--protected", nargs="*", default=None,
        help="Protected attribute column names (auto-detected if not specified)"
    )
    args = ap.parse_args()

    # ── Validate inputs ──────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"{Fore.RED}Error: Model file '{args.model}' not found.{Style.RESET_ALL}")
        sys.exit(1)
    if not os.path.exists(args.csv):
        print(f"{Fore.RED}Error: CSV file '{args.csv}' not found.{Style.RESET_ALL}")
        sys.exit(1)

    output_dir = os.path.dirname(os.path.abspath(args.csv))
    banner()

    # ── Load model ───────────────────────────────────────────────────
    p_info(f"Loading model: {args.model}")
    model = load_model(args.model)
    model_type = type(model).__name__
    p_info(f"Model type: {model_type}")

    # ── Load test data ───────────────────────────────────────────────
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"{Fore.RED}CSV read error: {e}{Style.RESET_ALL}")
        sys.exit(1)

    min_size = dynamic_min_size(len(df))

    print(f"\n  {Style.BRIGHT}Test Dataset:{Style.RESET_ALL}    {args.csv}")
    print(f"  {Style.BRIGHT}Rows:{Style.RESET_ALL}             {len(df):,}")
    print(f"  {Style.BRIGHT}Columns:{Style.RESET_ALL}          {len(df.columns)}  ({', '.join(df.columns)})")
    print(f"  {Style.BRIGHT}Min group size:{Style.RESET_ALL}   {min_size} (auto-scaled)")

    # Handle missing values
    miss = df.columns[df.isnull().any()].tolist()
    if miss:
        p_warn(f"Missing values in: {', '.join(miss)}")

    # ── Detect / validate target ─────────────────────────────────────
    if args.target not in df.columns:
        print(f"{Fore.RED}Error: Target '{args.target}' not found in CSV. "
              f"Available: {list(df.columns)}{Style.RESET_ALL}")
        sys.exit(1)
    target = args.target
    print(f"  {Style.BRIGHT}Target column:{Style.RESET_ALL}    {target}")

    # ── Detect protected attributes ──────────────────────────────────
    if args.protected:
        protected_attrs = [p for p in args.protected if p in df.columns]
        missing_p = [p for p in args.protected if p not in df.columns]
        if missing_p:
            p_warn(f"Protected columns not found: {', '.join(missing_p)}")
    else:
        protected_attrs = detect_protected(df)

    if not protected_attrs:
        print(f"{Fore.RED}No protected attributes found. "
              f"Use --protected to specify them manually.{Style.RESET_ALL}")
        sys.exit(1)
    print(f"  {Style.BRIGHT}Protected attrs:{Style.RESET_ALL}  {', '.join(protected_attrs)}")

    # ── Bin numeric protected attributes ──────────────────────────────
    df, protected_attrs, binned_info = bin_numeric_protected(df, protected_attrs)
    binned_originals = set(binned_info.keys())

    # ── Binarise target ──────────────────────────────────────────────
    df, pos_label, trivial = binarise_target(df, target)
    print(f"  {Style.BRIGHT}Positive label:{Style.RESET_ALL}   '{pos_label}'")
    print(f"  {Style.BRIGHT}Actual pos rate:{Style.RESET_ALL}  {df[target].mean():.1%}")

    if trivial:
        p_warn("Target has only 1 value — model fairness metrics will be limited.")

    # ── Prepare features and get predictions ─────────────────────────
    p_info("Preparing features for model prediction...")
    X, feature_cols, encoders = prepare_features(df, protected_attrs, target, binned_originals, model=model)

    try:
        predictions, probabilities = get_predictions(model, X)
        pred_rate = predictions.mean()
        p_info(f"Model prediction rate: {pred_rate:.1%} positive")

        if HAS_SKLEARN:
            acc = accuracy_score(df[target].values, predictions)
            p_info(f"Model accuracy: {acc:.1%}")
    except Exception as e:
        print(f"{Fore.RED}Error: Model prediction failed: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Tip: Ensure the CSV has the same columns the model was trained on.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Feature columns used: {feature_cols}{Style.RESET_ALL}")
        sys.exit(1)

    # ── Run all 6 layers ─────────────────────────────────────────────
    l1_results = layer1(df, protected_attrs, target, predictions, min_size)
    l2_results = layer2(df, protected_attrs, target, predictions, min_size)
    l3_results = layer3(df, protected_attrs, target, probabilities, min_size)
    l4_results = layer4(df, model, X, feature_cols, protected_attrs, target, min_size)
    l5_results = layer5(df, model, X, feature_cols, protected_attrs, target, binned_originals, min_size)
    l6_results = layer6(l1_results, l2_results, l3_results, l4_results, l5_results)

    # ── Print summary ────────────────────────────────────────────────
    verdict = print_summary(l1_results, l2_results, l3_results, l4_results, l5_results, l6_results)
    save_results(verdict, l1_results, l2_results, l3_results, l4_results, l5_results, l6_results, output_dir)
    print(f"{Fore.MAGENTA}{'━' * 62}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
