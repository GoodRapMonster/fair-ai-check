#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║  FairSight CLI — Terminal-Based Bias Detection Tool              ║
║  6-Layer fairness analysis pipeline for tabular datasets         ║
╚═══════════════════════════════════════════════════════════════════╝

Usage:
    python fairsight.py --csv data.csv
    python fairsight.py --csv data.csv --target loan_approved
    python fairsight.py --csv data.csv --history fairsight_history.json
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pointbiserialr
from colorama import Fore, Style, Back, init

# Optional ML libraries
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import LabelEncoder
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
DI_THRESHOLD = 0.8
DP_THRESHOLD = 0.1
EO_THRESHOLD = 0.1
PP_THRESHOLD = 0.1
INTERSECTIONAL_DEVIATION = 0.15
HIGH_PROXY_CORR = 0.7
MEDIUM_PROXY_CORR = 0.5
DISTRIBUTION_SD_GAP = 1.0
DRIFT_THRESHOLD = 0.05
MIN_SUBGROUP_SIZE = 30


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON ENCODER FOR NUMPY TYPES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types during JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRETTY PRINTING HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def banner():
    """Print the FairSight banner."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║        ⚖️  FairSight CLI — Bias Detection Tool       ║")
    print("  ║        Comprehensive 6-Layer Fairness Audit         ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)


def section_header(layer_num, title):
    """Print a styled section header."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'━' * 62}")
    print(f"  LAYER {layer_num} — {title}")
    print(f"{'━' * 62}{Style.RESET_ALL}\n")


def p_pass(text):
    print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}  {text}")


def p_fail(text):
    print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL}  {text}")


def p_warn(text):
    print(f"  {Fore.YELLOW}⚠ WARN{Style.RESET_ALL}  {text}")


def p_info(text):
    print(f"  {Fore.CYAN}ℹ INFO{Style.RESET_ALL}  {text}")


def p_rec(text):
    """Print a recommendation."""
    print(f"  {Fore.MAGENTA}→ REC {Style.RESET_ALL}  {text}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTO-DETECTION UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_protected_attributes(df):
    """Auto-detect protected attributes by scanning column names."""
    protected = []
    for col in df.columns:
        col_lower = col.lower().strip()
        for kw in PROTECTED_KEYWORDS:
            if kw in col_lower:
                protected.append(col)
                break
    return protected


def bin_numeric_protected(df, protected_attrs):
    """
    For numeric protected attributes with many unique values (e.g. age),
    create a binned version so fairness analysis uses meaningful groups
    instead of hundreds of single-value groups.
    Returns (modified_df, updated_protected_attrs, binning_info).
    """
    df = df.copy()
    new_attrs = []
    binning_info = {}

    for attr in protected_attrs:
        if pd.api.types.is_numeric_dtype(df[attr]) and df[attr].nunique() > 10:
            # Bin into meaningful ranges
            col_lower = attr.lower()
            if "age" in col_lower:
                bins = [0, 25, 35, 45, 55, 65, 120]
                labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
            else:
                # Generic quintile binning
                df[f"{attr}_group"], bin_edges = pd.qcut(
                    df[attr], q=4, retbins=True, duplicates="drop"
                )
                df[f"{attr}_group"] = df[f"{attr}_group"].astype(str)
                new_attrs.append(f"{attr}_group")
                binning_info[attr] = f"Binned into quartiles → {attr}_group"
                p_info(f"Binned numeric '{attr}' into quartiles → '{attr}_group'")
                continue

            df[f"{attr}_group"] = pd.cut(
                df[attr], bins=bins, labels=labels, right=True
            ).astype(str)
            new_attrs.append(f"{attr}_group")
            binning_info[attr] = f"Binned into {labels} → {attr}_group"
            p_info(f"Binned numeric '{attr}' into age ranges → '{attr}_group'")
        else:
            new_attrs.append(attr)

    return df, new_attrs, binning_info


def detect_target_column(df, target_arg=None):
    """Detect or validate the target column."""
    if target_arg:
        if target_arg in df.columns:
            return target_arg
        print(f"\n{Fore.RED}  Error: Target column '{target_arg}' not found in CSV.{Style.RESET_ALL}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)
    # Default: last column
    return df.columns[-1]


def identify_groups(df, attr, target):
    """
    Identify privileged (highest positive rate) and unprivileged groups.
    Returns (privileged_value, dict_of_group_rates).
    """
    rates = df.groupby(attr)[target].mean()
    privileged = rates.idxmax()
    return privileged, rates.to_dict()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 1 — STANDARD FAIRNESS METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer1(df, protected_attrs, target):
    """Compute 4 standard fairness metrics for each protected attribute."""
    section_header(1, "Standard Fairness Metrics")

    results = {"status": "PASS", "metrics": {}, "issues": []}

    # ── Train a model for EOD / PPD ──────────────────────────────────
    predictions = None
    if HAS_SKLEARN:
        try:
            feature_cols = [
                c for c in df.columns
                if c not in protected_attrs and c != target
            ]
            X = df[feature_cols].copy()
            y = df[target].values

            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = X.fillna(X.median())

            if len(np.unique(y)) >= 2:
                clf = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, random_state=42
                )
                predictions = cross_val_predict(clf, X, y, cv=5)
                p_info(f"Trained GBM for Equal Opportunity & Predictive Parity (5-fold CV)")
        except Exception as exc:
            p_warn(f"Could not train model for EOD/PPD: {exc}")
    else:
        p_warn("scikit-learn not available — EOD and PPD will be skipped")

    actual = df[target].values

    for attr in protected_attrs:
        print(f"\n  {Style.BRIGHT}Protected attribute: {attr}{Style.RESET_ALL}")
        groups = df[attr].unique()
        n_groups = len(groups)

        # Warn on small groups
        for g in groups:
            cnt = (df[attr] == g).sum()
            if cnt < MIN_SUBGROUP_SIZE:
                p_warn(f"Group '{g}' has only {cnt} samples (< {MIN_SUBGROUP_SIZE})")

        privileged, group_rates = identify_groups(df, attr, target)
        priv_mask = (df[attr] == privileged).values
        unpriv_mask = ~priv_mask

        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            p_pass(f"Group uniform — no unprivileged comparison possible. Passing by default.")
            results["metrics"][attr] = {
                "disparate_impact": {"value": 1.0, "pass": True},
                "demographic_parity": {"value": 0.0, "pass": True},
                "equal_opportunity": {"value": 0.0, "pass": True},
                "predictive_parity": {"value": 0.0, "pass": True},
            }
            continue

        priv_rate = df.loc[priv_mask, target].mean()
        unpriv_rate = df.loc[unpriv_mask, target].mean()

        attr_metrics = {}

        # 1. Disparate Impact Ratio
        di = unpriv_rate / priv_rate if priv_rate > 0 else 0.0
        di = 1.0 if np.isnan(di) else di
        di_ok = di >= DI_THRESHOLD or np.isnan(di)
        attr_metrics["disparate_impact"] = {
            "value": round(di, 4),
            "threshold": DI_THRESHOLD,
            "pass": di_ok,
            "privileged": str(privileged),
            "priv_rate": round(priv_rate, 4),
            "unpriv_rate": round(unpriv_rate, 4),
        }
        (p_pass if di_ok else p_fail)(
            f"Disparate Impact: {di:.4f}  "
            f"({'≥' if di_ok else '<'} {DI_THRESHOLD})  "
            f"[priv={privileged} {priv_rate:.1%} | unpriv {unpriv_rate:.1%}]"
        )
        if not di_ok:
            results["status"] = "FAIL"
            results["issues"].append(f"DI={di:.2f} on {attr}")

        # 2. Demographic Parity Gap
        dp = unpriv_rate - priv_rate
        dp = 0.0 if np.isnan(dp) else dp
        dp_ok = abs(dp) <= DP_THRESHOLD
        attr_metrics["demographic_parity"] = {
            "value": round(dp, 4),
            "threshold": DP_THRESHOLD,
            "pass": dp_ok,
        }
        (p_pass if dp_ok else p_fail)(
            f"Demographic Parity Gap: {dp:+.4f}  "
            f"({'|gap| ≤' if dp_ok else '|gap| >'} {DP_THRESHOLD})"
        )
        if not dp_ok:
            results["status"] = "FAIL"
            results["issues"].append(f"DP={dp:+.2f} on {attr}")

        # 3. Equal Opportunity Difference (needs model predictions)
        if predictions is not None:
            priv_actual_pos = actual[priv_mask] == 1
            unpriv_actual_pos = actual[unpriv_mask] == 1

            priv_tpr = (
                predictions[priv_mask][priv_actual_pos].mean()
                if priv_actual_pos.sum() > 0
                else 0.0
            )
            unpriv_tpr = (
                predictions[unpriv_mask][unpriv_actual_pos].mean()
                if unpriv_actual_pos.sum() > 0
                else 0.0
            )
            eod = unpriv_tpr - priv_tpr
            eod = 0.0 if np.isnan(eod) else eod
            eod_ok = abs(eod) <= EO_THRESHOLD
            attr_metrics["equal_opportunity"] = {
                "value": round(eod, 4),
                "threshold": EO_THRESHOLD,
                "pass": eod_ok,
                "priv_tpr": round(priv_tpr, 4),
                "unpriv_tpr": round(unpriv_tpr, 4),
            }
            (p_pass if eod_ok else p_fail)(
                f"Equal Opportunity Diff: {eod:+.4f}  "
                f"({'|diff| ≤' if eod_ok else '|diff| >'} {EO_THRESHOLD})"
            )
            if not eod_ok:
                results["status"] = "FAIL"
                results["issues"].append(f"EOD={eod:+.2f} on {attr}")
        else:
            attr_metrics["equal_opportunity"] = {"value": None, "pass": None}

        # 4. Predictive Parity Difference
        if predictions is not None:
            priv_pred_pos = predictions[priv_mask] == 1
            unpriv_pred_pos = predictions[unpriv_mask] == 1

            priv_ppv = (
                actual[priv_mask][priv_pred_pos].mean()
                if priv_pred_pos.sum() > 0
                else 1.0 # default to 1 if no positive predictions
            )
            unpriv_ppv = (
                actual[unpriv_mask][unpriv_pred_pos].mean()
                if unpriv_pred_pos.sum() > 0
                else 1.0
            )
            ppd = unpriv_ppv - priv_ppv
            ppd = 0.0 if np.isnan(ppd) else ppd
            ppd_ok = abs(ppd) <= PP_THRESHOLD
            attr_metrics["predictive_parity"] = {
                "value": round(ppd, 4),
                "threshold": PP_THRESHOLD,
                "pass": ppd_ok,
                "priv_ppv": round(priv_ppv, 4),
                "unpriv_ppv": round(unpriv_ppv, 4),
            }
            (p_pass if ppd_ok else p_fail)(
                f"Predictive Parity Diff: {ppd:+.4f}  "
                f"({'|diff| ≤' if ppd_ok else '|diff| >'} {PP_THRESHOLD})"
            )
            if not ppd_ok:
                results["status"] = "FAIL"
                results["issues"].append(f"PPD={ppd:+.2f} on {attr}")
        else:
            attr_metrics["predictive_parity"] = {"value": None, "pass": None}

        results["metrics"][attr] = attr_metrics

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 2 — INTERSECTIONAL ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer2(df, protected_attrs, target):
    """Intersectional bias analysis across combinations of protected attributes."""
    section_header(2, "Intersectional Analysis")

    results = {"status": "PASS", "subgroups": [], "issues": [], "worst": None}

    if len(protected_attrs) < 2:
        p_info("Only one protected attribute detected — skipping intersectional analysis.")
        results["status"] = "SKIP"
        return results

    overall_rate = df[target].mean()
    p_info(f"Overall positive rate: {overall_rate:.1%}")

    worst_deviation = 0.0
    worst_group = None
    flagged_entries = []  # collect all flagged subgroups for sorted reporting

    # Only analyze pairwise combinations (2-way) to keep output manageable
    for combo in combinations(protected_attrs, 2):
        combo_label = " × ".join(combo)
        p_info(f"Analyzing intersection: {combo_label}")

        grouped = df.groupby(list(combo))[target]
        small_count = 0
        for name, group in grouped:
            if isinstance(name, str):
                name = (name,)
            sub_label = "+".join(str(v) for v in name)
            n = len(group)
            rate = group.mean()
            deviation = abs(rate - overall_rate)

            entry = {
                "intersection": combo_label,
                "subgroup": sub_label,
                "n": int(n),
                "rate": round(rate, 4),
                "overall_rate": round(overall_rate, 4),
                "deviation": round(deviation, 4),
            }

            # Skip very small subgroups from output (still record them)
            if n < MIN_SUBGROUP_SIZE:
                entry["small_sample"] = True
                small_count += 1
                results["subgroups"].append(entry)
                # Still track worst even if small
                if deviation > worst_deviation:
                    worst_deviation = deviation
                    worst_group = {
                        "subgroup": sub_label,
                        "rate": round(rate, 4),
                        "deviation": round(deviation, 4),
                        "n": int(n),
                        "small_sample": True,
                    }
                continue

            if deviation > INTERSECTIONAL_DEVIATION:
                flag = "below" if rate < overall_rate else "above"
                flagged_entries.append((deviation, sub_label, rate, flag, n))
                results["status"] = "FAIL"
                entry["flagged"] = True
                results["issues"].append(
                    f"{sub_label}: {rate:.0%} vs {overall_rate:.0%} overall"
                )
            else:
                p_pass(f"  {sub_label}: {rate:.1%}  (n={n}, within threshold)")

            if deviation > worst_deviation:
                worst_deviation = deviation
                worst_group = {
                    "subgroup": sub_label,
                    "rate": round(rate, 4),
                    "deviation": round(deviation, 4),
                    "n": int(n),
                }

            results["subgroups"].append(entry)

        if small_count > 0:
            p_warn(f"  {small_count} subgroup(s) skipped (n < {MIN_SUBGROUP_SIZE})")

    # Print flagged subgroups sorted by severity
    if flagged_entries:
        flagged_entries.sort(reverse=True)
        print(f"\n  {Style.BRIGHT}Flagged intersectional subgroups (by severity):{Style.RESET_ALL}")
        for dev, label, rate, direction, n in flagged_entries[:10]:  # top 10
            p_fail(
                f"  {label}: {rate:.1%} positive rate vs "
                f"{overall_rate:.1%} overall ({direction} by {dev:.1%}, n={n})"
            )
        if len(flagged_entries) > 10:
            p_info(f"  ... and {len(flagged_entries) - 10} more flagged subgroups")

    if worst_group:
        results["worst"] = worst_group
        small_note = " ⚠ small sample" if worst_group.get("small_sample") else ""
        print(
            f"\n  {Style.BRIGHT}Worst affected group:{Style.RESET_ALL} "
            f"{worst_group['subgroup']}  "
            f"(rate={worst_group['rate']:.1%}, "
            f"deviation={worst_group['deviation']:.1%}, "
            f"n={worst_group.get('n', '?')}){small_note}"
        )

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 3 — PROXY DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _encode_column(series):
    """Encode a column to numeric for correlation. Returns array or None."""
    if pd.api.types.is_numeric_dtype(series):
        return series.dropna().values, series.dropna().index
    if series.dtype in ("object", "category"):
        if series.nunique() <= 2:
            vals = series.astype("category").cat.codes
            mask = vals >= 0
            return vals[mask].values.astype(float), vals[mask].index
    return None, None


def layer3(df, protected_attrs, target, binned_originals=None):
    """Detect proxy variables via correlation and SHAP analysis."""
    section_header(3, "Proxy Variable Detection")

    binned_originals = binned_originals or set()
    results = {"status": "PASS", "proxies": [], "shap_top": [], "issues": []}
    non_protected = [
        c for c in df.columns
        if c not in protected_attrs and c != target and c not in binned_originals
    ]

    all_proxies = []

    for attr in protected_attrs:
        attr_encoded, attr_idx = _encode_column(df[attr])
        if attr_encoded is None:
            # Multi-class categorical: one-vs-rest encoding
            for val in df[attr].unique():
                binary = (df[attr] == val).astype(float)
                attr_encoded_bin = binary.values
                for col in non_protected:
                    col_encoded, col_idx = _encode_column(df[col])
                    if col_encoded is None:
                        continue
                    common = df.index
                    try:
                        r_val, p_val = pointbiserialr(
                            attr_encoded_bin[common], col_encoded[common] if len(col_encoded) == len(df) else col_encoded
                        )
                    except Exception:
                        continue
                    if np.isnan(r_val):
                        continue
                    risk = None
                    if abs(r_val) > HIGH_PROXY_CORR and p_val < 0.05:
                        risk = "HIGH"
                    elif abs(r_val) > MEDIUM_PROXY_CORR and p_val < 0.05:
                        risk = "MEDIUM"
                    if risk:
                        all_proxies.append({
                            "feature": col,
                            "protected_attr": f"{attr}={val}",
                            "correlation": round(r_val, 4),
                            "p_value": round(p_val, 6),
                            "risk": risk,
                        })
            continue

        for col in non_protected:
            col_encoded, col_idx = _encode_column(df[col])
            if col_encoded is None:
                continue
            # Align indices
            common_idx = attr_idx.intersection(col_idx)
            if len(common_idx) < MIN_SUBGROUP_SIZE:
                continue
            a = df.loc[common_idx, attr] if attr_encoded is None else attr_encoded[np.isin(attr_idx, common_idx)]
            c = col_encoded[np.isin(col_idx, common_idx)]

            # Re-encode attr for common index
            attr_ser = df[attr].loc[common_idx]
            attr_enc2, _ = _encode_column(attr_ser)
            if attr_enc2 is None:
                continue

            try:
                r_val, p_val = pointbiserialr(attr_enc2, c)
            except Exception:
                try:
                    r_val, p_val = stats.pearsonr(attr_enc2.astype(float), c.astype(float))
                except Exception:
                    continue
            if np.isnan(r_val):
                continue

            risk = None
            if abs(r_val) > HIGH_PROXY_CORR and p_val < 0.05:
                risk = "HIGH"
            elif abs(r_val) > MEDIUM_PROXY_CORR and p_val < 0.05:
                risk = "MEDIUM"
            if risk:
                all_proxies.append({
                    "feature": col,
                    "protected_attr": attr,
                    "correlation": round(r_val, 4),
                    "p_value": round(p_val, 6),
                    "risk": risk,
                })

    # Deduplicate and keep highest correlation per feature-attr pair
    seen = {}
    for proxy in all_proxies:
        key = (proxy["feature"], proxy["protected_attr"])
        if key not in seen or abs(proxy["correlation"]) > abs(seen[key]["correlation"]):
            seen[key] = proxy
    all_proxies = sorted(seen.values(), key=lambda x: abs(x["correlation"]), reverse=True)

    # Report top 5
    top5 = all_proxies[:5]
    for p in top5:
        if p["risk"] == "HIGH":
            p_fail(
                f"{p['feature']} is {Fore.RED}HIGH RISK{Style.RESET_ALL} proxy for "
                f"{p['protected_attr']}  (r={p['correlation']:.4f}, p={p['p_value']:.2e})"
            )
            results["status"] = "WARN" if results["status"] != "FAIL" else "FAIL"
            results["issues"].append(
                f"{p['feature']} is HIGH RISK proxy for {p['protected_attr']} (r={p['correlation']:.2f})"
            )
        elif p["risk"] == "MEDIUM":
            p_warn(
                f"{p['feature']} is {Fore.YELLOW}MEDIUM RISK{Style.RESET_ALL} proxy for "
                f"{p['protected_attr']}  (r={p['correlation']:.4f}, p={p['p_value']:.2e})"
            )
            if results["status"] == "PASS":
                results["status"] = "WARN"
            results["issues"].append(
                f"{p['feature']} is MEDIUM RISK proxy for {p['protected_attr']} (r={p['correlation']:.2f})"
            )

    if not top5:
        p_pass("No significant proxy variables detected.")

    results["proxies"] = all_proxies

    # ── SHAP analysis ────────────────────────────────────────────────
    if HAS_SKLEARN and HAS_SHAP:
        try:
            p_info("Running SHAP feature importance analysis...")
            feature_cols = [
                c for c in df.columns if c not in protected_attrs and c != target
            ]
            X = df[feature_cols].copy()
            y = df[target].values

            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = X.fillna(X.median())

            clf = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=42
            )
            clf.fit(X, y)

            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X)

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_ranking = sorted(
                zip(feature_cols, mean_abs_shap),
                key=lambda x: x[1],
                reverse=True,
            )
            results["shap_top"] = [
                {"feature": f, "mean_abs_shap": round(s, 4)} for f, s in shap_ranking[:5]
            ]

            print()
            p_info("Top 5 features by SHAP importance:")
            for rank, (feat, val) in enumerate(shap_ranking[:5], 1):
                # Check if this feature is also a proxy
                is_proxy = any(p["feature"] == feat for p in all_proxies)
                flag = f" {Fore.RED}← PROXY{Style.RESET_ALL}" if is_proxy else ""
                print(f"    {rank}. {feat:20s}  SHAP={val:.4f}{flag}")
        except Exception as exc:
            p_warn(f"SHAP analysis failed: {exc}")
    elif not HAS_SHAP:
        p_info("SHAP not installed — skipping SHAP analysis")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 4 — FEATURE DISTRIBUTION ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer4(df, protected_attrs, target, binned_originals=None):
    """Analyze feature distributions across protected groups."""
    section_header(4, "Feature Distribution Analysis")

    binned_originals = binned_originals or set()
    results = {"status": "PASS", "distributions": [], "issues": []}

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in protected_attrs and c != target and c not in binned_originals
    ]

    if not numeric_cols:
        p_info("No numeric non-protected columns to analyze.")
        return results

    for attr in protected_attrs:
        groups = df[attr].unique()
        if len(groups) < 2:
            continue

        print(f"\n  {Style.BRIGHT}Comparing distributions across: {attr}{Style.RESET_ALL}")

        for col in numeric_cols:
            group_stats = {}
            for g in groups:
                vals = df.loc[df[attr] == g, col].dropna()
                group_stats[str(g)] = {
                    "mean": round(vals.mean(), 2) if len(vals) > 0 else None,
                    "std": round(vals.std(), 2) if len(vals) > 0 else None,
                    "n": len(vals),
                }

            # Compute max mean difference vs pooled std
            means = [s["mean"] for s in group_stats.values() if s["mean"] is not None]
            stds = [s["std"] for s in group_stats.values() if s["std"] is not None]

            if len(means) < 2 or not stds:
                continue

            max_diff = max(means) - min(means)
            pooled_std = np.mean(stds) if stds else 1.0
            sd_gap = max_diff / pooled_std if pooled_std > 0 else 0.0

            entry = {
                "feature": col,
                "protected_attr": attr,
                "group_stats": group_stats,
                "max_mean_diff": round(max_diff, 2),
                "pooled_std": round(pooled_std, 2),
                "sd_gap": round(sd_gap, 2),
            }

            if sd_gap > DISTRIBUTION_SD_GAP:
                # Find which groups differ most
                sorted_groups = sorted(group_stats.items(), key=lambda x: x[1]["mean"] if x[1]["mean"] is not None else 0)
                low_g = sorted_groups[0][0]
                high_g = sorted_groups[-1][0]
                p_warn(
                    f"  {col}: {sd_gap:.1f} SD gap between {attr} groups  "
                    f"[{low_g}={sorted_groups[0][1]['mean']:.1f} vs "
                    f"{high_g}={sorted_groups[-1][1]['mean']:.1f}]"
                )
                if results["status"] == "PASS":
                    results["status"] = "WARN"
                results["issues"].append(
                    f"{col} recorded with {sd_gap:.1f} SD gap between {attr} groups"
                )
                entry["flagged"] = True
            else:
                p_pass(f"  {col}: {sd_gap:.2f} SD gap  (within threshold)")

            results["distributions"].append(entry)

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 5 — FAIRNESS OVER TIME (DRIFT DETECTION)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer5(l1_results, history_path, output_dir):
    """Compare current fairness metrics to previous run for drift detection."""
    section_header(5, "Fairness Over Time — Drift Detection")

    results = {"status": "PASS", "trends": {}, "issues": []}

    # ── Save current results to history file ─────────────────────────
    history_out = os.path.join(output_dir, "fairsight_history.json")
    current_snapshot = {
        "timestamp": datetime.now().isoformat(),
        "metrics": l1_results.get("metrics", {}),
    }

    history_data = []
    if os.path.exists(history_out):
        try:
            with open(history_out, "r") as f:
                history_data = json.load(f)
            if not isinstance(history_data, list):
                history_data = [history_data]
        except Exception:
            history_data = []

    # ── Load separate history file if provided ───────────────────────
    prev = None
    if history_path and os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, list) and len(loaded) > 0:
                prev = loaded[-1]  # most recent
            elif isinstance(loaded, dict):
                prev = loaded
            p_info(f"Loaded previous results from: {history_path}")
        except Exception as exc:
            p_warn(f"Could not load history file: {exc}")
    elif not history_path:
        # Try auto-loading from default history
        if len(history_data) > 0:
            prev = history_data[-1]
            p_info(f"Auto-loaded previous run from: {history_out}")

    if prev is None:
        p_info("No previous results available — cannot compute drift.")
        results["status"] = "PASS"
        results["issues"].append("No history file provided")

        # Append and save
        history_data.append(current_snapshot)
        with open(history_out, "w") as f:
            json.dump(history_data, f, indent=2, cls=NumpyEncoder)
        p_info(f"Current results saved to: {history_out}")
        return results

    # ── Compare DI values ────────────────────────────────────────────
    prev_metrics = prev.get("metrics", {})
    curr_metrics = l1_results.get("metrics", {})
    prev_ts = prev.get("timestamp", "unknown")

    p_info(f"Comparing against run from: {prev_ts}")
    print()

    for attr, curr_m in curr_metrics.items():
        curr_di = curr_m.get("disparate_impact", {}).get("value")
        prev_di = prev_metrics.get(attr, {}).get("disparate_impact", {}).get("value")

        if curr_di is None or prev_di is None:
            continue

        delta = curr_di - prev_di
        if delta > 0.01:
            trend = "improving ↑"
        elif delta < -0.01:
            trend = "degrading ↓"
        else:
            trend = "stable ─"

        trend_entry = {
            "attr": attr,
            "current_di": round(curr_di, 4),
            "previous_di": round(prev_di, 4),
            "delta": round(delta, 4),
            "trend": trend,
        }
        results["trends"][attr] = trend_entry

        if delta < -DRIFT_THRESHOLD:
            p_fail(
                f"DI on {attr}: {prev_di:.4f} → {curr_di:.4f}  "
                f"(Δ={delta:+.4f}, {trend})  ⚠ DEGRADED"
            )
            results["status"] = "WARN"
            results["issues"].append(
                f"DI on {attr} degraded by {abs(delta):.3f}"
            )
        else:
            indicator = Fore.GREEN if "improving" in trend else Fore.YELLOW if "stable" in trend else Fore.RED
            p_pass(
                f"DI on {attr}: {prev_di:.4f} → {curr_di:.4f}  "
                f"(Δ={delta:+.4f}, {indicator}{trend}{Style.RESET_ALL})"
            )

    # Save updated history
    history_data.append(current_snapshot)
    with open(history_out, "w") as f:
        json.dump(history_data, f, indent=2, cls=NumpyEncoder)
    p_info(f"Current results appended to: {history_out}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 6 — VERDICT & RECOMMENDATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def layer6(l1, l2, l3, l4, l5):
    """Generate overall verdict, plain-English explanations, and recommendations."""
    section_header(6, "Human-Readable Verdict & Recommendations")

    results = {"recommendations": [], "explanations": []}

    # ── Explanations + recommendations for Layer 1 failures ──────────
    for attr, metrics in l1.get("metrics", {}).items():
        di_info = metrics.get("disparate_impact", {})
        if di_info.get("pass") is False:
            val = di_info["value"]
            priv = di_info.get("privileged", "unknown")
            explanation = (
                f"Disparate Impact on '{attr}' is {val:.2f} (threshold: {DI_THRESHOLD}).\n"
                f"    This means the unprivileged group receives positive outcomes at only "
                f"{val:.0%} the rate\n"
                f"    of the privileged group ('{priv}'). This falls below the 80% rule\n"
                f"    used in US federal guidelines for adverse impact."
            )
            rec = (
                f"Apply reweighting or SMOTE to balance training data for '{attr}'. "
                f"Consider collecting more representative data."
            )
            print(f"  {Fore.RED}▸{Style.RESET_ALL} {explanation}")
            p_rec(rec)
            print()
            results["explanations"].append(explanation)
            results["recommendations"].append(rec)

        dp_info = metrics.get("demographic_parity", {})
        if dp_info.get("pass") is False:
            val = dp_info["value"]
            explanation = (
                f"Demographic Parity Gap on '{attr}' is {val:+.4f} (threshold: ±{DP_THRESHOLD}).\n"
                f"    The rate of positive outcomes differs significantly between groups,\n"
                f"    indicating the decision process is not group-blind."
            )
            rec = (
                f"Apply threshold adjustment or post-processing calibration "
                f"to equalize positive outcome rates across '{attr}' groups."
            )
            print(f"  {Fore.RED}▸{Style.RESET_ALL} {explanation}")
            p_rec(rec)
            print()
            results["explanations"].append(explanation)
            results["recommendations"].append(rec)

        eod_info = metrics.get("equal_opportunity", {})
        if eod_info.get("pass") is False:
            val = eod_info["value"]
            explanation = (
                f"Equal Opportunity Difference on '{attr}' is {val:+.4f} (threshold: ±{EO_THRESHOLD}).\n"
                f"    Among truly qualified individuals, the model's true positive rate\n"
                f"    differs between groups — meaning qualified people in one group are\n"
                f"    less likely to be correctly identified."
            )
            rec = (
                f"Retrain the model with equalized odds constraints or apply "
                f"post-hoc calibration to equalize TPR across '{attr}' groups."
            )
            print(f"  {Fore.RED}▸{Style.RESET_ALL} {explanation}")
            p_rec(rec)
            print()
            results["explanations"].append(explanation)
            results["recommendations"].append(rec)

        ppd_info = metrics.get("predictive_parity", {})
        if ppd_info.get("pass") is False:
            val = ppd_info["value"]
            explanation = (
                f"Predictive Parity Difference on '{attr}' is {val:+.4f} (threshold: ±{PP_THRESHOLD}).\n"
                f"    The precision (positive predictive value) of the model differs\n"
                f"    between groups — positive predictions are less reliable for one group."
            )
            rec = (
                f"Apply calibration or threshold adjustment to equalize PPV "
                f"across '{attr}' groups."
            )
            print(f"  {Fore.RED}▸{Style.RESET_ALL} {explanation}")
            p_rec(rec)
            print()
            results["explanations"].append(explanation)
            results["recommendations"].append(rec)

    # ── Recommendations for intersectional issues ────────────────────
    if l2.get("worst"):
        wg = l2["worst"]
        explanation = (
            f"Intersectional bias detected: subgroup '{wg['subgroup']}' has a positive\n"
            f"    outcome rate of {wg['rate']:.1%}, deviating {wg['deviation']:.1%} from the overall rate.\n"
            f"    Bias may not be visible when looking at individual attributes alone."
        )
        rec = (
            f"Perform targeted data augmentation for the '{wg['subgroup']}' subgroup "
            f"and monitor intersectional metrics separately."
        )
        print(f"  {Fore.YELLOW}▸{Style.RESET_ALL} {explanation}")
        p_rec(rec)
        print()
        results["explanations"].append(explanation)
        results["recommendations"].append(rec)

    # ── Recommendations for proxy variables ──────────────────────────
    for proxy in l3.get("proxies", [])[:3]:  # top 3
        if proxy["risk"] == "HIGH":
            explanation = (
                f"Column '{proxy['feature']}' is a HIGH RISK proxy for "
                f"'{proxy['protected_attr']}'\n"
                f"    (correlation: {proxy['correlation']:.4f}). The model could use this\n"
                f"    feature to reconstruct protected group membership."
            )
            rec = (
                f"Remove or decorrelate column '{proxy['feature']}' before training. "
                f"Consider using adversarial debiasing."
            )
            print(f"  {Fore.RED}▸{Style.RESET_ALL} {explanation}")
            p_rec(rec)
            print()
            results["explanations"].append(explanation)
            results["recommendations"].append(rec)
        elif proxy["risk"] == "MEDIUM":
            explanation = (
                f"Column '{proxy['feature']}' is a MEDIUM RISK proxy for "
                f"'{proxy['protected_attr']}'\n"
                f"    (correlation: {proxy['correlation']:.4f}). Moderate association detected."
            )
            rec = (
                f"Monitor '{proxy['feature']}' — consider decorrelation or removal if "
                f"fairness cannot be achieved otherwise."
            )
            print(f"  {Fore.YELLOW}▸{Style.RESET_ALL} {explanation}")
            p_rec(rec)
            print()
            results["explanations"].append(explanation)
            results["recommendations"].append(rec)

    # ── Recommendations for distribution gaps ────────────────────────
    for dist in l4.get("distributions", []):
        if dist.get("flagged"):
            explanation = (
                f"Feature '{dist['feature']}' is recorded with a {dist['sd_gap']:.1f} SD gap\n"
                f"    between '{dist['protected_attr']}' groups. This suggests the data\n"
                f"    collection process may differ across groups."
            )
            rec = (
                f"Investigate data collection process for '{dist['feature']}'. "
                f"Consider standardization or separate calibration per group."
            )
            print(f"  {Fore.YELLOW}▸{Style.RESET_ALL} {explanation}")
            p_rec(rec)
            print()
            results["explanations"].append(explanation)
            results["recommendations"].append(rec)

    # ── Recommendations for drift ────────────────────────────────────
    for attr, trend in l5.get("trends", {}).items():
        if "degrading" in trend.get("trend", ""):
            explanation = (
                f"Fairness on '{attr}' has degraded since the last run.\n"
                f"    Disparate Impact dropped from {trend['previous_di']:.4f} to "
                f"{trend['current_di']:.4f} (Δ={trend['delta']:+.4f})."
            )
            rec = (
                f"Retrain model — fairness has degraded over time for '{attr}'. "
                f"Investigate recent data or model changes."
            )
            print(f"  {Fore.RED}▸{Style.RESET_ALL} {explanation}")
            p_rec(rec)
            print()
            results["explanations"].append(explanation)
            results["recommendations"].append(rec)

    if not results["recommendations"]:
        p_pass("No actionable recommendations — all metrics within acceptable bounds.")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY TABLE & FINAL VERDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_summary(l1, l2, l3, l4, l5, l6):
    """Print the final summary table and overall verdict."""

    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'═' * 62}")
    print(f"  FINAL SUMMARY")
    print(f"{'═' * 62}{Style.RESET_ALL}\n")

    layers = [
        ("L1", "Fairness Metrics", l1),
        ("L2", "Intersectional", l2),
        ("L3", "Proxy Detection", l3),
        ("L4", "Distribution", l4),
        ("L5", "Drift Detection", l5),
    ]

    # Table header
    print(f"  {'Layer':<6}{'Component':<22}{'Status':<10}{'Issues Found'}")
    print(f"  {'─' * 6}{'─' * 22}{'─' * 10}{'─' * 24}")

    fails = 0
    warns = 0

    for tag, name, res in layers:
        status = res.get("status", "PASS")
        issues = res.get("issues", [])
        issue_str = "; ".join(issues[:2]) if issues else "—"
        if len(issues) > 2:
            issue_str += f" (+{len(issues)-2} more)"

        if status == "FAIL":
            color = Fore.RED
            fails += 1
        elif status == "WARN":
            color = Fore.YELLOW
            warns += 1
        elif status == "SKIP":
            color = Fore.CYAN
        else:
            color = Fore.GREEN

        print(
            f"  {tag:<6}{name:<22}"
            f"{color}{status:<10}{Style.RESET_ALL}"
            f"{issue_str}"
        )

    # Layer 6 row
    n_recs = len(l6.get("recommendations", []))
    print(
        f"  {'L6':<6}{'Recommendations':<22}"
        f"{Fore.CYAN}{'—':<10}{Style.RESET_ALL}"
        f"{n_recs} recommendation(s) generated"
    )

    # ── Overall verdict ──────────────────────────────────────────────
    print(f"\n  {'─' * 56}")

    if fails > 0:
        verdict = "BIASED"
        icon = "❌"
        color = Fore.RED
    elif warns > 0:
        verdict = "BORDERLINE"
        icon = "⚠️"
        color = Fore.YELLOW
    else:
        verdict = "UNBIASED"
        icon = "✅"
        color = Fore.GREEN

    summary_parts = []
    if fails:
        summary_parts.append(f"{fails} layer(s) failed")
    if warns:
        summary_parts.append(f"{warns} warning(s)")
    summary_str = ", ".join(summary_parts) if summary_parts else "All layers passed"

    print(
        f"\n  {Style.BRIGHT}OVERALL: {color}{icon}  {verdict}{Style.RESET_ALL}"
        f"  — {summary_str}"
    )
    print()

    return verdict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAVE ALL RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def save_results(verdict, l1, l2, l3, l4, l5, l6, output_dir):
    """Save comprehensive results to JSON."""
    out_path = os.path.join(output_dir, "fairsight_results.json")

    payload = {
        "fairsight_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "overall_verdict": verdict,
        "layer1_fairness_metrics": l1,
        "layer2_intersectional": l2,
        "layer3_proxy_detection": l3,
        "layer4_distribution": l4,
        "layer5_drift": l5,
        "layer6_recommendations": l6,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)

    p_info(f"Full results saved to: {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    init(autoreset=False)  # colorama

    parser = argparse.ArgumentParser(
        description="FairSight CLI — Terminal-based bias detection tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fairsight.py --csv data.csv\n"
            "  python fairsight.py --csv data.csv --target loan_approved\n"
            "  python fairsight.py --csv data.csv --history prev_results.json\n"
        ),
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file to analyze")
    parser.add_argument("--target", default=None, help="Target/outcome column name (default: last column)")
    parser.add_argument("--history", default=None, help="Path to previous results JSON for drift detection")
    args = parser.parse_args()

    # ── Validate input ───────────────────────────────────────────────
    if not os.path.exists(args.csv):
        print(f"{Fore.RED}Error: File '{args.csv}' not found.{Style.RESET_ALL}")
        sys.exit(1)

    output_dir = os.path.dirname(os.path.abspath(args.csv))

    # ── Load data ────────────────────────────────────────────────────
    banner()
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"{Fore.RED}Error reading CSV: {e}{Style.RESET_ALL}")
        sys.exit(1)

    # ── Dataset overview ─────────────────────────────────────────────
    print(f"  {Style.BRIGHT}Dataset:{Style.RESET_ALL} {args.csv}")
    print(f"  {Style.BRIGHT}Rows:{Style.RESET_ALL}    {len(df):,}")
    print(f"  {Style.BRIGHT}Columns:{Style.RESET_ALL} {len(df.columns)}")
    print(f"  {Style.BRIGHT}Columns:{Style.RESET_ALL} {', '.join(df.columns)}")

    # Handle missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        p_warn(f"Missing values in: {', '.join(missing_cols)}")
        p_info("Missing numeric values will be imputed with median where needed.")

    # ── Detect protected attributes ──────────────────────────────────
    protected_attrs = detect_protected_attributes(df)
    if not protected_attrs:
        print(f"\n{Fore.RED}  No protected attributes detected in column names.{Style.RESET_ALL}")
        print(f"  Looked for keywords: {', '.join(PROTECTED_KEYWORDS)}")
        print(f"  Your columns: {', '.join(df.columns)}")
        sys.exit(1)
    print(f"\n  {Style.BRIGHT}Protected attributes detected:{Style.RESET_ALL} {', '.join(protected_attrs)}")

    # ── Detect target column (before binning, so auto-detect finds original last column) ──
    target = detect_target_column(df, args.target)
    print(f"  {Style.BRIGHT}Target column:{Style.RESET_ALL} {target}")

    # ── Bin numeric protected attributes (e.g. age → age_group) ──────
    df, protected_attrs, binning_info = bin_numeric_protected(df, protected_attrs)

    # Ensure target is binary-like
    unique_targets = df[target].nunique()
    if unique_targets > 2:
        if pd.api.types.is_numeric_dtype(df[target]):
            p_warn(
                f"Target '{target}' has {unique_targets} unique values. "
                f"Binarizing at median for fairness metrics."
            )
            median_val = df[target].median()
            df[target] = (df[target] >= median_val).astype(int)
        else:
            p_warn(
                f"Target '{target}' has {unique_targets} unique string values. "
                f"Selecting the most frequent as positive class."
            )
            top_val = df[target].mode()[0]
            df[target] = (df[target] == top_val).astype(int)
    elif unique_targets == 2:
        # Ensure 0/1 encoding
        vals = sorted(df[target].unique())
        df[target] = df[target].map({vals[0]: 0, vals[1]: 1})
    elif unique_targets == 1:
        p_warn(f"Target '{target}' has only 1 unique value.")
        df[target] = 1

    overall_rate = df[target].mean()
    print(f"  {Style.BRIGHT}Overall positive rate:{Style.RESET_ALL} {overall_rate:.1%}")

    # Track which original columns were binned so we exclude them from proxy/distribution analysis
    binned_originals = set(binning_info.keys())

    # ── Run all layers ───────────────────────────────────────────────
    l1_results = layer1(df, protected_attrs, target)
    l2_results = layer2(df, protected_attrs, target)
    l3_results = layer3(df, protected_attrs, target, binned_originals)
    l4_results = layer4(df, protected_attrs, target, binned_originals)
    l5_results = layer5(l1_results, args.history, output_dir)
    l6_results = layer6(l1_results, l2_results, l3_results, l4_results, l5_results)

    # ── Final summary ────────────────────────────────────────────────
    verdict = print_summary(l1_results, l2_results, l3_results, l4_results, l5_results, l6_results)

    # ── Save results ─────────────────────────────────────────────────
    save_results(verdict, l1_results, l2_results, l3_results, l4_results, l5_results, l6_results, output_dir)

    print(f"{Fore.CYAN}{'━' * 62}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
