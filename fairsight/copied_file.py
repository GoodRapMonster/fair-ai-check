#!/usr/bin/env python3
"""
FairSight CLI v2.0 — 6-Layer Bias Detection — Robust for any CSV
Patches: FIX-01 to FIX-12 applied over v1
"""
import argparse, json, math, os, sys, warnings
from datetime import datetime
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pointbiserialr
from colorama import Fore, Style, init

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

PROTECTED_KEYWORDS       = ["gender","sex","race","ethnicity","age","religion","nationality","disability","marital"]
DI_THRESHOLD             = 0.8
DP_THRESHOLD             = 0.1
EO_THRESHOLD             = 0.1
PP_THRESHOLD             = 0.1
INTERSECTIONAL_DEVIATION = 0.15
HIGH_PROXY_CORR          = 0.7
MEDIUM_PROXY_CORR        = 0.5
DISTRIBUTION_SD_GAP      = 1.0
DRIFT_THRESHOLD          = 0.05

# FIX-03: scale min subgroup size to dataset size
def dynamic_min_size(n):
    if n <= 20:   return 2
    if n <= 50:   return 5
    if n <= 200:  return 10
    if n <= 1000: return 20
    return 30

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return None if math.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)

def banner():
    print(f"\n{Fore.CYAN}{Style.BRIGHT}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   ⚖  FairSight CLI v2.0 — Bias Detection Tool       ║")
    print("  ║      Comprehensive 6-Layer Fairness Audit            ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)

def shdr(n, t): print(f"\n{Fore.CYAN}{Style.BRIGHT}{'━'*62}\n  LAYER {n} — {t}\n{'━'*62}{Style.RESET_ALL}\n")
def p_pass(t): print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}  {t}")
def p_fail(t): print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL}  {t}")
def p_warn(t): print(f"  {Fore.YELLOW}⚠ WARN{Style.RESET_ALL}  {t}")
def p_info(t): print(f"  {Fore.CYAN}ℹ INFO{Style.RESET_ALL}  {t}")
def p_skip(t): print(f"  {Fore.CYAN}⊘ SKIP{Style.RESET_ALL}  {t}")
def p_rec(t):  print(f"  {Fore.MAGENTA}→ REC {Style.RESET_ALL}  {t}")

def detect_protected(df):
    return [c for c in df.columns if any(k in c.lower() for k in PROTECTED_KEYWORDS)]

def bin_numeric_protected(df, attrs):
    df = df.copy(); new_attrs = []; info = {}
    for attr in attrs:
        col = df[attr]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > 10:
            if "age" in attr.lower():
                df[attr+"_group"] = pd.cut(col,[0,25,35,45,55,65,120],
                    labels=["18-25","26-35","36-45","46-55","56-65","65+"],right=True).astype(str)
            else:
                try:    df[attr+"_group"] = pd.qcut(col,q=4,duplicates="drop").astype(str)
                except: df[attr+"_group"] = col.astype(str)
            new_attrs.append(attr+"_group"); info[attr] = True
            p_info(f"Binned numeric '{attr}' → '{attr}_group'")
        else:
            new_attrs.append(attr)
    return df, new_attrs, info

def detect_target(df, arg):
    if arg:
        if arg in df.columns: return arg
        print(f"{Fore.RED}Target '{arg}' not found. Columns: {list(df.columns)}{Style.RESET_ALL}"); sys.exit(1)
    return df.columns[-1]

def binarise_target(df, target):
    """FIX-02 FIX-04 FIX-12: safe binarisation of any target type."""
    df = df.copy(); col = df[target]; nu = col.nunique()
    if nu <= 1:
        p_warn(f"Target '{target}' has only 1 unique value — model metrics will be skipped.")
        df[target] = 1; return df, "1", True
    if nu == 2:
        vals = col.dropna().unique().tolist()
        pos = next((v for v in vals if ">" in str(v) or str(v).strip().lower() in ("1","yes","true","high")),
                   sorted(vals, key=str)[-1])
        df[target] = (col == pos).astype(int); return df, str(pos), False
    if pd.api.types.is_numeric_dtype(col):
        med = col.median()
        p_warn(f"Multi-value numeric target — binarising at median ({med}).")
        df[target] = (col >= med).astype(int); return df, f">={med}", False
    top = col.mode()[0]
    p_warn(f"Multi-value string target — using '{top}' as positive class.")
    df[target] = (col == top).astype(int); return df, str(top), False

def safe_di(u, p):
    """FIX-05: NaN/Inf-safe disparate impact."""
    if p is None or math.isnan(p) or p == 0 or u is None or math.isnan(u): return None
    return u / p

def writable_dir(path):
    """FIX-11: fallback to cwd if path not writable."""
    try:
        os.makedirs(path, exist_ok=True)
        t = os.path.join(path, ".fs_test"); open(t,"w").close(); os.remove(t)
        return path
    except: return os.getcwd()

def encode_col(series):
    """FIX-09: safe numeric encoding; None if zero-variance or uncompatible."""
    s = series.dropna()
    if s.nunique() < 2: return None, None
    if pd.api.types.is_numeric_dtype(s): return s.values, s.index
    if s.nunique() <= 10:
        codes = s.astype("category").cat.codes
        return codes.values.astype(float), codes.index
    return None, None

# ══════════════════════════════════════════════════════════════════
# LAYER 1
# ══════════════════════════════════════════════════════════════════
def layer1(df, protected_attrs, target, trivial, min_size):
    shdr(1, "Standard Fairness Metrics")
    res = {"status":"PASS","metrics":{},"issues":[]}

    # FIX-06: train model safely
    predictions = None
    if HAS_SKLEARN and not trivial:
        try:
            fcols = [c for c in df.columns if c not in protected_attrs and c != target]
            X = df[fcols].copy()
            y = df[target].values
            for c in X.select_dtypes(include=["object","category"]).columns:
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
            X = X.fillna(X.median(numeric_only=True))
            n_cls = len(np.unique(y)); n = len(y)
            cv = min(5, n//2) if n >= 10 else 0
            if n_cls >= 2 and cv >= 2:
                clf = GradientBoostingClassifier(n_estimators=50,max_depth=3,random_state=42)
                predictions = cross_val_predict(clf, X, y, cv=cv)
                p_info(f"Trained GBM ({cv}-fold CV) for EOD & PPD")
            else:
                p_skip("Model training skipped — need ≥2 classes and ≥10 rows")
        except Exception as e:
            p_warn(f"Model training failed: {e}")

    actual = df[target].values

    for attr in protected_attrs:
        print(f"\n  {Style.BRIGHT}Protected: {attr}{Style.RESET_ALL}")
        groups = df[attr].dropna().unique()
        for g in groups:
            cnt = (df[attr]==g).sum()
            if cnt < min_size: p_warn(f"Group '{g}' n={cnt} (< {min_size})")

        # FIX-01: require ≥2 valid groups
        valid = [g for g in groups if (df[attr]==g).sum() >= min_size]
        if len(valid) < 2:
            p_skip(f"'{attr}' has <2 groups with ≥{min_size} samples — not enough data to measure bias")
            res["metrics"][attr] = {"skipped":True,"reason":"insufficient_groups"}
            continue

        sub = df[df[attr].isin(valid)]
        rates = sub.groupby(attr)[target].mean()
        priv  = rates.idxmax(); unpriv = rates.idxmin()
        pr    = float(rates[priv]); ur = float(rates[unpriv])
        pm    = df[attr]==priv;     um = df[attr]==unpriv
        am    = {}

        # DI
        di = safe_di(ur, pr)
        if di is None:
            p_skip(f"DI: cannot compute (priv_rate={pr:.3f})")
            am["disparate_impact"] = {"value":None,"pass":None,"reason":"degenerate"}
        else:
            ok = di >= DI_THRESHOLD
            am["disparate_impact"] = {"value":round(di,4),"threshold":DI_THRESHOLD,"pass":ok,
                "privileged":str(priv),"priv_rate":round(pr,4),"unpriv_rate":round(ur,4)}
            (p_pass if ok else p_fail)(
                f"Disparate Impact: {di:.4f}  ({'≥' if ok else '<'} {DI_THRESHOLD})  "
                f"[priv={priv} {pr:.1%} | unpriv={unpriv} {ur:.1%}]")
            if not ok: res["status"]="FAIL"; res["issues"].append(f"DI={di:.2f} on '{attr}'")

        # DP
        dp = ur - pr; ok = abs(dp) <= DP_THRESHOLD
        am["demographic_parity"] = {"value":round(dp,4),"threshold":DP_THRESHOLD,"pass":ok}
        (p_pass if ok else p_fail)(
            f"Demographic Parity Gap: {dp:+.4f}  ({'|gap| ≤' if ok else '|gap| >'} {DP_THRESHOLD})")
        if not ok: res["status"]="FAIL"; res["issues"].append(f"DP={dp:+.2f} on '{attr}'")

        # EO
        if predictions is not None:
            try:
                pp = actual[pm.values]==1; up = actual[um.values]==1
                pt = float(predictions[pm.values][pp].mean()) if pp.sum()>0 else 0.0
                ut = float(predictions[um.values][up].mean()) if up.sum()>0 else 0.0
                eod = ut-pt; ok = abs(eod)<=EO_THRESHOLD
                am["equal_opportunity"] = {"value":round(eod,4),"threshold":EO_THRESHOLD,"pass":ok,
                    "priv_tpr":round(pt,4),"unpriv_tpr":round(ut,4)}
                (p_pass if ok else p_fail)(
                    f"Equal Opportunity Diff: {eod:+.4f}  ({'|diff| ≤' if ok else '|diff| >'} {EO_THRESHOLD})")
                if not ok: res["status"]="FAIL"; res["issues"].append(f"EOD={eod:+.2f} on '{attr}'")
            except Exception as e:
                p_warn(f"EOD failed: {e}"); am["equal_opportunity"]={"value":None,"pass":None}
        else:
            p_skip("EOD: no model predictions"); am["equal_opportunity"]={"value":None,"pass":None}

        # PP
        if predictions is not None:
            try:
                pp2 = predictions[pm.values]==1; up2 = predictions[um.values]==1
                pv = float(actual[pm.values][pp2].mean()) if pp2.sum()>0 else 0.0
                uv = float(actual[um.values][up2].mean()) if up2.sum()>0 else 0.0
                ppd = uv-pv; ok = abs(ppd)<=PP_THRESHOLD
                am["predictive_parity"] = {"value":round(ppd,4),"threshold":PP_THRESHOLD,"pass":ok,
                    "priv_ppv":round(pv,4),"unpriv_ppv":round(uv,4)}
                (p_pass if ok else p_fail)(
                    f"Predictive Parity Diff: {ppd:+.4f}  ({'|diff| ≤' if ok else '|diff| >'} {PP_THRESHOLD})")
                if not ok: res["status"]="FAIL"; res["issues"].append(f"PPD={ppd:+.2f} on '{attr}'")
            except Exception as e:
                p_warn(f"PPD failed: {e}"); am["predictive_parity"]={"value":None,"pass":None}
        else:
            p_skip("PPD: no model predictions"); am["predictive_parity"]={"value":None,"pass":None}

        res["metrics"][attr] = am
    return res

# ══════════════════════════════════════════════════════════════════
# LAYER 2
# ══════════════════════════════════════════════════════════════════
def layer2(df, protected_attrs, target, min_size):
    shdr(2, "Intersectional Analysis")
    res = {"status":"PASS","subgroups":[],"issues":[],"worst":None}
    if len(protected_attrs) < 2:
        p_skip("Need ≥2 protected attributes for intersectional analysis."); res["status"]="SKIP"; return res

    overall = df[target].mean(); p_info(f"Overall positive rate: {overall:.1%}")
    worst_dev = 0.0; worst_grp = None; flagged = []

    for combo in combinations(protected_attrs, 2):
        lbl = " × ".join(combo); p_info(f"Analyzing: {lbl}")
        small = 0
        for name, grp in df.groupby(list(combo))[target]:
            if isinstance(name, str): name = (name,)
            sub = "+".join(str(v) for v in name)
            n = len(grp); rate = float(grp.mean()); dev = abs(rate - float(overall))
            entry = {"intersection":lbl,"subgroup":sub,"n":int(n),
                     "rate":round(rate,4),"overall_rate":round(float(overall),4),"deviation":round(dev,4)}
            if n < min_size:
                entry["small_sample"] = True; small += 1; res["subgroups"].append(entry)
                if dev > worst_dev: worst_dev = dev; worst_grp = {**entry,"small_sample":True}
                continue
            if dev > INTERSECTIONAL_DEVIATION:
                flagged.append((dev, sub, rate, "below" if rate < overall else "above", n))
                res["status"] = "FAIL"; entry["flagged"] = True
                res["issues"].append(f"{sub}: {rate:.0%} vs {overall:.0%}")
            else:
                p_pass(f"  {sub}: {rate:.1%}  (n={n})")
            if dev > worst_dev: worst_dev = dev; worst_grp = entry
            res["subgroups"].append(entry)
        if small: p_warn(f"  {small} subgroup(s) n<{min_size} skipped")

    # FIX-08: if no non-small subgroups were flagged, revert to PASS
    if res["status"] == "FAIL" and not flagged: res["status"] = "PASS"

    if flagged:
        flagged.sort(reverse=True)
        print(f"\n  {Style.BRIGHT}Flagged subgroups (severity order):{Style.RESET_ALL}")
        for dev, sub, rate, dir_, n in flagged[:10]:
            p_fail(f"  {sub}: {rate:.1%} vs {overall:.1%} ({dir_} by {dev:.1%}, n={n})")
        if len(flagged) > 10: p_info(f"  … +{len(flagged)-10} more")

    if worst_grp:
        res["worst"] = worst_grp
        note = " ⚠ small sample" if worst_grp.get("small_sample") else ""
        print(f"\n  {Style.BRIGHT}Worst group:{Style.RESET_ALL} {worst_grp['subgroup']}  "
              f"(rate={worst_grp['rate']:.1%}, dev={worst_grp['deviation']:.1%}, n={worst_grp['n']}){note}")
    return res

# ══════════════════════════════════════════════════════════════════
# LAYER 3
# ══════════════════════════════════════════════════════════════════
def layer3(df, protected_attrs, target, binned_orig=None, trivial=False, min_size=5):
    shdr(3, "Proxy Variable Detection")
    binned_orig = binned_orig or set()
    res = {"status":"PASS","proxies":[],"shap_top":[],"issues":[]}
    non_prot = [c for c in df.columns if c not in protected_attrs and c != target and c not in binned_orig]
    all_px = []

    for attr in protected_attrs:
        ae, ai = encode_col(df[attr])
        if ae is None:
            for val in df[attr].dropna().unique():
                binary = (df[attr]==val).astype(float).values
                for col in non_prot:
                    ce, _ = encode_col(df[col])
                    if ce is None or len(ce) != len(df): continue
                    try:
                        r, p = pointbiserialr(binary, ce)
                        if math.isnan(r): continue
                    except: continue
                    risk = ("HIGH" if abs(r)>HIGH_PROXY_CORR and p<0.05 else
                            "MEDIUM" if abs(r)>MEDIUM_PROXY_CORR and p<0.05 else None)
                    if risk: all_px.append({"feature":col,"protected_attr":f"{attr}={val}",
                        "correlation":round(r,4),"p_value":round(p,6),"risk":risk})
            continue
        for col in non_prot:
            ce, ci = encode_col(df[col])
            if ce is None: continue
            common = ai.intersection(ci)
            if len(common) < max(min_size, 3): continue
            ae2, _ = encode_col(df.loc[common, attr])
            ce2, _ = encode_col(df.loc[common, col])
            if ae2 is None or ce2 is None: continue
            try:
                r, p = pointbiserialr(ae2, ce2)
                if math.isnan(r): continue
            except:
                try: r, p = stats.pearsonr(ae2.astype(float), ce2.astype(float))
                except: continue
                if math.isnan(r): continue
            risk = ("HIGH" if abs(r)>HIGH_PROXY_CORR and p<0.05 else
                    "MEDIUM" if abs(r)>MEDIUM_PROXY_CORR and p<0.05 else None)
            if risk: all_px.append({"feature":col,"protected_attr":attr,
                "correlation":round(r,4),"p_value":round(p,6),"risk":risk})

    seen = {}
    for px in all_px:
        k = (px["feature"], px["protected_attr"])
        if k not in seen or abs(px["correlation"]) > abs(seen[k]["correlation"]): seen[k] = px
    all_px = sorted(seen.values(), key=lambda x: abs(x["correlation"]), reverse=True)

    for px in all_px[:5]:
        if px["risk"] == "HIGH":
            p_fail(f"{px['feature']} HIGH RISK proxy for {px['protected_attr']}  (r={px['correlation']:.4f})")
            if res["status"] == "PASS": res["status"] = "WARN"
            res["issues"].append(f"{px['feature']} HIGH proxy for {px['protected_attr']}")
        else:
            p_warn(f"{px['feature']} MEDIUM RISK proxy for {px['protected_attr']}  (r={px['correlation']:.4f})")
            if res["status"] == "PASS": res["status"] = "WARN"
            res["issues"].append(f"{px['feature']} MEDIUM proxy for {px['protected_attr']}")

    if not all_px: p_pass("No significant proxy variables detected.")
    res["proxies"] = all_px

    # FIX-07: SHAP only when safe
    if HAS_SKLEARN and HAS_SHAP and not trivial:
        try:
            fcols = [c for c in df.columns if c not in protected_attrs and c != target and c not in binned_orig]
            X = df[fcols].copy(); y = df[target].values
            if len(np.unique(y)) >= 2 and len(y) >= 10:
                for c in X.select_dtypes(include=["object","category"]).columns:
                    X[c] = LabelEncoder().fit_transform(X[c].astype(str))
                X = X.fillna(X.median(numeric_only=True))
                clf = GradientBoostingClassifier(n_estimators=50,max_depth=3,random_state=42)
                clf.fit(X, y)
                expl = shap.TreeExplainer(clf)
                sv = expl.shap_values(X)
                mshap = np.abs(sv).mean(axis=0)
                ranking = sorted(zip(fcols, mshap), key=lambda x: x[1], reverse=True)
                res["shap_top"] = [{"feature":f,"mean_abs_shap":round(float(s),4)} for f,s in ranking[:5]]
                print(); p_info("Top 5 features by SHAP importance:")
                for i,(f,v) in enumerate(ranking[:5],1):
                    flag = f" {Fore.RED}← PROXY{Style.RESET_ALL}" if any(px["feature"]==f for px in all_px) else ""
                    print(f"    {i}. {f:25s}  SHAP={v:.4f}{flag}")
            else:
                p_skip("SHAP: need ≥2 target classes and ≥10 rows")
        except Exception as e:
            p_warn(f"SHAP failed: {e}")
    elif not HAS_SHAP:
        p_skip("SHAP not installed — pip install shap")
    return res

# ══════════════════════════════════════════════════════════════════
# LAYER 4
# ══════════════════════════════════════════════════════════════════
def layer4(df, protected_attrs, target, binned_orig=None, min_size=5):
    shdr(4, "Feature Distribution Analysis")
    binned_orig = binned_orig or set()
    res = {"status":"PASS","distributions":[],"issues":[]}
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in protected_attrs and c != target and c not in binned_orig]
    if not num_cols: p_skip("No numeric non-protected columns."); return res

    for attr in protected_attrs:
        groups = df[attr].dropna().unique()
        valid  = [g for g in groups if (df[attr]==g).sum() >= min_size]
        if len(valid) < 2: p_skip(f"'{attr}': <2 groups with ≥{min_size} samples"); continue
        print(f"\n  {Style.BRIGHT}Distribution gaps across: {attr}{Style.RESET_ALL}")
        for col in num_cols:
            # FIX-10: skip zero-variance columns
            if df[col].std() == 0: p_skip(f"  '{col}': zero variance"); continue
            gstats = {}
            for g in valid:
                vals = df.loc[df[attr]==g, col].dropna()
                if len(vals) == 0: continue
                gstats[str(g)] = {"mean":round(float(vals.mean()),2),"std":round(float(vals.std()),2),"n":int(len(vals))}
            if len(gstats) < 2: continue
            means = [s["mean"] for s in gstats.values()]
            stds  = [s["std"]  for s in gstats.values() if s["std"] > 0]
            if not stds: p_skip(f"  '{col}': all groups zero std"); continue
            diff = max(means)-min(means); pstd = float(np.mean(stds)); sdg = diff/pstd
            entry = {"feature":col,"protected_attr":attr,"group_stats":gstats,
                     "max_mean_diff":round(diff,2),"pooled_std":round(pstd,2),"sd_gap":round(sdg,2)}
            if sdg > DISTRIBUTION_SD_GAP:
                sg = sorted(gstats.items(), key=lambda x: x[1]["mean"])
                p_warn(f"  '{col}': {sdg:.1f} SD gap  [{sg[0][0]}={sg[0][1]['mean']} vs {sg[-1][0]}={sg[-1][1]['mean']}]")
                if res["status"] == "PASS": res["status"] = "WARN"
                entry["flagged"] = True; res["issues"].append(f"'{col}' {sdg:.1f} SD gap across {attr}")
            else:
                p_pass(f"  '{col}': {sdg:.2f} SD gap")
            res["distributions"].append(entry)
    if not res["issues"]: p_pass("No significant distribution gaps found.")
    return res

# ══════════════════════════════════════════════════════════════════
# LAYER 5
# ══════════════════════════════════════════════════════════════════
def layer5(l1, history_path, output_dir):
    shdr(5, "Fairness Drift Over Time")
    res = {"status":"PASS","trends":{},"issues":[]}
    hist_out = os.path.join(output_dir, "fairsight_history.json")
    snapshot = {"timestamp":datetime.now().isoformat(),"metrics":l1.get("metrics",{})}
    hist = []
    if os.path.exists(hist_out):
        try:
            with open(hist_out) as f: loaded = json.load(f)
            hist = loaded if isinstance(loaded, list) else [loaded]
        except: hist = []
    prev = None
    if history_path and os.path.exists(history_path):
        try:
            with open(history_path) as f: loaded = json.load(f)
            prev = loaded[-1] if isinstance(loaded, list) else loaded
            p_info(f"Loaded history: {history_path}")
        except Exception as e: p_warn(f"Could not load history: {e}")
    elif hist:
        prev = hist[-1]; p_info(f"Auto-loaded previous run from: {hist_out}")

    if prev is None:
        p_skip("No previous run found — saving snapshot for next comparison.")
        res["issues"].append("No history available")
    else:
        pm = prev.get("metrics",{}); cm = l1.get("metrics",{})
        p_info(f"Comparing against: {prev.get('timestamp','unknown')}"); print()
        for attr, m in cm.items():
            if m.get("skipped"): continue
            cdi = m.get("disparate_impact",{}).get("value")
            pdi = pm.get(attr,{}).get("disparate_impact",{}).get("value")
            if cdi is None or pdi is None: continue
            delta = cdi - pdi
            trend = ("improving ↑" if delta > 0.01 else "degrading ↓" if delta < -0.01 else "stable ─")
            res["trends"][attr] = {"attr":attr,"current_di":round(cdi,4),"previous_di":round(pdi,4),
                                    "delta":round(delta,4),"trend":trend}
            if delta < -DRIFT_THRESHOLD:
                p_fail(f"DI on '{attr}': {pdi:.4f} → {cdi:.4f}  (Δ={delta:+.4f}, {trend})")
                res["status"] = "WARN"; res["issues"].append(f"DI on '{attr}' degraded {abs(delta):.3f}")
            else:
                col = Fore.GREEN if "improving" in trend else Fore.YELLOW
                p_pass(f"DI on '{attr}': {pdi:.4f} → {cdi:.4f}  (Δ={delta:+.4f}, {col}{trend}{Style.RESET_ALL})")

    hist.append(snapshot)
    try:
        with open(hist_out,"w") as f: json.dump(hist, f, indent=2, cls=NumpyEncoder)
        p_info(f"Snapshot saved → {hist_out}")
    except Exception as e: p_warn(f"Could not save history: {e}")
    return res

# ══════════════════════════════════════════════════════════════════
# LAYER 6
# ══════════════════════════════════════════════════════════════════
def layer6(l1, l2, l3, l4, l5):
    shdr(6, "Human-Readable Verdict & Recommendations")
    res = {"recommendations":[],"explanations":[]}

    def emit(color, expl, rec):
        print(f"  {color}▸{Style.RESET_ALL} {expl}")
        p_rec(rec); print()
        res["explanations"].append(expl); res["recommendations"].append(rec)

    for attr, m in l1.get("metrics",{}).items():
        if m.get("skipped"): continue
        di = m.get("disparate_impact",{})
        if di.get("pass") is False:
            emit(Fore.RED,
                f"Disparate Impact on '{attr}' = {di['value']:.2f} (threshold {DI_THRESHOLD}).\n"
                f"    Unprivileged group gets positive outcomes at only {di['value']:.0%} the rate of "
                f"privileged group ('{di.get('privileged','?')}').",
                f"Apply reweighting or SMOTE to balance data for '{attr}'.")
        dp = m.get("demographic_parity",{})
        if dp.get("pass") is False:
            emit(Fore.RED,
                f"Demographic Parity Gap on '{attr}' = {dp['value']:+.4f}  (threshold ±{DP_THRESHOLD}).\n"
                f"    Outcome rates differ significantly between groups.",
                f"Apply threshold adjustment to equalise positive rates across '{attr}' groups.")
        eod = m.get("equal_opportunity",{})
        if eod.get("pass") is False:
            emit(Fore.RED,
                f"Equal Opportunity Diff on '{attr}' = {eod['value']:+.4f}  (threshold ±{EO_THRESHOLD}).\n"
                f"    Qualified individuals in one group are less likely to be identified correctly.",
                f"Retrain with equalized-odds constraints or post-hoc calibrate TPR for '{attr}'.")
        ppd = m.get("predictive_parity",{})
        if ppd.get("pass") is False:
            emit(Fore.RED,
                f"Predictive Parity Diff on '{attr}' = {ppd['value']:+.4f}  (threshold ±{PP_THRESHOLD}).\n"
                f"    Model precision differs between groups.",
                f"Calibrate model or adjust thresholds to equalise PPV across '{attr}' groups.")

    if l2.get("worst") and not l2["worst"].get("small_sample"):
        wg = l2["worst"]
        emit(Fore.YELLOW,
            f"Intersectional bias: '{wg['subgroup']}' rate={wg['rate']:.1%}, "
            f"deviates {wg['deviation']:.1%} from overall.\n"
            f"    Not visible when looking at individual attributes alone.",
            f"Augment data for '{wg['subgroup']}' and monitor intersectional metrics separately.")

    for px in l3.get("proxies",[])[:3]:
        col = Fore.RED if px["risk"]=="HIGH" else Fore.YELLOW
        emit(col,
            f"'{px['feature']}' is a {px['risk']} RISK proxy for '{px['protected_attr']}' "
            f"(r={px['correlation']:.4f}).\n    The model can reconstruct group membership indirectly.",
            f"{'Remove' if px['risk']=='HIGH' else 'Investigate'} '{px['feature']}' before training.")

    for d in l4.get("distributions",[]):
        if d.get("flagged"):
            emit(Fore.YELLOW,
                f"'{d['feature']}' has {d['sd_gap']:.1f} SD gap across '{d['protected_attr']}' groups.\n"
                f"    Data collection may differ systematically between groups.",
                f"Audit collection for '{d['feature']}'. Consider per-group normalisation.")

    for attr, t in l5.get("trends",{}).items():
        if "degrading" in t.get("trend",""):
            emit(Fore.RED,
                f"Fairness drift on '{attr}': DI {t['previous_di']:.4f} → {t['current_di']:.4f}.",
                f"Retrain — fairness degraded for '{attr}' since last run.")

    if not res["recommendations"]: p_pass("No actionable recommendations — all metrics within bounds.")
    return res

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
def print_summary(l1, l2, l3, l4, l5, l6):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'═'*62}\n  FINAL SUMMARY\n{'═'*62}{Style.RESET_ALL}\n")
    layers = [("L1","Fairness Metrics",l1),("L2","Intersectional",l2),
              ("L3","Proxy Detection",l3),("L4","Distribution",l4),("L5","Drift Detection",l5)]
    print(f"  {'Layer':<6}{'Component':<22}{'Status':<12}{'Issues'}")
    print(f"  {'─'*6}{'─'*22}{'─'*12}{'─'*26}")
    fails = warns = 0
    for tag, name, r in layers:
        st = r.get("status","PASS"); iss = r.get("issues",[])
        istr = ("; ".join(iss[:2]) + (f" (+{len(iss)-2} more)" if len(iss)>2 else "")) if iss else "—"
        col = (Fore.RED if st=="FAIL" else Fore.YELLOW if st=="WARN" else Fore.CYAN if st=="SKIP" else Fore.GREEN)
        if st=="FAIL": fails+=1
        if st=="WARN": warns+=1
        print(f"  {tag:<6}{name:<22}{col}{st:<12}{Style.RESET_ALL}{istr}")
    nr = len(l6.get("recommendations",[]))
    print(f"  {'L6':<6}{'Recommendations':<22}{Fore.CYAN}{'—':<12}{Style.RESET_ALL}{nr} recommendation(s)")
    print(f"\n  {'─'*58}")
    if fails>0:   verdict,icon,col = "BIASED",    "❌", Fore.RED
    elif warns>0: verdict,icon,col = "BORDERLINE","⚠ ", Fore.YELLOW
    else:         verdict,icon,col = "UNBIASED",  "✅", Fore.GREEN
    parts = []
    if fails: parts.append(f"{fails} layer(s) failed")
    if warns: parts.append(f"{warns} warning(s)")
    summ = ", ".join(parts) if parts else "All layers passed"
    print(f"\n  {Style.BRIGHT}OVERALL: {col}{icon}  {verdict}{Style.RESET_ALL}  — {summ}\n")
    return verdict

def save_results(verdict, l1, l2, l3, l4, l5, l6, output_dir):
    out = os.path.join(output_dir, "fairsight_results.json")
    try:
        with open(out,"w") as f:
            json.dump({"fairsight_version":"2.0.0","timestamp":datetime.now().isoformat(),
                "overall_verdict":verdict,"layer1":l1,"layer2":l2,"layer3":l3,
                "layer4":l4,"layer5":l5,"layer6":l6}, f, indent=2, cls=NumpyEncoder)
        p_info(f"Results saved → {out}")
    except Exception as e: p_warn(f"Could not save results: {e}")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    init(autoreset=False)
    ap = argparse.ArgumentParser(description="FairSight CLI v2 — robust bias detection for any CSV")
    ap.add_argument("--csv",     required=True)
    ap.add_argument("--target",  default=None)
    ap.add_argument("--history", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"{Fore.RED}Error: '{args.csv}' not found.{Style.RESET_ALL}"); sys.exit(1)

    output_dir = writable_dir(os.path.dirname(os.path.abspath(args.csv)))
    banner()

    try:    df = pd.read_csv(args.csv)
    except Exception as e: print(f"{Fore.RED}CSV read error: {e}{Style.RESET_ALL}"); sys.exit(1)

    min_size = dynamic_min_size(len(df))
    print(f"  {Style.BRIGHT}Dataset:{Style.RESET_ALL}      {args.csv}")
    print(f"  {Style.BRIGHT}Rows:{Style.RESET_ALL}         {len(df):,}")
    print(f"  {Style.BRIGHT}Columns:{Style.RESET_ALL}      {len(df.columns)}  ({', '.join(df.columns)})")
    print(f"  {Style.BRIGHT}Min group size:{Style.RESET_ALL} {min_size} (auto-scaled to dataset)")

    miss = df.columns[df.isnull().any()].tolist()
    if miss: p_warn(f"Missing values in: {', '.join(miss)}")

    protected_attrs = detect_protected(df)
    if not protected_attrs:
        print(f"{Fore.RED}No protected attributes found. Add columns named: {', '.join(PROTECTED_KEYWORDS)}{Style.RESET_ALL}"); sys.exit(1)
    print(f"\n  {Style.BRIGHT}Protected attributes:{Style.RESET_ALL} {', '.join(protected_attrs)}")

    target = detect_target(df, args.target)
    print(f"  {Style.BRIGHT}Target column:{Style.RESET_ALL}  {target}")

    df, protected_attrs, binned_info = bin_numeric_protected(df, protected_attrs)
    binned_orig = set(binned_info.keys())

    df, pos_label, trivial = binarise_target(df, target)
    print(f"  {Style.BRIGHT}Positive label:{Style.RESET_ALL} '{pos_label}'")
    print(f"  {Style.BRIGHT}Positive rate:{Style.RESET_ALL}  {df[target].mean():.1%}")
    if trivial: p_warn("Trivial target — only distribution & proxy layers will be fully meaningful.")

    l1 = layer1(df, protected_attrs, target, trivial, min_size)
    l2 = layer2(df, protected_attrs, target, min_size)
    l3 = layer3(df, protected_attrs, target, binned_orig, trivial, min_size)
    l4 = layer4(df, protected_attrs, target, binned_orig, min_size)
    l5 = layer5(l1, args.history, output_dir)
    l6 = layer6(l1, l2, l3, l4, l5)

    verdict = print_summary(l1, l2, l3, l4, l5, l6)
    save_results(verdict, l1, l2, l3, l4, l5, l6, output_dir)
    print(f"{Fore.CYAN}{'━'*62}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()
