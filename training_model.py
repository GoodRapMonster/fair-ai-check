#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║  Quick Model Trainer — Train & export a model from any CSV       ║
║  Companion to ModelSight (model bias detection)                  ║
╚═══════════════════════════════════════════════════════════════════╝

Usage:
    python training_model.py --csv data.csv
    python training_model.py --csv data.csv --target loan_approved
    python training_model.py --csv data.csv --target income --split 0.3
    python training_model.py --csv data.csv --target income --model-name my_model

Outputs:
    <model-name>.pkl        — trained GradientBoostingClassifier
    <model-name>_test.csv   — held-out test split (for ModelSight)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from colorama import Fore, Style, init

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROTECTED_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "age",
    "religion", "nationality", "disability", "marital",
]


def p_info(t):
    print(f"  {Fore.CYAN}ℹ{Style.RESET_ALL}  {t}")


def p_pass(t):
    print(f"  {Fore.GREEN}✓{Style.RESET_ALL}  {t}")


def p_warn(t):
    print(f"  {Fore.YELLOW}⚠{Style.RESET_ALL}  {t}")


def p_fail(t):
    print(f"  {Fore.RED}✗{Style.RESET_ALL}  {t}")


def detect_target(df, arg):
    """Auto-detect or validate the target column."""
    if arg:
        if arg in df.columns:
            return arg
        print(f"{Fore.RED}Error: Target '{arg}' not found.{Style.RESET_ALL}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Heuristics: look for common target-like column names
    target_hints = [
        "target", "label", "class", "outcome", "result",
        "approved", "default", "fraud", "churn", "survived",
        "diagnosis", "income", "salary", "hired", "admitted",
    ]
    for col in df.columns:
        if any(h in col.lower() for h in target_hints):
            return col

    # Fall back to last column
    return df.columns[-1]


def binarise_target(df, target):
    """Convert target to binary 0/1."""
    col = df[target]
    nu = col.nunique()

    if nu <= 1:
        p_fail(f"Target '{target}' has only 1 unique value — cannot train a classifier.")
        sys.exit(1)

    if nu == 2:
        vals = col.dropna().unique().tolist()
        # Try to pick the "positive" class intelligently
        pos = next(
            (v for v in vals if ">" in str(v) or str(v).strip().lower() in ("1", "yes", "true", "high")),
            sorted(vals, key=str)[-1],
        )
        df[target] = (col == pos).astype(int)
        p_info(f"Binary target: positive class = '{pos}'")
        return df, str(pos)

    if pd.api.types.is_numeric_dtype(col):
        med = col.median()
        df[target] = (col >= med).astype(int)
        p_warn(f"Numeric target with {nu} values — binarised at median ({med})")
        return df, f">={med}"

    top = col.mode()[0]
    df[target] = (col == top).astype(int)
    p_warn(f"Categorical target with {nu} values — using '{top}' as positive class")
    return df, str(top)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    init(autoreset=False)

    if not HAS_SKLEARN:
        print(f"{Fore.RED}Error: scikit-learn is required. Run: pip install scikit-learn joblib{Style.RESET_ALL}")
        sys.exit(1)

    ap = argparse.ArgumentParser(
        description="Quick Model Trainer — train a classifier from any CSV and export .pkl"
    )
    ap.add_argument("--csv", required=True, help="Path to CSV dataset")
    ap.add_argument("--target", default=None, help="Target column name (auto-detected if omitted)")
    ap.add_argument("--split", type=float, default=0.3, help="Test split ratio (default: 0.3)")
    ap.add_argument("--model-name", default=None, help="Output model name (default: derived from CSV name)")
    args = ap.parse_args()

    # ── Validate ─────────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        print(f"{Fore.RED}Error: '{args.csv}' not found.{Style.RESET_ALL}")
        sys.exit(1)

    if not 0.05 <= args.split <= 0.9:
        print(f"{Fore.RED}Error: --split must be between 0.05 and 0.9{Style.RESET_ALL}")
        sys.exit(1)

    # ── Banner ───────────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{Style.BRIGHT}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   🏋️  Quick Model Trainer                            ║")
    print("  ║      Train → Export → Audit with ModelSight         ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)

    # ── Load CSV ─────────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"{Fore.RED}CSV read error: {e}{Style.RESET_ALL}")
        sys.exit(1)

    print(f"  {Style.BRIGHT}Dataset:{Style.RESET_ALL}    {args.csv}")
    print(f"  {Style.BRIGHT}Rows:{Style.RESET_ALL}       {len(df):,}")
    print(f"  {Style.BRIGHT}Columns:{Style.RESET_ALL}    {len(df.columns)}")
    print(f"  {Style.BRIGHT}Columns:{Style.RESET_ALL}    {', '.join(df.columns)}")

    if len(df) < 10:
        p_fail("Dataset too small (< 10 rows). Need more data to train a model.")
        sys.exit(1)

    # ── Handle missing values ────────────────────────────────────────
    miss = df.columns[df.isnull().any()].tolist()
    if miss:
        p_warn(f"Missing values in: {', '.join(miss)}")
        # Fill numeric with median, categorical with mode
        for c in df.select_dtypes(include=[np.number]).columns:
            df[c] = df[c].fillna(df[c].median())
        for c in df.select_dtypes(include=["object", "category"]).columns:
            df[c] = df[c].fillna(df[c].mode()[0] if len(df[c].mode()) > 0 else "unknown")
        p_info("Imputed missing values (median for numeric, mode for categorical)")

    # ── Detect target ────────────────────────────────────────────────
    target = detect_target(df, args.target)
    print(f"\n  {Style.BRIGHT}Target column:{Style.RESET_ALL}  {target}")
    print(f"  {Style.BRIGHT}Unique values:{Style.RESET_ALL}  {df[target].nunique()}")

    # ── Binarise ─────────────────────────────────────────────────────
    df, pos_label = binarise_target(df, target)
    pos_rate = df[target].mean()
    print(f"  {Style.BRIGHT}Positive rate:{Style.RESET_ALL}   {pos_rate:.1%}")

    if pos_rate < 0.02 or pos_rate > 0.98:
        p_warn(f"Severely imbalanced target ({pos_rate:.1%} positive). Model may be unreliable.")

    # ── Detect protected attributes ──────────────────────────────────
    prot_attrs = [c for c in df.columns if any(k in c.lower() for k in PROTECTED_KEYWORDS) and c != target]
    if prot_attrs:
        p_info(f"Protected attributes detected: {', '.join(prot_attrs)}")
        p_warn("These will be INCLUDED as features (to test for model bias with ModelSight)")
    else:
        p_info("No protected attributes detected in column names.")

    # ── Prepare features ─────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].copy()
    y = df[target].values

    # Encode categoricals
    encoders = {}
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le

    if cat_cols:
        p_info(f"Encoded {len(cat_cols)} categorical column(s): {', '.join(cat_cols)}")

    # ── Train/test split ─────────────────────────────────────────────
    print(f"\n  {Style.BRIGHT}Train/Test split:{Style.RESET_ALL}  {1 - args.split:.0%} / {args.split:.0%}")

    try:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=args.split, random_state=42, stratify=y
        )
    except ValueError:
        # Stratify may fail if a class has too few samples
        p_warn("Stratified split failed — using random split instead")
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=args.split, random_state=42
        )

    print(f"  {Style.BRIGHT}Train size:{Style.RESET_ALL}      {len(X_train):,} rows")
    print(f"  {Style.BRIGHT}Test size:{Style.RESET_ALL}       {len(X_test):,} rows")

    # ── Train model ──────────────────────────────────────────────────
    print(f"\n  {Style.BRIGHT}Training GradientBoostingClassifier...{Style.RESET_ALL}")

    # Scale n_estimators based on dataset size
    n_est = min(200, max(50, len(X_train) // 5))
    max_depth = 3 if len(X_train) < 500 else 4

    model = GradientBoostingClassifier(
        n_estimators=n_est,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    train_pred_rate = model.predict(X_train).mean()
    test_pred_rate = model.predict(X_test).mean()

    print(f"\n  {Style.BRIGHT}Results:{Style.RESET_ALL}")
    print(f"    Train accuracy:      {Fore.GREEN}{train_acc:.1%}{Style.RESET_ALL}")
    print(f"    Test accuracy:       {Fore.GREEN}{test_acc:.1%}{Style.RESET_ALL}")
    print(f"    Train pred rate:     {train_pred_rate:.1%}")
    print(f"    Test pred rate:      {test_pred_rate:.1%}")

    if train_acc - test_acc > 0.15:
        p_warn(f"Model may be overfitting (train={train_acc:.1%} vs test={test_acc:.1%})")

    # Feature importance
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  {Style.BRIGHT}Top features by importance:{Style.RESET_ALL}")
    for i, (f, imp) in enumerate(importances[:8], 1):
        bar = "█" * int(imp / importances[0][1] * 15)
        is_protected = any(k in f.lower() for k in PROTECTED_KEYWORDS)
        flag = f"  {Fore.RED}← PROTECTED{Style.RESET_ALL}" if is_protected else ""
        print(f"    {i:2d}. {f:25s}  {imp:.4f}  {Fore.CYAN}{bar}{Style.RESET_ALL}{flag}")

    # ── Save model ───────────────────────────────────────────────────
    base_name = args.model_name or os.path.splitext(os.path.basename(args.csv))[0] + "_model"
    output_dir = os.path.dirname(os.path.abspath(args.csv))

    model_path = os.path.join(output_dir, f"{base_name}.pkl")
    test_path = os.path.join(output_dir, f"{base_name}_test.csv")

    # Save model
    joblib.dump(model, model_path)
    p_pass(f"Model saved → {model_path}")

    # Save test set with ORIGINAL (non-encoded) values for ModelSight
    test_df = df.iloc[idx_test].copy()
    test_df.to_csv(test_path, index=False)
    p_pass(f"Test set saved → {test_path} ({len(test_df)} rows)")

    # ── Print next steps ─────────────────────────────────────────────
    print(f"\n{'━' * 62}")
    print(f"  {Style.BRIGHT}Next steps — audit this model for bias:{Style.RESET_ALL}\n")
    print(f"  {Fore.CYAN}python modelsight.py --model {model_path} \\")
    print(f"                     --csv {test_path} \\")
    print(f"                     --target {target}{Style.RESET_ALL}\n")

    # Full pipeline suggestion
    print(f"  {Style.BRIGHT}Full pipeline:{Style.RESET_ALL}")
    print(f"    1. {Fore.GREEN}python fairsight_v2.py --csv {args.csv} --target {target}{Style.RESET_ALL}  ← check data bias")
    print(f"    2. {Fore.GREEN}python training_model.py --csv {args.csv} --target {target}{Style.RESET_ALL}  ← train model")
    print(f"    3. {Fore.GREEN}python modelsight.py --model {model_path} --csv {test_path} --target {target}{Style.RESET_ALL}  ← check model bias")
    print(f"{'━' * 62}\n")


if __name__ == "__main__":
    main()
