#!/usr/bin/env python3
"""Generate sample_data.csv with deliberately injected bias for testing FairSight CLI."""

import numpy as np
import pandas as pd

def generate():
    np.random.seed(42)
    n = 500

    # ── Race distribution ─────────────────────────────────────────────
    race = np.random.choice(
        ['White', 'Black', 'Hispanic', 'Asian'],
        size=n, p=[0.50, 0.25, 0.15, 0.10]
    )

    # ── Gender ────────────────────────────────────────────────────────
    gender = np.random.choice(['Male', 'Female'], size=n, p=[0.55, 0.45])

    # ── Age ───────────────────────────────────────────────────────────
    age = np.random.normal(35, 10, n).astype(int).clip(18, 70)

    # ── Zip code — deliberately correlated with race (proxy) ─────────
    zip_map = {
        'White':    [10001, 10002, 10003, 10004, 10005],
        'Black':    [20001, 20002, 20003, 20004, 20005],
        'Hispanic': [30001, 30002, 30003, 10001, 10002],
        'Asian':    [40001, 40002, 40003, 10001, 10003],
    }
    zip_code = np.array([np.random.choice(zip_map[r]) for r in race])

    # ── Years experience — slightly biased recording for women ───────
    years_experience = np.random.normal(10, 5, n).clip(0, 40).round(1)
    female_mask = gender == 'Female'
    years_experience[female_mask] = (years_experience[female_mask] - 2.5).clip(0, 40)

    # ── Income — biased by race ──────────────────────────────────────
    income = np.random.normal(60000, 20000, n).clip(20000, 150000)
    income[race == 'Black'] -= 12000
    income[race == 'Hispanic'] -= 7000
    income = income.clip(20000, 150000).round(0).astype(int)

    # ── Credit score ─────────────────────────────────────────────────
    credit_score = np.random.normal(700, 50, n).clip(300, 850).astype(int)

    # ── Loan approved — deliberately biased outcome ──────────────────
    # Black: ~40%, White: ~75%, Hispanic: ~55%, Asian: ~65%
    # Additional gender penalty
    loan_approved = np.zeros(n, dtype=int)
    base_rates = {'White': 0.75, 'Black': 0.40, 'Hispanic': 0.55, 'Asian': 0.65}

    for i in range(n):
        prob = base_rates[race[i]]
        if gender[i] == 'Female':
            prob -= 0.05
        # Slight credit score influence
        prob += (credit_score[i] - 700) * 0.0005
        prob = np.clip(prob, 0.05, 0.95)
        loan_approved[i] = 1 if np.random.random() < prob else 0

    # ── Build DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'race': race,
        'zip_code': zip_code,
        'years_experience': years_experience,
        'income': income,
        'credit_score': credit_score,
        'loan_approved': loan_approved,
    })

    out_path = 'sample_data.csv'
    df.to_csv(out_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"✓ Generated {len(df)} rows → {out_path}\n")
    print("Loan approval rates by race:")
    for r in ['White', 'Black', 'Hispanic', 'Asian']:
        rate = df.loc[df['race'] == r, 'loan_approved'].mean()
        print(f"  {r:10s}  {rate:.1%}")
    print("\nLoan approval rates by gender:")
    for g in ['Male', 'Female']:
        rate = df.loc[df['gender'] == g, 'loan_approved'].mean()
        print(f"  {g:10s}  {rate:.1%}")
    print(f"\nPreview:\n{df.head(10).to_string(index=False)}")


if __name__ == '__main__':
    generate()
