import pandas as pd
import numpy as np
from typing import Optional
import io


def generate_biased_dataset(
    n_rows: int = 1000,
    bias_level: float = 0.6,
    domain: str = "hiring",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset with controllable bias for testing.
    bias_level: 0.0 = no bias, 1.0 = extreme bias
    """
    rng = np.random.RandomState(random_state)
    n = n_rows

    if domain == "hiring":
        gender = rng.choice([0, 1], size=n, p=[0.45, 0.55])  # 0=Female, 1=Male
        age = rng.normal(38, 10, n).astype(int).clip(22, 65)
        experience = rng.normal(8, 4, n).astype(int).clip(0, 30)
        education = rng.choice([0, 1, 2, 3], size=n, p=[0.1, 0.3, 0.4, 0.2])
        skills_score = rng.uniform(0.3, 1.0, n)
        interview_score = rng.uniform(0.2, 1.0, n)
        zip_code = np.where(
            gender == 0,
            rng.choice([10001, 10002, 10003], n),
            rng.choice([20001, 20002, 20003], n),
        )

        # Outcome with controlled bias
        base_prob = 0.3 + 0.3 * skills_score + 0.2 * (experience / 30) + 0.1 * (education / 3)
        gender_boost = (gender == 1) * bias_level * 0.35
        prob = np.clip(base_prob + gender_boost + rng.normal(0, 0.05, n), 0.01, 0.99)
        outcome = rng.binomial(1, prob)

        return pd.DataFrame({
            "gender": gender,
            "age": age,
            "years_experience": experience,
            "education_level": education,
            "skills_score": skills_score.round(3),
            "interview_score": interview_score.round(3),
            "zip_code": zip_code,
            "hired": outcome,
        })

    elif domain == "lending":
        race = rng.choice([0, 1, 2], size=n, p=[0.55, 0.25, 0.20])  # 0=White,1=Black,2=Hispanic
        income = rng.normal(65000, 20000, n).astype(int).clip(20000, 200000)
        credit_score = rng.normal(680, 80, n).astype(int).clip(300, 850)
        loan_amount = rng.normal(25000, 10000, n).astype(int).clip(5000, 100000)
        employment_years = rng.exponential(5, n).round(1).clip(0, 40)
        debt_ratio = rng.uniform(0.1, 0.6, n).round(3)
        zip_code = np.where(
            race == 0,
            rng.choice([60601, 60602, 60603], n),
            rng.choice([60620, 60621, 60622], n),
        )

        base_prob = (
            0.2
            + 0.3 * ((credit_score - 300) / 550)
            + 0.2 * (income / 200000)
            - 0.15 * debt_ratio
            + 0.1 * (employment_years / 40)
        )
        race_penalty = (race != 0) * bias_level * 0.25
        prob = np.clip(base_prob - race_penalty + rng.normal(0, 0.05, n), 0.01, 0.99)
        outcome = rng.binomial(1, prob)

        return pd.DataFrame({
            "race": race,
            "income": income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
            "debt_to_income_ratio": debt_ratio,
            "zip_code": zip_code,
            "loan_approved": outcome,
        })

    elif domain == "medical":
        gender = rng.choice([0, 1], size=n, p=[0.50, 0.50])  # 0=Female, 1=Male
        age = rng.normal(50, 15, n).astype(int).clip(18, 90)
        severity_score = rng.uniform(0.2, 1.0, n).round(3)
        insurance_type = rng.choice([0, 1, 2], size=n, p=[0.25, 0.45, 0.30])
        prior_visits = rng.poisson(3, n)
        pain_level = rng.randint(1, 11, n)
        wait_time_mins = rng.exponential(30, n).round(0).astype(int)

        base_prob = 0.3 + 0.4 * severity_score + 0.1 * (pain_level / 10)
        gender_penalty = (gender == 0) * bias_level * 0.25
        insurance_boost = (insurance_type == 2) * 0.15
        prob = np.clip(base_prob - gender_penalty + insurance_boost + rng.normal(0, 0.05, n), 0.01, 0.99)
        outcome = rng.binomial(1, prob)

        return pd.DataFrame({
            "gender": gender,
            "age": age,
            "severity_score": severity_score,
            "insurance_type": insurance_type,
            "prior_visits": prior_visits,
            "pain_level": pain_level,
            "wait_time_mins": wait_time_mins,
            "prioritized": outcome,
        })

    else:
        # Generic dataset
        protected = rng.choice([0, 1], size=n)
        feature_1 = rng.normal(0, 1, n).round(3)
        feature_2 = rng.uniform(0, 1, n).round(3)
        feature_3 = rng.randint(1, 10, n)

        base_prob = 0.4 + 0.3 * feature_2
        bias_effect = (protected == 1) * bias_level * 0.3
        prob = np.clip(base_prob + bias_effect + rng.normal(0, 0.05, n), 0.01, 0.99)
        outcome = rng.binomial(1, prob)

        return pd.DataFrame({
            "protected_attribute": protected,
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3,
            "outcome": outcome,
        })


def get_builtin_dataset_info():
    """Metadata for built-in dataset options shown on Upload page."""
    return [
        {
            "id": "compas",
            "name": "COMPAS Recidivism Dataset",
            "description": "Criminal recidivism predictions from Broward County, FL. "
                           "Found to be highly biased against Black defendants — twice as likely "
                           "to be falsely flagged as high-risk.",
            "rows": 7214,
            "columns": 52,
            "protected_attrs": ["race", "sex", "age"],
            "domain": "criminal_justice",
            "source": "ProPublica / AIF360",
            "known_bias": "Black defendants: 45% false-positive rate vs 23% for White defendants",
        },
        {
            "id": "adult",
            "name": "Adult Income Dataset (Census)",
            "description": "UCI Census income prediction (>50K/year). "
                           "Shows significant gender and race bias in income predictions.",
            "rows": 48842,
            "columns": 14,
            "protected_attrs": ["sex", "race"],
            "domain": "hiring",
            "source": "UCI ML Repository / AIF360",
            "known_bias": "Women earn >50K at 11% vs 31% for men in this dataset",
        },
        {
            "id": "german",
            "name": "German Credit Dataset",
            "description": "German bank credit risk classification. "
                           "Exhibits age and gender discrimination in loan approval decisions.",
            "rows": 1000,
            "columns": 20,
            "protected_attrs": ["age", "sex"],
            "domain": "lending",
            "source": "UCI ML Repository / AIF360",
            "known_bias": "Applicants over 25 face 18% higher rejection rate",
        },
    ]
