import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class BiasEngine:
    """Core bias computation engine implementing standard fairness metrics."""

    def compute_disparate_impact(
        self, df: pd.DataFrame, outcome: str, protected: str, privileged_value: Any
    ) -> float:
        """
        EEOC 4/5ths rule: ratio of positive outcome rates between groups.
        DI = P(Y=1|unprivileged) / P(Y=1|privileged)
        Score of 1.0 = perfect fairness. Below 0.8 = disparate impact exists.
        """
        try:
            privileged = df[df[protected] == privileged_value]
            unprivileged = df[df[protected] != privileged_value]

            if len(privileged) == 0 or len(unprivileged) == 0:
                return 1.0

            priv_rate = privileged[outcome].mean()
            unpriv_rate = unprivileged[outcome].mean()

            if priv_rate == 0:
                return 1.0

            return float(np.clip(unpriv_rate / priv_rate, 0, 1))
        except Exception:
            return 1.0

    def compute_statistical_parity_diff(
        self, df: pd.DataFrame, outcome: str, protected: str, privileged_value: Any
    ) -> float:
        """
        Statistical parity difference: P(Y=1|unprivileged) - P(Y=1|privileged)
        Score of 0 = perfect fairness. Negative = discrimination against unprivileged.
        Returns normalized to 0-1 where 1=fair (abs diff normalized from -1..0..1 to 0..1)
        """
        try:
            privileged = df[df[protected] == privileged_value]
            unprivileged = df[df[protected] != privileged_value]

            if len(privileged) == 0 or len(unprivileged) == 0:
                return 1.0

            diff = unprivileged[outcome].mean() - privileged[outcome].mean()
            # Normalize: 0 diff → score 1.0, -1 or +1 diff → score 0.0
            return float(np.clip(1.0 - abs(diff), 0, 1))
        except Exception:
            return 1.0

    def compute_equal_opportunity_diff(
        self,
        df: pd.DataFrame,
        outcome: str,
        predictions: str,
        protected: str,
        privileged_value: Any,
    ) -> float:
        """
        Equal opportunity: difference in true positive rates between groups.
        EOD = TPR(unprivileged) - TPR(privileged)
        Score of 0 = perfect. Returns normalized 0-1.
        """
        try:
            def tpr(subset):
                positives = subset[subset[outcome] == 1]
                if len(positives) == 0:
                    return 0.0
                return (positives[predictions] == 1).mean()

            priv = df[df[protected] == privileged_value]
            unpriv = df[df[protected] != privileged_value]

            diff = tpr(unpriv) - tpr(priv)
            return float(np.clip(1.0 - abs(diff), 0, 1))
        except Exception:
            return 1.0

    def compute_average_odds_diff(
        self,
        df: pd.DataFrame,
        outcome: str,
        predictions: str,
        protected: str,
        privileged_value: Any,
    ) -> float:
        """
        Average odds: mean of TPR diff and FPR diff between groups.
        AOD = 0.5 * ((FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv))
        Returns normalized 0-1.
        """
        try:
            def tpr_fpr(subset):
                pos = subset[subset[outcome] == 1]
                neg = subset[subset[outcome] == 0]
                tpr = (pos[predictions] == 1).mean() if len(pos) > 0 else 0
                fpr = (neg[predictions] == 1).mean() if len(neg) > 0 else 0
                return tpr, fpr

            priv = df[df[protected] == privileged_value]
            unpriv = df[df[protected] != privileged_value]

            tpr_p, fpr_p = tpr_fpr(priv)
            tpr_u, fpr_u = tpr_fpr(unpriv)

            aod = 0.5 * ((fpr_u - fpr_p) + (tpr_u - tpr_p))
            return float(np.clip(1.0 - abs(aod), 0, 1))
        except Exception:
            return 1.0

    def compute_theil_index(
        self, df: pd.DataFrame, outcome: str, protected: str
    ) -> float:
        """
        Theil index measures inequality in benefit distribution.
        Lower Theil = more equal. Returns normalized 0-1 score (1=fair).
        """
        try:
            group_rates = df.groupby(protected)[outcome].mean()
            overall_rate = df[outcome].mean()

            if overall_rate == 0 or len(group_rates) < 2:
                return 1.0

            # Generalized Entropy Index (GE0 = Theil L)
            ratios = group_rates / overall_rate
            ratios = ratios.replace(0, 1e-10)
            theil = -np.mean(np.log(ratios))

            # Normalize to 0-1 (theil > 1 is extreme, clip at 1)
            return float(np.clip(1.0 - min(abs(theil), 1.0), 0, 1))
        except Exception:
            return 1.0

    def compute_individual_fairness(
        self, df: pd.DataFrame, outcome: str, protected: str
    ) -> float:
        """
        Individual fairness: similar individuals should receive similar outcomes.
        Approximated via consistency score across K nearest neighbors.
        """
        try:
            from sklearn.neighbors import NearestNeighbors

            feature_cols = [c for c in df.columns if c not in [outcome, protected]]
            if not feature_cols:
                return 1.0

            X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
            if len(X) < 10:
                return 1.0

            n_neighbors = min(5, len(X) - 1)
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
            _, indices = nbrs.kneighbors(X)

            y = df[outcome].values
            consistency_scores = []
            for i, neighbors in enumerate(indices):
                neighbor_outcomes = y[neighbors]
                consistency = 1 - abs(y[i] - neighbor_outcomes.mean())
                consistency_scores.append(consistency)

            return float(np.clip(np.mean(consistency_scores), 0, 1))
        except Exception:
            return 1.0

    def compute_calibration(
        self, df: pd.DataFrame, outcome: str, predictions: str, protected: str
    ) -> float:
        """
        Calibration: predicted probabilities match actual outcomes equally across groups.
        Returns normalized score (1 = well calibrated).
        """
        try:
            if predictions not in df.columns:
                return 1.0

            groups = df[protected].unique()
            group_calibrations = []

            for group in groups:
                subset = df[df[protected] == group]
                if len(subset) < 5:
                    continue
                actual = subset[outcome].mean()
                predicted = subset[predictions].mean()
                group_calibrations.append(abs(actual - predicted))

            if not group_calibrations:
                return 1.0

            cal_diff = max(group_calibrations) - min(group_calibrations)
            return float(np.clip(1.0 - cal_diff, 0, 1))
        except Exception:
            return 1.0

    def compute_bias_score(self, metrics: dict) -> float:
        """
        Composite fairness score (0-1, 1=perfectly fair).
        Weighted average of all computed fairness metrics.
        """
        weights = {
            "disparate_impact": 0.25,
            "statistical_parity": 0.15,
            "equal_opportunity": 0.15,
            "average_odds": 0.15,
            "theil_index": 0.10,
            "individual_fairness": 0.10,
            "calibration": 0.10,
        }

        total_weight = 0
        weighted_sum = 0

        for metric_key, weight in weights.items():
            if metric_key in metrics:
                weighted_sum += metrics[metric_key] * weight
                total_weight += weight

        if total_weight == 0:
            return 1.0

        return float(np.clip(weighted_sum / total_weight, 0, 1))

    def compute_fingerprint(
        self,
        df: pd.DataFrame,
        outcome: str,
        all_metrics: dict,
        proxy_scores: list,
        intersectional_results: list,
        protected_attrs: Dict[str, Any],
    ) -> dict:
        """
        8-axis bias fingerprint for radar chart visualization.
        Returns values 0-1 where 0=most biased, 1=most fair.
        (For the radar chart, distance from center = fairness)
        """
        # Gender bias
        gender_attrs = [a for a in protected_attrs if "gender" in a.lower() or "sex" in a.lower()]
        gender_bias = 0.5
        if gender_attrs:
            attr = gender_attrs[0]
            priv = protected_attrs.get(attr)
            gender_bias = self.compute_disparate_impact(df, outcome, attr, priv)

        # Race bias
        race_attrs = [a for a in protected_attrs if "race" in a.lower() or "ethnic" in a.lower()]
        race_bias = 0.5
        if race_attrs:
            attr = race_attrs[0]
            priv = protected_attrs.get(attr)
            race_bias = self.compute_disparate_impact(df, outcome, attr, priv)

        # Age bias
        age_attrs = [a for a in protected_attrs if "age" in a.lower()]
        age_bias = 0.5
        if age_attrs:
            attr = age_attrs[0]
            priv = protected_attrs.get(attr)
            age_bias = self.compute_disparate_impact(df, outcome, attr, priv)

        # Proxy risk (normalized max correlation)
        proxy_risk = 1.0
        if proxy_scores:
            max_corr = max([p.get("correlation", 0) for p in proxy_scores], default=0)
            proxy_risk = float(1.0 - max_corr)

        # Intersectional (worst subgroup gap)
        intersectional = 1.0
        if intersectional_results:
            worst_gap = max(
                [abs(r.get("positive_rate", 0.5) - r.get("overall_rate", 0.5)) for r in intersectional_results],
                default=0,
            )
            intersectional = float(1.0 - min(worst_gap * 2, 1.0))

        # Provenance (use calibration as proxy)
        provenance = all_metrics.get("calibration", 0.8)

        # Severity (fraction of metrics failing — threshold 0.8)
        metric_vals = [v for v in all_metrics.values() if isinstance(v, float)]
        failing = sum(1 for v in metric_vals if v < 0.8)
        severity = float(1.0 - (failing / len(metric_vals))) if metric_vals else 1.0

        # Spread (std deviation of outcome rates across groups)
        spread = 1.0
        for attr_name in protected_attrs:
            try:
                group_rates = df.groupby(attr_name)[outcome].mean()
                std_dev = group_rates.std()
                spread = min(spread, float(1.0 - min(std_dev * 2, 1.0)))
            except Exception:
                pass

        return {
            "gender_bias": round(gender_bias, 3),
            "race_bias": round(race_bias, 3),
            "age_bias": round(age_bias, 3),
            "proxy_risk": round(proxy_risk, 3),
            "intersectional": round(intersectional, 3),
            "provenance": round(provenance, 3),
            "severity": round(severity, 3),
            "spread": round(spread, 3),
        }

    def compute_impact(self, bias_gap: float, scale: int, years: int) -> dict:
        """
        Estimate real human impact of bias.
        affected = scale * bias_gap * years
        """
        affected_total = int(scale * abs(bias_gap) * years)
        affected_per_month = round(affected_total / max(years * 12, 1), 1)
        return {
            "affected_total": affected_total,
            "affected_per_month": affected_per_month,
            "bias_gap": round(bias_gap, 4),
        }

    def get_risk_level(self, bias_score: float) -> str:
        """Map composite bias score to risk level."""
        if bias_score < 0.5:
            return "CRITICAL"
        elif bias_score < 0.65:
            return "HIGH"
        elif bias_score < 0.80:
            return "MEDIUM"
        elif bias_score < 0.90:
            return "LOW"
        else:
            return "PASS"
