import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any
import math
import warnings
warnings.filterwarnings('ignore')


# ── Adaptive bias score weights per domain (ported from CLI v2.1) ────────────
DOMAIN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "hiring":          {"DI": 0.20, "DP": 0.15, "proxy": 0.25, "label": 0.15, "dist": 0.10, "intersect": 0.15},
    "lending":         {"DI": 0.30, "DP": 0.20, "proxy": 0.20, "label": 0.10, "dist": 0.08, "intersect": 0.12},
    "medical":         {"DI": 0.10, "DP": 0.10, "proxy": 0.10, "label": 0.35, "dist": 0.15, "intersect": 0.20},
    "criminal_justice":{"DI": 0.25, "DP": 0.15, "proxy": 0.15, "label": 0.25, "dist": 0.10, "intersect": 0.10},
    "other":           {"DI": 0.20, "DP": 0.15, "proxy": 0.15, "label": 0.15, "dist": 0.15, "intersect": 0.20},
}

_VERDICT_LEVELS = [
    (15,  "CLEAN",            "✅"),
    (35,  "MINOR ISSUES",     "🟡"),
    (55,  "MODERATE BIAS",    "🟠"),
    (75,  "SIGNIFICANT BIAS", "🔴"),
    (101, "SEVERELY BIASED",  "❌"),
]


def dynamic_min_group_size(n: int) -> int:
    """Scale minimum subgroup size with dataset size (CLI FIX-03)."""
    if n <= 20:   return 2
    if n <= 50:   return 5
    if n <= 200:  return 10
    if n <= 1000: return 20
    return 30


def safe_disparate_impact(unpriv_rate: Optional[float], priv_rate: Optional[float]) -> Optional[float]:
    """NaN / Inf / zero-division safe DI (CLI FIX-05)."""
    if priv_rate is None or unpriv_rate is None:
        return None
    try:
        if math.isnan(priv_rate) or math.isnan(unpriv_rate) or priv_rate == 0:
            return None
        return float(np.clip(unpriv_rate / priv_rate, 0, 2))
    except Exception:
        return None


class BiasEngine:
    """Core bias computation engine implementing standard fairness metrics."""

    # ── Statistical Significance (Bootstrapping) ─────────────────────────────
    
    def _bootstrap_ci(
        self, df: pd.DataFrame, outcome: str, protected: str, privileged_value: Any, 
        metric_func, n_iterations: int = 100, confidence_level: float = 0.95
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute 95% Confidence Interval and empirical p-value for a bias metric.
        H0 (null hypothesis): Metric value indicates perfect fairness (DI=1.0 or DP=0.0 gap).
        Returns: (lower_bound, upper_bound, p_value)
        """
        if len(df) < 50:
            return None, None, None
            
        try:
            results = []
            for _ in range(n_iterations):
                # Resample with replacement
                sample = df.sample(frac=1.0, replace=True)
                val = metric_func(sample, outcome, protected, privileged_value)
                results.append(val)
                
            results = np.array([r for r in results if r is not None and not math.isnan(r)])
            if len(results) < n_iterations // 2:
                return None, None, None

            # CI
            alpha = 1.0 - confidence_level
            lower = float(np.percentile(results, alpha / 2.0 * 100))
            upper = float(np.percentile(results, (1.0 - alpha / 2.0) * 100))
            
            # Empirical p-value
            # For DI, perfect fairness is 1.0. For DP gap, perfect is 0.0 (or 1.0 for normalized DP).
            # We determine the "fair" baseline based on the sample mean.
            mean_val = np.mean(results)
            if mean_val > 0.5: # Likely a 0-1 normalized metric or DI
                baseline = 1.0
                # What % of bootstrap samples cross the baseline?
                if mean_val < baseline: 
                    p_val = float(np.mean(results >= baseline)) * 2 # two-tailed
                else:
                    p_val = float(np.mean(results <= baseline)) * 2
            else: # Likely a raw gap metric centered at 0
                baseline = 0.0
                if mean_val < baseline:
                    p_val = float(np.mean(results >= baseline)) * 2
                else:
                    p_val = float(np.mean(results <= baseline)) * 2
                    
            p_val = min(1.0, float(p_val))
            return round(lower, 4), round(upper, 4), round(p_val, 4)
            
        except Exception:
            return None, None, None

    # ── Individual metric computers ──────────────────────────────────────────

    def compute_disparate_impact(
        self, df: pd.DataFrame, outcome: str, protected: str, privileged_value: Any,
        compute_stats: bool = False
    ) -> float | Tuple[float, Optional[float], Optional[float], Optional[float]]:
        """
        EEOC 4/5ths rule: ratio of selection rates.
        Multi-group: returns the minimum DI ratio between any subgroup and the privileged group.
        """
        def _calc(d, out, prot, priv):
            rates = d.groupby(prot)[out].mean()
            if priv not in rates.index:
                return 1.0
            priv_rate = float(rates[priv])
            if priv_rate == 0:
                return 1.0
            
            # Find the minimum selection rate among all groups
            min_rate = float(rates.min())
            result = safe_disparate_impact(min_rate, priv_rate)
            return float(np.clip(result, 0, 1)) if result is not None else 1.0

        score = _calc(df, outcome, protected, privileged_value)
        
        if not compute_stats:
            return score
            
        lower, upper, p_val = self._bootstrap_ci(df, outcome, protected, privileged_value, _calc)
        return score, lower, upper, p_val

    def compute_statistical_parity_diff(
        self, df: pd.DataFrame, outcome: str, protected: str, privileged_value: Any,
        compute_stats: bool = False
    ) -> float | Tuple[float, Optional[float], Optional[float], Optional[float]]:
        """
        Multi-group: returns 1 - max(absolute difference from privileged rate).
        Normalized 0-1 where 1 = perfect fairness.
        """
        def _calc(d, out, prot, priv):
            rates = d.groupby(prot)[out].mean()
            if priv not in rates.index:
                return 1.0
            priv_rate = float(rates[priv])
            
            # Find the largest absolute gap between any group and the privileged group
            gaps = (rates - priv_rate).abs()
            max_gap = float(gaps.max())
            return float(np.clip(1.0 - max_gap, 0, 1))

        score = _calc(df, outcome, protected, privileged_value)
        
        if not compute_stats:
            return score
            
        lower, upper, p_val = self._bootstrap_ci(df, outcome, protected, privileged_value, _calc)
        return score, lower, upper, p_val
        
        if not compute_stats:
            return score
            
        lower, upper, p_val = self._bootstrap_ci(df, outcome, protected, privileged_value, _calc)
        return score, lower, upper, p_val

    def compute_raw_dp_gap(
        self, df: pd.DataFrame, outcome: str, protected: str, privileged_value: Any
    ) -> Optional[float]:
        """Raw demographic parity gap (signed). For multi-group, returns the largest magnitude gap from privileged."""
        try:
            rates = df.groupby(protected)[outcome].mean()
            if privileged_value not in rates.index:
                return 0.0
            priv_rate = rates[privileged_value]
            gaps = rates - priv_rate
            # Find the largest magnitude gap
            idx_max = gaps.abs().idxmax()
            return round(float(gaps[idx_max]), 4)
        except Exception:
            return None

    def compute_equal_opportunity_diff(
        self, df: pd.DataFrame, outcome: str, predictions: str,
        protected: str, privileged_value: Any,
    ) -> float:
        """Equal opportunity: max difference in true positive rates. Returns 0-1."""
        try:
            # TPR per group
            def get_tpr(subset):
                pos = subset[subset[outcome] == 1]
                return float((pos[predictions] == 1).mean()) if len(pos) > 0 else None
            
            group_tprs = {}
            for val in df[protected].unique():
                tpr = get_tpr(df[df[protected] == val])
                if tpr is not None:
                    group_tprs[val] = tpr
            
            if privileged_value not in group_tprs or len(group_tprs) < 2:
                return 1.0
                
            priv_tpr = group_tprs[privileged_value]
            max_diff = max(abs(tpr - priv_tpr) for tpr in group_tprs.values())
            return float(np.clip(1.0 - max_diff, 0, 1))
        except Exception:
            return 1.0

    def compute_average_odds_diff(
        self, df: pd.DataFrame, outcome: str, predictions: str,
        protected: str, privileged_value: Any,
    ) -> float:
        """Average odds: max mean of |TPR diff| and |FPR diff|. Returns 0-1."""
        try:
            def get_rates(subset):
                pos = subset[subset[outcome] == 1]
                neg = subset[subset[outcome] == 0]
                tpr = float((pos[predictions] == 1).mean()) if len(pos) > 0 else None
                fpr = float((neg[predictions] == 1).mean()) if len(neg) > 0 else None
                return tpr, fpr
            
            group_stats = {}
            for val in df[protected].unique():
                tpr, fpr = get_rates(df[df[protected] == val])
                if tpr is not None and fpr is not None:
                    group_stats[val] = (tpr, fpr)
            
            if privileged_value not in group_stats or len(group_stats) < 2:
                return 1.0
                
            t_p, f_p = group_stats[privileged_value]
            max_aod = 0.0
            for t_u, f_u in group_stats.values():
                aod = 0.5 * (abs(f_u - f_p) + abs(t_u - t_p))
                max_aod = max(max_aod, aod)
                
            return float(np.clip(1.0 - max_aod, 0, 1))
        except Exception:
            return 1.0

    def compute_theil_index(self, df: pd.DataFrame, outcome: str, protected: str) -> float:
        """Theil index measures inequality in benefit distribution. Returns 0-1."""
        try:
            group_rates  = df.groupby(protected)[outcome].mean()
            overall_rate = df[outcome].mean()
            if overall_rate == 0 or len(group_rates) < 2:
                return 1.0
            ratios = (group_rates / overall_rate).replace(0, 1e-10)
            theil  = float(-np.mean(np.log(ratios)))
            return float(np.clip(1.0 - min(abs(theil), 1.0), 0, 1))
        except Exception:
            return 1.0

    def compute_individual_fairness(
        self, df: pd.DataFrame, outcome: str, protected: str
    ) -> float:
        """Individual fairness via KNN consistency. Returns 0-1."""
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
            scores = [1 - abs(y[i] - y[neighbors].mean()) for i, neighbors in enumerate(indices)]
            return float(np.clip(np.mean(scores), 0, 1))
        except Exception:
            return 1.0

    def compute_calibration(
        self, df: pd.DataFrame, outcome: str, predictions: str, protected: str
    ) -> float:
        """Calibration: predicted prob matches actual rate equally across groups. Returns 0-1."""
        try:
            if predictions not in df.columns:
                return 1.0
            cals = []
            for group in df[protected].unique():
                sub = df[df[protected] == group]
                if len(sub) < 5:
                    continue
                cals.append(abs(sub[outcome].mean() - sub[predictions].mean()))
            if not cals:
                return 1.0
            cal_diff = max(cals) - min(cals)
            return float(np.clip(1.0 - cal_diff, 0, 1))
        except Exception:
            return 1.0

    # ── Composite score (0-1 scale, legacy) ─────────────────────────────────

    def compute_bias_score(self, metrics: dict) -> float:
        """
        Composite fairness score (0-1, 1=perfectly fair).
        Weighted average of all computed fairness metrics.
        """
        weights = {
            "disparate_impact":  0.25,
            "statistical_parity": 0.15,
            "equal_opportunity": 0.15,
            "average_odds":      0.15,
            "theil_index":       0.10,
            "individual_fairness": 0.10,
            "calibration":       0.10,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for key, w in weights.items():
            if key in metrics:
                weighted_sum += metrics[key] * w
                total_weight += w
        if total_weight == 0:
            return 1.0
        return float(np.clip(weighted_sum / total_weight, 0, 1))

    # ── Adaptive bias score breakdown (0-100 scale, CLI-ported) ─────────────

    def compute_bias_score_breakdown(
        self,
        raw_metrics: Dict[str, float],
        proxy_results: List[Dict[str, Any]],
        intersectional_results: List[Dict[str, Any]],
        df: pd.DataFrame,
        protected_attrs: List[str],
        domain: str = "other",
        predictions_col: Optional[str] = None,
        outcome_col: Optional[str] = None,
        privileged_value: Any = None
    ) -> Dict[str, Any]:
        """
        Adaptive, domain-weighted bias score (0–100).
        Ported from CLI v2.1 compute_bias_score().
        Returns: score, verdict, weights, severities, contributions, primary_driver, quick_win.
        """
        n = len(df)
        w = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["other"]).copy()

        # Size reliability: DI & intersect less reliable on small datasets
        size_factor = 0.2 if n < 100 else 0.5 if n < 500 else 0.8 if n < 2000 else 1.0
        for key in ("DI", "intersect"):
            w[key] *= size_factor

        # Group imbalance: downweight intersect if heavily imbalanced
        if protected_attrs:
            try:
                shares    = df[protected_attrs[0]].dropna().value_counts(normalize=True)
                min_share = float(shares.min()) if len(shares) > 0 else 0.0
                imb_factor = (0.3 if min_share < 0.05 else 0.6 if min_share < 0.15
                               else 0.85 if min_share < 0.30 else 1.0)
                w["intersect"] *= imb_factor
            except Exception:
                pass

        # Renormalize to sum = 1.0
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}

        # ── Raw severities (0–100) ──────────────────────────────────────────
        di_raw  = raw_metrics.get("disparate_impact", 1.0)
        di_sev  = max(0.0, (0.8 - di_raw) / 0.8) * 100

        dp_norm     = raw_metrics.get("statistical_parity", 1.0)
        dp_gap      = abs(1.0 - dp_norm)          # unnormalize back to gap magnitude
        dp_sev      = min(dp_gap / 0.5, 1.0) * 100

        proxy_sev   = min(len(proxy_results) / 5.0, 1.0) * 100

        # Distribution severity: use the worst SD-gap proxy via IQR spread per attr
        dist_sev = 0.0
        try:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            numeric_feats = [c for c in numeric_cols if c not in protected_attrs and df[c].std() > 0]
            if numeric_feats and protected_attrs:
                for attr in protected_attrs[:1]:
                    group_means = df.groupby(attr)[numeric_feats].mean()
                    if len(group_means) >= 2:
                        pooled_std = df[numeric_feats].std().mean()
                        mean_spread = (group_means.max() - group_means.min()).mean()
                        sd_gap = float(mean_spread / pooled_std) if pooled_std > 0 else 0.0
                        dist_sev = min(sd_gap / 3.0, 1.0) * 100
        except Exception:
            pass

        worst_dev   = 0.0
        if intersectional_results:
            worst_dev = max(
                (abs(r.get("positive_rate", 0.5) - r.get("overall_rate", 0.5))
                 for r in intersectional_results), default=0.0
            )
        inter_sev = min(worst_dev / 0.5, 1.0) * 100

        label_sev = 0.0
        if predictions_col and outcome_col and protected_attrs and privileged_value is not None:
            try:
                primary_attr = protected_attrs[0]
                # Compare error rates: P(Ŷ != Y)
                err_series = df[predictions_col] != df[outcome_col]
                
                error_rates = err_series.groupby(df[primary_attr]).mean()
                if privileged_value in error_rates.index:
                    priv_err = error_rates[privileged_value]
                    # Max magnitude of error rate difference from privileged group
                    # If any group is mislabeled more than the privileged group, it's a sign of bias.
                    max_err_gap = (error_rates - priv_err).max()
                    label_sev = min(max(0, float(max_err_gap)) / 0.15, 1.0) * 100
            except Exception:
                pass

        severities = {
            "DI":       round(di_sev,    1),
            "DP":       round(dp_sev,    1),
            "proxy":    round(proxy_sev, 1),
            "label":    round(label_sev, 1),
            "dist":     round(dist_sev,  1),
            "intersect":round(inter_sev, 1),
        }

        contributions = {k: round(w[k] * severities[k], 1) for k in w}
        score         = round(sum(contributions.values()), 1)
        primary       = max(contributions, key=lambda k: contributions[k])

        quick_wins = sorted(
            [(v, k, round(score - v, 1)) for k, v in contributions.items() if v > 0],
            reverse=True,
        )
        qw = list(quick_wins[0]) if quick_wins else None

        verdict = next(label for threshold, label, _ in _VERDICT_LEVELS if score <= threshold)

        labels = {
            "DI": "Disparate Impact", "DP": "Demog. Parity",
            "proxy": "Proxy vars", "label": "Label bias",
            "dist": "Distribution", "intersect": "Intersectional",
        }

        return {
            "score":          score,
            "verdict":        verdict,
            "domain":         domain,
            "weights":        {k: round(v, 4) for k, v in w.items()},
            "severities":     severities,
            "contributions":  contributions,
            "primary_driver": primary,
            "quick_win":      qw,
            "labels":         labels,
        }

    # ── Bias fingerprint (radar chart) ───────────────────────────────────────

    def compute_fingerprint(
        self,
        df: pd.DataFrame,
        outcome: str,
        all_metrics: dict,
        proxy_scores: list,
        intersectional_results: list,
        protected_attrs: Dict[str, Any],
    ) -> dict:
        """8-axis bias fingerprint for radar chart. 0=most biased, 1=most fair."""
        def _di_for(keywords):
            attrs = [a for a in protected_attrs if any(k in a.lower() for k in keywords)]
            if not attrs:
                return 0.5
            attr = attrs[0]
            priv = protected_attrs.get(attr)
            return self.compute_disparate_impact(df, outcome, attr, priv) if priv is not None else 0.5

        gender_bias   = _di_for(["gender", "sex"])
        race_bias     = _di_for(["race", "ethnic"])
        age_bias      = _di_for(["age"])

        proxy_risk = 1.0
        if proxy_scores:
            max_corr = max((p.get("correlation", 0) for p in proxy_scores), default=0.0)
            proxy_risk = float(1.0 - max_corr)

        intersectional = 1.0
        if intersectional_results:
            worst_gap = max(
                (abs(r.get("positive_rate", 0.5) - r.get("overall_rate", 0.5)) for r in intersectional_results),
                default=0.0,
            )
            intersectional = float(1.0 - min(worst_gap * 2, 1.0))

        provenance   = all_metrics.get("calibration", 0.8)
        metric_vals  = [v for v in all_metrics.values() if isinstance(v, float)]
        failing      = sum(1 for v in metric_vals if v < 0.8)
        severity     = float(1.0 - (failing / len(metric_vals))) if metric_vals else 1.0

        spread = 1.0
        for attr_name in protected_attrs:
            try:
                group_rates = df.groupby(attr_name)[outcome].mean()
                std_dev = group_rates.std()
                spread  = min(spread, float(1.0 - min(std_dev * 2, 1.0)))
            except Exception:
                pass

        return {
            "gender_bias":    round(gender_bias,    3),
            "race_bias":      round(race_bias,      3),
            "age_bias":       round(age_bias,       3),
            "proxy_risk":     round(proxy_risk,     3),
            "intersectional": round(intersectional, 3),
            "provenance":     round(provenance,     3),
            "severity":       round(severity,       3),
            "spread":         round(spread,         3),
        }

    # ── Human impact estimate ─────────────────────────────────────────────────

    def compute_impact(self, bias_gap: float, scale: int, years: int) -> dict:
        affected_total    = int(scale * abs(bias_gap) * years)
        affected_per_month = round(affected_total / max(years * 12, 1), 1)
        return {
            "affected_total":     affected_total,
            "affected_per_month": affected_per_month,
            "bias_gap":           round(bias_gap, 4),
        }

    def get_risk_level(self, bias_score: float) -> str:
        """Map composite bias score (0-1) to risk level."""
        if bias_score < 0.5:   return "CRITICAL"
        elif bias_score < 0.65: return "HIGH"
        elif bias_score < 0.80: return "MEDIUM"
        elif bias_score < 0.90: return "LOW"
        else:                   return "PASS"
