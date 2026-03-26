import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pointbiserialr
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any
import math
import warnings
warnings.filterwarnings('ignore')


class ProxyDetector:
    """
    Detects features that act as proxies for protected attributes.
    Uses Cramér's V for categorical pairs and point-biserial / Pearson
    for numeric pairs. De-duplicates by keeping the best correlation
    per (feature, protected_attr) pair (CLI v2.1 improvement).
    """

    RISK_THRESHOLDS = {
        "low":       0.3,
        "medium":    0.5,
        "high":      0.7,
        "confirmed": 0.8,
    }

    def detect_proxies(
        self, df: pd.DataFrame, protected_attrs: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Compute correlation between all non-protected features and protected attributes.
        Returns de-duplicated, ranked list of proxy risks.
        """
        le = LabelEncoder()
        raw_results: List[Dict[str, Any]] = []

        for protected in protected_attrs:
            if protected not in df.columns:
                continue

            feature_cols = [c for c in df.columns if c != protected]
            n_unique = df[protected].dropna().nunique()

            # ── Binary / low-cardinality protected → point-biserial per value ──
            if n_unique <= 10:
                for val in df[protected].dropna().unique():
                    binary = (df[protected] == val).astype(float).values
                    for feature in feature_cols:
                        corr = self._best_corr_pbiserial(df[feature], binary)
                        if corr is None:
                            continue
                        raw_results.append(self._make_result(
                            feature, f"{protected}={val}", corr
                        ))
            else:
                # ── High-cardinality numeric protected → encode + Pearson / PB ──
                try:
                    protected_enc = le.fit_transform(
                        df[protected].astype(str).fillna("unknown")
                    ).astype(float)
                except Exception:
                    continue

                for feature in feature_cols:
                    corr = self._best_corr_encoded(df[feature], protected_enc)
                    if corr is None:
                        continue
                    raw_results.append(self._make_result(feature, protected, corr))

        # ── De-duplicate: keep best correlation per (feature, protected_attr) ──
        seen: Dict[tuple, Dict[str, Any]] = {}
        for r in raw_results:
            key = (r["feature"], r["protected_attr"])
            if key not in seen or abs(r["correlation"]) > abs(seen[key]["correlation"]):
                seen[key] = r

        deduped = sorted(seen.values(), key=lambda x: x["correlation"], reverse=True)

        # Enrich with risk level and recommendation
        results = []
        for r in deduped:
            r["risk_level"]     = self._get_risk_level(r["correlation"])
            r["recommendation"] = self._get_recommendation(
                r["feature"], r["protected_attr"], r["correlation"]
            )
            results.append(r)

        return results

    # ── Correlation helpers ──────────────────────────────────────────────────

    def _best_corr_pbiserial(
        self, series: pd.Series, binary: np.ndarray
    ) -> float | None:
        """Point-biserial correlation between a binary array and a feature series."""
        try:
            col = series.fillna(0)
            if col.dtype == object or str(col.dtype) == "category":
                col = LabelEncoder().fit_transform(col.astype(str)).astype(float)
            else:
                col = col.values.astype(float)
            if len(np.unique(col)) < 2:
                return None
            r, p = pointbiserialr(binary, col)
            if math.isnan(r):
                return None
            return round(abs(float(r)), 4)
        except Exception:
            return None

    def _best_corr_encoded(
        self, series: pd.Series, protected_enc: np.ndarray
    ) -> float | None:
        """Pearson or point-biserial correlation for an encoded feature series."""
        try:
            col = series.fillna(0)
            if col.dtype == object or str(col.dtype) == "category":
                if col.nunique() <= 10:
                    col = LabelEncoder().fit_transform(col.astype(str)).astype(float)
                    r, p = pointbiserialr(col, protected_enc)
                else:
                    return None
            else:
                col = col.values.astype(float)
                if len(np.unique(col)) < 2:
                    return None
                r, p = stats.pearsonr(col, protected_enc)
            if math.isnan(r):
                return None
            return round(abs(float(r)), 4)
        except Exception:
            return None

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Cramér's V association for categorical × categorical."""
        try:
            cm    = pd.crosstab(x, y)
            chi2  = stats.chi2_contingency(cm)[0]
            n     = len(x)
            phi2  = chi2 / n
            r, k  = cm.shape
            phi2c = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rc    = r - ((r-1)**2)/(n-1)
            kc    = k - ((k-1)**2)/(n-1)
            denom = min(kc-1, rc-1)
            return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0
        except Exception:
            return 0.0

    # ── Result helpers ────────────────────────────────────────────────────────

    def _make_result(self, feature: str, protected_attr: str, corr: float) -> Dict[str, Any]:
        return {
            "feature":        feature,
            "protected_attr": protected_attr,
            "correlation":    round(float(corr), 4),
        }

    def _get_risk_level(self, corr: float) -> str:
        if corr >= self.RISK_THRESHOLDS["confirmed"]: return "proxy_confirmed"
        if corr >= self.RISK_THRESHOLDS["high"]:      return "high_risk"
        if corr >= self.RISK_THRESHOLDS["medium"]:    return "medium_risk"
        if corr >= self.RISK_THRESHOLDS["low"]:       return "low_risk"
        return "no_risk"

    def _get_recommendation(self, feature: str, protected: str, corr: float) -> str:
        pct = int(corr * 100)
        if corr >= 0.8:
            return (f"'{feature}' is {pct}% correlated with '{protected}' — "
                    f"confirmed proxy. Remove or apply Disparate Impact Remover before training.")
        if corr >= 0.5:
            return (f"'{feature}' has significant {pct}% correlation with '{protected}'. "
                    f"Consider transforming or removing. Apply repair_level=0.8 with DIR.")
        if corr >= 0.3:
            return (f"'{feature}' has moderate {pct}% correlation with '{protected}'. "
                    f"Monitor during model training.")
        return f"'{feature}' shows low correlation ({pct}%) with '{protected}'."

    # ── Heatmap ───────────────────────────────────────────────────────────────

    def build_heatmap(
        self,
        proxy_results: List[Dict[str, Any]],
        all_features: List[str],
        protected_attrs: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Build feature × protected_attr correlation matrix for D3 heatmap."""
        lookup = {(r["feature"], r["protected_attr"]): r["correlation"] for r in proxy_results}
        return {
            feature: {pa: lookup.get((feature, pa), 0.0) for pa in protected_attrs}
            for feature in all_features
        }
