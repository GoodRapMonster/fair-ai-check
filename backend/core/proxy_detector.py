import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ProxyDetector:
    """
    Detects features that act as proxies for protected attributes.
    Uses correlation analysis to identify hidden discrimination vectors.
    """

    RISK_THRESHOLDS = {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "confirmed": 0.8,
    }

    def detect_proxies(
        self, df: pd.DataFrame, protected_attrs: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Compute correlation between all non-protected features and protected attributes.
        Returns ranked list of proxy risks.
        """
        le = LabelEncoder()
        results = []

        for protected in protected_attrs:
            if protected not in df.columns:
                continue

            # Encode protected attribute
            try:
                protected_encoded = le.fit_transform(
                    df[protected].astype(str).fillna("unknown")
                )
            except Exception:
                continue

            # Check all other columns
            feature_cols = [
                c for c in df.columns if c != protected
            ]

            for feature in feature_cols:
                try:
                    col_data = df[feature].fillna(0)

                    if col_data.dtype == object or str(col_data.dtype) == "category":
                        feature_encoded = le.fit_transform(
                            col_data.astype(str)
                        )
                    else:
                        feature_encoded = col_data.values.astype(float)

                    # Compute Cramer's V for categorical, Pearson for numeric
                    if df[feature].dtype == object or df[feature].nunique() < 10:
                        corr = self._cramers_v(
                            df[feature].astype(str),
                            df[protected].astype(str),
                        )
                    else:
                        corr, _ = stats.pearsonr(feature_encoded, protected_encoded)
                        corr = abs(corr)

                    if np.isnan(corr):
                        corr = 0.0

                    risk_level = self._get_risk_level(corr)
                    recommendation = self._get_recommendation(feature, protected, corr)

                    results.append({
                        "feature": feature,
                        "protected_attr": protected,
                        "correlation": round(float(corr), 4),
                        "risk_level": risk_level,
                        "recommendation": recommendation,
                    })
                except Exception:
                    continue

        # Sort by correlation descending
        results.sort(key=lambda x: x["correlation"], reverse=True)
        return results

    def build_heatmap(
        self, proxy_results: List[Dict[str, Any]], all_features: List[str], protected_attrs: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Build matrix of feature x protected_attr correlations for D3 heatmap.
        """
        heatmap = {}
        lookup = {(r["feature"], r["protected_attr"]): r["correlation"] for r in proxy_results}

        for feature in all_features:
            heatmap[feature] = {}
            for protected in protected_attrs:
                heatmap[feature][protected] = lookup.get((feature, protected), 0.0)

        return heatmap

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Compute Cramér's V association statistic for categorical variables."""
        try:
            confusion_matrix = pd.crosstab(x, y)
            chi2 = stats.chi2_contingency(confusion_matrix)[0]
            n = len(x)
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            denom = min((kcorr - 1), (rcorr - 1))
            if denom <= 0:
                return 0.0
            return float(np.sqrt(phi2corr / denom))
        except Exception:
            return 0.0

    def _get_risk_level(self, corr: float) -> str:
        if corr >= self.RISK_THRESHOLDS["confirmed"]:
            return "proxy_confirmed"
        elif corr >= self.RISK_THRESHOLDS["high"]:
            return "high_risk"
        elif corr >= self.RISK_THRESHOLDS["medium"]:
            return "medium_risk"
        elif corr >= self.RISK_THRESHOLDS["low"]:
            return "low_risk"
        else:
            return "no_risk"

    def _get_recommendation(self, feature: str, protected: str, corr: float) -> str:
        pct = int(corr * 100)
        if corr >= 0.8:
            return (
                f"'{feature}' is {pct}% correlated with '{protected}' — "
                f"it is acting as a confirmed proxy. Remove this feature or apply "
                f"Disparate Impact Remover before training."
            )
        elif corr >= 0.5:
            return (
                f"'{feature}' has significant {pct}% correlation with '{protected}'. "
                f"Consider transforming or removing this feature. "
                f"Apply repair_level=0.8 with DIR."
            )
        elif corr >= 0.3:
            return (
                f"'{feature}' has moderate {pct}% correlation with '{protected}'. "
                f"Monitor this feature during model training."
            )
        else:
            return f"'{feature}' shows low correlation ({pct}%) with '{protected}'."
