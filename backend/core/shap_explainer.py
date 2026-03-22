import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    Computes SHAP feature importance values for the trained model,
    providing explainability for individual predictions and global feature importance.
    """

    def compute_shap_values(
        self,
        model,
        X: pd.DataFrame,
        max_samples: int = 200,
    ) -> Dict[str, float]:
        """
        Compute mean absolute SHAP values per feature.
        Returns dict of {feature_name: importance_score}.
        """
        try:
            import shap
            # Use a sample for performance
            sample_size = min(max_samples, len(X))
            X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X

            explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="correlation_dependent")
            shap_values = explainer.shap_values(X_sample)

            if shap_values is None:
                return self._fallback_importance(model, X)

            # Mean absolute SHAP per feature
            if isinstance(shap_values, list):
                shap_abs = np.abs(shap_values[1]).mean(axis=0)
            else:
                shap_abs = np.abs(shap_values).mean(axis=0)

            total = shap_abs.sum()
            if total == 0:
                return self._fallback_importance(model, X)

            importance = {}
            for i, col in enumerate(X.columns):
                importance[col] = round(float(shap_abs[i] / total), 4)

            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        except Exception:
            return self._fallback_importance(model, X)

    def _fallback_importance(
        self, model, X: pd.DataFrame
    ) -> Dict[str, float]:
        """Use model coefficients as fallback when SHAP fails."""
        try:
            coefs = np.abs(model.coef_[0])
            total = coefs.sum()
            if total == 0:
                return {col: 1/len(X.columns) for col in X.columns}
            importance = {col: round(float(c/total), 4) for col, c in zip(X.columns, coefs)}
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {col: round(1/len(X.columns), 4) for col in X.columns}
