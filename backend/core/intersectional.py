import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class IntersectionalAnalyzer:
    """
    Detects bias that only appears at the intersection of multiple protected attributes.
    Identifies 'invisible bias' missed by single-attribute analysis.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        outcome: str,
        protected_attrs: List[str],
        privileged_groups: Dict[str, Any],
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Compute outcome rates for all subgroup combinations up to max_depth.
        Returns ranked list of subgroup results with invisible bias flags.
        """
        overall_rate = df[outcome].mean()
        results = []
        valid_attrs = [a for a in protected_attrs if a in df.columns]

        # Generate combinations of protected attributes (depth 1 and 2)
        for depth in range(1, min(max_depth + 1, len(valid_attrs) + 1)):
            for attr_combo in combinations(valid_attrs, depth):
                subgroup_results = self._analyze_subgroup(
                    df, outcome, list(attr_combo), overall_rate
                )
                results.extend(subgroup_results)

        return sorted(results, key=lambda x: abs(x["positive_rate"] - x["overall_rate"]), reverse=True)

    def _analyze_subgroup(
        self,
        df: pd.DataFrame,
        outcome: str,
        attrs: List[str],
        overall_rate: float,
    ) -> List[Dict[str, Any]]:
        """Analyze outcome rates for a specific combination of attributes."""
        results = []

        try:
            groups = df.groupby(attrs)
        except Exception:
            return results

        for group_keys, group_df in groups:
            if len(group_df) < 10:  # Skip tiny groups
                continue

            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)

            subgroup_label = " & ".join(
                f"{attr}={val}" for attr, val in zip(attrs, group_keys)
            )

            pos_rate = float(group_df[outcome].mean())
            gap = pos_rate - overall_rate

            # Severity: normalized distance from overall rate
            severity = float(min(abs(gap) * 2, 1.0))

            results.append({
                "subgroup": subgroup_label,
                "attributes": attrs,
                "size": len(group_df),
                "positive_rate": round(pos_rate, 4),
                "overall_rate": round(overall_rate, 4),
                "gap": round(gap, 4),
                "severity": round(severity, 4),
                "is_invisible_bias": len(attrs) > 1,  # Will be refined below
            })

        return results

    def find_invisible_bias(
        self,
        single_attr_results: List[Dict[str, Any]],
        intersectional_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Find cases where single-attribute analysis passes (DI > 0.8)
        but intersectional analysis reveals significant disparities.
        These are 'invisible bias' patterns.
        """
        # Build lookup of single-attribute subgroups that passed
        single_passing = set()
        for r in single_attr_results:
            if r.get("severity", 1.0) < 0.3:  # Low severity = passing
                # Extract the base attribute
                parts = r["subgroup"].split("=")
                if len(parts) >= 1:
                    single_passing.add(parts[0].strip())

        invisible = []
        for r in intersectional_results:
            if len(r.get("attributes", [])) <= 1:
                continue

            # Check if each individual attribute was passing
            attrs = r.get("attributes", [])
            all_attrs_passing = any(a in single_passing for a in attrs)

            # This is invisible bias if at least one attr was passing individually
            # but intersection reveals significant disparity
            if all_attrs_passing and r.get("severity", 0) >= 0.3:
                invisible_result = dict(r)
                invisible_result["is_invisible_bias"] = True
                invisible.append(invisible_result)

        return invisible
