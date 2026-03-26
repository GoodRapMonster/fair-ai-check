import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from core.bias_engine import dynamic_min_group_size


class IntersectionalAnalyzer:
    """
    Detects bias that only appears at the intersection of multiple protected attributes.
    Identifies 'invisible bias' missed by single-attribute analysis.
    Improvements over v1: dynamic min group size, small_sample flag, worst_subgroup tracking.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        outcome: str,
        protected_attrs: List[str],
        privileged_groups: Dict[str, Any],
        max_depth: int = 2,
        min_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute outcome rates for all subgroup combinations up to max_depth.
        Returns ranked list of subgroup results with invisible bias flags.
        min_size defaults to dynamic_min_group_size(len(df)) if not provided.
        """
        if min_size is None:
            min_size = dynamic_min_group_size(len(df))

        overall_rate = float(df[outcome].mean())
        results: List[Dict[str, Any]] = []
        valid_attrs = [a for a in protected_attrs if a in df.columns]

        for depth in range(1, min(max_depth + 1, len(valid_attrs) + 1)):
            for attr_combo in combinations(valid_attrs, depth):
                subgroup_results = self._analyze_subgroup(
                    df, outcome, list(attr_combo), overall_rate, min_size
                )
                results.extend(subgroup_results)

        # Sort: non-small-sample first, then by deviation descending
        results.sort(
            key=lambda x: (x.get("small_sample", False), -abs(x["positive_rate"] - x["overall_rate"]))
        )
        return results

    def _analyze_subgroup(
        self,
        df: pd.DataFrame,
        outcome: str,
        attrs: List[str],
        overall_rate: float,
        min_size: int,
    ) -> List[Dict[str, Any]]:
        """Analyze outcome rates for a specific combination of attributes."""
        results = []
        try:
            groups = df.groupby(attrs)
        except Exception:
            return results

        for group_keys, group_df in groups:
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)

            subgroup_label = " & ".join(
                f"{attr}={val}" for attr, val in zip(attrs, group_keys)
            )
            n        = len(group_df)
            pos_rate = float(group_df[outcome].mean())
            gap      = pos_rate - overall_rate
            severity = float(min(abs(gap) * 2, 1.0))

            entry: Dict[str, Any] = {
                "subgroup":        subgroup_label,
                "attributes":      attrs,
                "size":            n,
                "positive_rate":   round(pos_rate,     4),
                "overall_rate":    round(overall_rate, 4),
                "gap":             round(gap,          4),
                "severity":        round(severity,     4),
                "is_invisible_bias": len(attrs) > 1,
            }

            if n < min_size:
                # Flag small sample but still include — don't silently drop (CLI improvement)
                entry["small_sample"] = True
            else:
                entry["small_sample"] = False

            results.append(entry)

        return results

    def find_invisible_bias(
        self,
        single_attr_results: List[Dict[str, Any]],
        intersectional_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find bias invisible at single-attribute level but visible at intersections."""
        single_passing = set()
        for r in single_attr_results:
            if r.get("severity", 1.0) < 0.3:
                parts = r["subgroup"].split("=")
                if len(parts) >= 1:
                    single_passing.add(parts[0].strip())

        invisible = []
        for r in intersectional_results:
            if r.get("small_sample"):
                continue
            if len(r.get("attributes", [])) <= 1:
                continue
            attrs = r.get("attributes", [])
            if any(a in single_passing for a in attrs) and r.get("severity", 0) >= 0.3:
                invisible_result = dict(r)
                invisible_result["is_invisible_bias"] = True
                invisible.append(invisible_result)

        return invisible

    def get_worst_subgroup(
        self, results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Return the subgroup with the largest deviation from overall rate."""
        non_small = [r for r in results if not r.get("small_sample")]
        if not non_small:
            # Fall back to small samples as last resort
            non_small = results
        if not non_small:
            return None
        return max(non_small, key=lambda r: abs(r["positive_rate"] - r["overall_rate"]))
