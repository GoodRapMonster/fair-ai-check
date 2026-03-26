"""
FairSight Drift Tracker
Persists DI snapshots per analysis run and detects fairness degradation over time.
Updated to use SQLite via SQLAlchemy instead of JSON files.
"""
import math
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session
from models.db_models import DriftSnapshot

DRIFT_THRESHOLD = 0.05   # DI drop > 5% = degrading


class DriftTracker:
    """
    Records DI snapshots on each analysis run and compares against the
    most recent prior snapshot to surface fairness drift.
    """

    def record_snapshot(self, db: Session, analysis_id: str, di_per_attr: Dict[str, Optional[float]]) -> None:
        """
        Persist a DI snapshot to the database.
        di_per_attr: {protected_attr: disparate_impact_value}
        """
        try:
            now = datetime.utcnow()
            clean_di = {
                k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4))
                for k, v in di_per_attr.items()
            }
            
            snapshot = DriftSnapshot(
                id=f"{analysis_id}_{now.timestamp()}",
                analysis_id=analysis_id,
                timestamp=now,
                di_per_attr=clean_di
            )
            db.add(snapshot)
            db.commit()
        except Exception:
            db.rollback()  # Never crash the API because history couldn't be written

    def compare_with_previous(
        self, db: Session, current_di: Dict[str, Optional[float]]
    ) -> List[Dict[str, Any]]:
        """
        Compare current DI values against the most recent prior snapshot in the database.
        Returns a list of drift result dicts.
        """
        try:
            # Get the most recent snapshot overall 
            # (In a real app with projects, you'd filter by project_id)
            prev_snapshot = db.query(DriftSnapshot).order_by(DriftSnapshot.timestamp.desc()).first()
            if not prev_snapshot:
                return []
                
            prev_di: Dict[str, Any] = prev_snapshot.di_per_attr or {}
        except Exception:
            return []

        results = []
        for attr, cur_val in current_di.items():
            if cur_val is None:
                continue
            prev_val = prev_di.get(attr)
            if prev_val is None:
                continue

            try:
                cur_f  = float(cur_val)
                prev_f = float(prev_val)
            except (TypeError, ValueError):
                continue

            delta  = cur_f - prev_f
            alert  = delta < -DRIFT_THRESHOLD

            if delta > 0.01:
                trend = "improving ↑"
            elif delta < -0.01:
                trend = "degrading ↓"
            else:
                trend = "stable ─"

            results.append({
                "attr":        attr,
                "current_di":  round(cur_f,  4),
                "previous_di": round(prev_f, 4),
                "delta":       round(delta,  4),
                "trend":       trend,
                "alert":       alert,
            })

        return results
