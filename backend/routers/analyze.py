import uuid
import json
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from sklearn.preprocessing import LabelEncoder

from models.schemas import (
    AnalyzeRequest, AnalyzeResponse, FairnessMetric, ProxyResult,
    IntersectionalResult, FingerprintData, RiskLevel,
    BiasScoreBreakdown, DriftResult,
)
from core.bias_engine import BiasEngine, dynamic_min_group_size
from core.proxy_detector import ProxyDetector
from core.intersectional import IntersectionalAnalyzer
from core.shap_explainer import SHAPExplainer
from core.mitigation import MitigationEngine
from core.drift_tracker import DriftTracker
from routers.upload import get_dataset
from core.database import get_db
from sqlalchemy.orm import Session
from models.db_models import AnalysisRecord, User
from routers.auth import get_current_user

router = APIRouter()

# ── In-memory stores ─────────────────────────────────────────────────────────
_analysis_store: dict = {}
_model_store:    dict = {}

# ── Singletons ────────────────────────────────────────────────────────────────
bias_engine           = BiasEngine()
proxy_detector        = ProxyDetector()
intersectional_analyzer = IntersectionalAnalyzer()
shap_explainer        = SHAPExplainer()
mitigation_engine     = MitigationEngine()
drift_tracker         = DriftTracker()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_dataset(
    request: AnalyzeRequest, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run full bias analysis on uploaded dataset."""
    df = get_dataset(request.dataset_id)
    analysis_id = str(uuid.uuid4())

    outcome          = request.outcome_col
    protected_attrs  = request.protected_attrs
    privileged_groups = request.privileged_groups
    domain           = request.domain.value

    # ── Validate columns ─────────────────────────────────────────────────────
    if outcome not in df.columns:
        raise HTTPException(status_code=400, detail=f"Outcome column '{outcome}' not found.")
    for attr in protected_attrs:
        if attr not in df.columns:
            raise HTTPException(status_code=400, detail=f"Protected attribute '{attr}' not found.")

    primary_attr     = protected_attrs[0]
    privileged_value = privileged_groups.get(primary_attr, df[primary_attr].mode()[0])

    # ── Prepare working copy ──────────────────────────────────────────────────
    df_work = df.copy()
    if df_work[outcome].dtype == object:
        le = LabelEncoder()
        df_work[outcome] = le.fit_transform(df_work[outcome].astype(str))

    # ── Dynamic min group size (CLI FIX-03) ───────────────────────────────────
    min_size = dynamic_min_group_size(len(df_work))

    # ── Train baseline model for prediction-based metrics ────────────────────
    model, scaler, feature_cols = mitigation_engine.train_baseline_model(
        df_work, outcome, protected_attrs
    )
    predictions_col = None
    if model is not None and scaler is not None and feature_cols:
        try:
            X = df_work[feature_cols].fillna(0)
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)
            df_work["_predictions"] = preds
            predictions_col = "_predictions"
        except Exception:
            pass

    # ── Standard fairness metrics (w/ stats for DI/SPD) ──────────────────────
    di_out = bias_engine.compute_disparate_impact(df_work, outcome, primary_attr, privileged_value, compute_stats=True)
    if isinstance(di_out, tuple):
        di, di_low, di_high, di_p = di_out
    else:
        di, di_low, di_high, di_p = di_out, None, None, None
        
    spd_out = bias_engine.compute_statistical_parity_diff(df_work, outcome, primary_attr, privileged_value, compute_stats=True)
    if isinstance(spd_out, tuple):
        spd, spd_low, spd_high, spd_p = spd_out
    else:
        spd, spd_low, spd_high, spd_p = spd_out, None, None, None
        
    raw_dp_gap = bias_engine.compute_raw_dp_gap(df_work, outcome, primary_attr, privileged_value)

    eod = (bias_engine.compute_equal_opportunity_diff(
               df_work, outcome, predictions_col or outcome, primary_attr, privileged_value)
           if predictions_col else spd)
    aod = (bias_engine.compute_average_odds_diff(
               df_work, outcome, predictions_col or outcome, primary_attr, privileged_value)
           if predictions_col else spd)
    ti    = bias_engine.compute_theil_index(df_work, outcome, primary_attr)
    ind_f = bias_engine.compute_individual_fairness(df_work, outcome, primary_attr)
    cal   = (bias_engine.compute_calibration(df_work, outcome, predictions_col, primary_attr)
             if predictions_col else 0.9)

    raw_metrics = {
        "disparate_impact":    di,
        "statistical_parity":  spd,
        "equal_opportunity":   eod,
        "average_odds":        aod,
        "theil_index":         ti,
        "individual_fairness": ind_f,
        "calibration":         cal,
    }

    def make_metric(name, key, score, threshold, description, math_exp, low=None, high=None, p=None):
        status = "PASS" if score >= threshold else ("BORDERLINE" if score >= threshold - 0.1 else "FAIL")
        is_sig = (p < 0.05) if p is not None else None
        return FairnessMetric(
            name=name, score=round(score, 4), threshold=threshold,
            status=status, description=description, math_explanation=math_exp,
            ci_lower=low, ci_upper=high, p_value=p, is_significant=is_sig
        )

    dp_gap_str = f" (raw gap: {raw_dp_gap:+.4f})" if raw_dp_gap is not None else ""
    metrics = [
        make_metric("Disparate Impact", "disparate_impact", di, 0.80,
            "Ratio of positive outcome rates between unprivileged and privileged groups. EEOC 4/5ths rule requires ≥0.8.",
            "DI = P(Ŷ=1|A=unprivileged) / P(Ŷ=1|A=privileged)", di_low, di_high, di_p),
        make_metric("Statistical Parity Difference", "statistical_parity", spd, 0.85,
            f"Difference in positive outcome rates. Zero means both groups receive same rate.{dp_gap_str}",
            "SPD = P(Ŷ=1|A=unprivileged) − P(Ŷ=1|A=privileged), normalized to 0–1", spd_low, spd_high, spd_p),
        make_metric("Equal Opportunity Difference", "equal_opportunity", eod, 0.85,
            "Difference in true positive rates (benefit given to actually qualified individuals).",
            "EOD = TPR(unprivileged) − TPR(privileged), normalized"),
        make_metric("Average Odds Difference", "average_odds", aod, 0.85,
            "Average of TPR and FPR differences across groups.",
            "AOD = 0.5 × [(FPR_unpriv − FPR_priv) + (TPR_unpriv − TPR_priv)], normalized"),
        make_metric("Theil Index", "theil_index", ti, 0.85,
            "Inequality measure of benefit distribution across groups.",
            "T = −(1/n) Σ log(rᵢ/μ) where rᵢ = group rate, normalized to 0–1"),
        make_metric("Individual Fairness", "individual_fairness", ind_f, 0.80,
            "Similar individuals should receive similar outcomes (KNN consistency).",
            "IF = mean consistency across K-nearest neighbors"),
        make_metric("Calibration", "calibration", cal, 0.85,
            "Predicted probabilities match actual outcome rates equally across groups.",
            "Cal = 1 − |P(Y=1|group) − P(Ŷ=1|group)| gap across groups"),
    ]

    bias_score = bias_engine.compute_bias_score(raw_metrics)
    risk_level = bias_engine.get_risk_level(bias_score)

    # ── Proxy detection (improved point-biserial + dedup) ─────────────────────
    proxy_results_raw = proxy_detector.detect_proxies(df_work, protected_attrs)
    proxy_results = [
        ProxyResult(
            feature=p["feature"],
            protected_attr=p["protected_attr"],
            correlation=p["correlation"],
            risk_level=p["risk_level"],
            recommendation=p["recommendation"],
        )
        for p in proxy_results_raw[:20]
    ]

    all_features = [
        c for c in df_work.columns
        if c not in protected_attrs + [outcome, "_predictions", "_weight"]
    ]
    proxy_heatmap = proxy_detector.build_heatmap(
        proxy_results_raw, all_features[:15], protected_attrs
    )

    # ── Intersectional analysis (dynamic min_size, small_sample flags) ────────
    intersectional_raw = intersectional_analyzer.analyze(
        df_work, outcome, protected_attrs, privileged_groups, min_size=min_size
    )
    intersectional_results = [
        IntersectionalResult(
            subgroup=r["subgroup"],
            size=r["size"],
            positive_rate=r["positive_rate"],
            overall_rate=r["overall_rate"],
            severity=r["severity"],
            is_invisible_bias=r.get("is_invisible_bias", False),
        )
        for r in intersectional_raw[:20]
    ]

    # ── Bias fingerprint ──────────────────────────────────────────────────────
    fingerprint_data = bias_engine.compute_fingerprint(
        df_work, outcome, raw_metrics,
        proxy_results_raw,
        [r.__dict__ for r in intersectional_results],
        {a: privileged_groups.get(a) for a in protected_attrs},
    )

    # ── SHAP values ───────────────────────────────────────────────────────────
    shap_values = {}
    if model is not None and scaler is not None and feature_cols:
        try:
            X = df_work[feature_cols].fillna(0)
            shap_values = shap_explainer.compute_shap_values(model, X)
        except Exception:
            pass

    # ── Impact estimate ───────────────────────────────────────────────────────
    priv_rate   = df_work[df_work[primary_attr] == privileged_value][outcome].mean()
    unpriv_rate = df_work[df_work[primary_attr] != privileged_value][outcome].mean()
    bias_gap    = float(priv_rate - unpriv_rate)
    impact      = bias_engine.compute_impact(bias_gap, scale=10000, years=1)

    # ── Adaptive bias score breakdown (domain-weighted, CLI-ported) ───────────
    breakdown_raw = bias_engine.compute_bias_score_breakdown(
        raw_metrics,
        proxy_results_raw,
        [r.__dict__ for r in intersectional_results],
        df_work,
        protected_attrs,
        domain=domain,
        predictions_col=predictions_col,
        outcome_col=outcome,
        privileged_value=privileged_value
    )
    bias_score_breakdown = BiasScoreBreakdown(
        score=breakdown_raw["score"],
        verdict=breakdown_raw["verdict"],
        domain=breakdown_raw["domain"],
        weights=breakdown_raw["weights"],
        severities=breakdown_raw["severities"],
        contributions=breakdown_raw["contributions"],
        primary_driver=breakdown_raw["primary_driver"],
        quick_win=breakdown_raw.get("quick_win"),
    )

    # ── Fairness drift (compare DI against previous run) ─────────────────────
    current_di_map = {primary_attr: di}
    # Also include per-attribute DI for multi-attr analyses
    for attr in protected_attrs[1:]:
        priv_val = privileged_groups.get(attr, df_work[attr].mode()[0])
        current_di_map[attr] = bias_engine.compute_disparate_impact(
            df_work, outcome, attr, priv_val
        )

    raw_drift       = drift_tracker.compare_with_previous(db, current_di_map)
    drift_tracker.record_snapshot(db, analysis_id, current_di_map)

    drift_results = [
        DriftResult(
            attr=d["attr"],
            current_di=d["current_di"],
            previous_di=d["previous_di"],
            delta=d["delta"],
            trend=d["trend"],
            alert=d["alert"],
        )
        for d in raw_drift
    ]

    # ── Persist for downstream use (mitigation, report, story) ───────────────
    analysis_data = {
        "analysis_id":            analysis_id,
        "dataset_id":             request.dataset_id,
        "outcome":                outcome,
        "protected_attrs":        protected_attrs,
        "privileged_groups":      privileged_groups,
        "domain":                 domain,
        "bias_score":             bias_score,
        "risk_level":             risk_level,
        "metrics":                [m.model_dump() for m in metrics],
        "proxy_results":          [p.model_dump() for p in proxy_results],
        "intersectional_results": [i.model_dump() for i in intersectional_results],
        "fingerprint":            fingerprint_data,
        "shap_values":            shap_values,
        "impact_estimate":        impact,
        "proxy_heatmap":          proxy_heatmap,
        "bias_score_breakdown":   breakdown_raw,
        "drift_results":          [d.model_dump() for d in drift_results],
    }
    _analysis_store[analysis_id] = analysis_data
    if model is not None:
        _model_store[analysis_id] = (model, scaler, feature_cols)

    # ── Database Persistence ─────────────────────────────────────────────────
    try:
        db_record = AnalysisRecord(
            analysis_id=analysis_id,
            dataset_id=request.dataset_id,
            outcome=outcome,
            domain=domain,
            bias_score=bias_score,
            risk_level=risk_level,
            full_result=analysis_data
        )
        db.add(db_record)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Failed to save to DB: {e}")

    return AnalyzeResponse(
        analysis_id=analysis_id,
        dataset_id=request.dataset_id,
        bias_score=round(bias_score, 4),
        risk_level=RiskLevel(risk_level),
        metrics=metrics,
        proxy_results=proxy_results,
        intersectional_results=intersectional_results,
        fingerprint=FingerprintData(**fingerprint_data),
        shap_values=shap_values,
        impact_estimate=impact,
        proxy_heatmap=proxy_heatmap,
        bias_score_breakdown=bias_score_breakdown,
        drift_results=drift_results,
    )

def get_analysis(analysis_id: str) -> dict:
    # First check memory (for recent runs needing model objects)
    if analysis_id in _analysis_store:
        return _analysis_store[analysis_id]
        
    # Then check DB (fallback for older runs/server reloads)
    try:
        from core.database import SessionLocal
        db = SessionLocal()
        record = db.query(AnalysisRecord).filter(AnalysisRecord.analysis_id == analysis_id).first()
        db.close()
        if record and record.full_result:
            return record.full_result
    except Exception:
        pass
        
    raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found.")


def get_model(analysis_id: str):
    return _model_store.get(analysis_id, (None, None, []))


# ── History Endpoints ────────────────────────────────────────────────────────

@router.get("/analyses")
async def list_analyses(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get list of past analyses."""
    records = db.query(AnalysisRecord).order_by(AnalysisRecord.timestamp.desc()).limit(50).all()
    return [{
        "analysis_id": r.analysis_id,
        "dataset_id": r.dataset_id,
        "timestamp": r.timestamp,
        "outcome": r.outcome,
        "domain": r.domain,
        "bias_score": r.bias_score,
        "risk_level": r.risk_level
    } for r in records]

@router.get("/analyses/{analysis_id}", response_model=AnalyzeResponse)
async def get_analysis_by_id(analysis_id: str, db: Session = Depends(get_db)):
    """Load a past analysis full result by ID."""
    record = db.query(AnalysisRecord).filter(AnalysisRecord.analysis_id == analysis_id).first()
    if not record or not record.full_result:
        raise HTTPException(status_code=404, detail="Analysis not found in DB.")
    return AnalyzeResponse(**record.full_result)
