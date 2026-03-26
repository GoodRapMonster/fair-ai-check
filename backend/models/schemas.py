from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class Domain(str, Enum):
    hiring = "hiring"
    lending = "lending"
    medical = "medical"
    criminal_justice = "criminal_justice"
    other = "other"


class RiskLevel(str, Enum):
    critical = "CRITICAL"
    high = "HIGH"
    medium = "MEDIUM"
    low = "LOW"
    passing = "PASS"


class Language(str, Enum):
    english = "English"
    hindi = "Hindi"
    spanish = "Spanish"
    french = "French"
    arabic = "Arabic"
    portuguese = "Portuguese"
    german = "German"
    mandarin = "Mandarin"


class ProtectedAttribute(BaseModel):
    column: str
    confidence: float
    detected_type: str


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: List[str]
    detected_protected_attrs: List[ProtectedAttribute]
    preview_rows: List[Dict[str, Any]]
    domain_suggestion: Optional[str] = None


class AnalyzeRequest(BaseModel):
    dataset_id: str
    outcome_col: str
    protected_attrs: List[str]
    domain: Domain
    privileged_groups: Dict[str, Any]


class FairnessMetric(BaseModel):
    name: str
    score: float
    threshold: float
    status: str  # PASS / FAIL / BORDERLINE
    description: str
    math_explanation: str
    
    # Statistical significance fields
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None


class ProxyResult(BaseModel):
    feature: str
    protected_attr: str
    correlation: float
    risk_level: str
    recommendation: str


class IntersectionalResult(BaseModel):
    subgroup: str
    size: int
    positive_rate: float
    overall_rate: float
    severity: float
    is_invisible_bias: bool


class FingerprintData(BaseModel):
    gender_bias: float
    race_bias: float
    age_bias: float
    proxy_risk: float
    intersectional: float
    provenance: float
    severity: float
    spread: float


class BiasScoreBreakdown(BaseModel):
    """Adaptive, domain-weighted bias score (0–100, higher = more biased)."""
    score: float
    verdict: str          # CLEAN / MINOR ISSUES / MODERATE BIAS / SIGNIFICANT BIAS / SEVERELY BIASED
    domain: str
    weights: Dict[str, float]          # renormalized per-component weights used
    severities: Dict[str, float]       # raw severity 0–100 per component
    contributions: Dict[str, float]    # weight × severity per component
    primary_driver: str                # component with biggest contribution
    quick_win: Optional[List[Any]] = None   # [contribution, component, score_after_fix]


class DriftResult(BaseModel):
    """Fairness drift result comparing DI against previous run."""
    attr: str
    current_di: float
    previous_di: float
    delta: float
    trend: str    # improving ↑ / degrading ↓ / stable ─
    alert: bool   # True when delta < -0.05


class AnalyzeResponse(BaseModel):
    analysis_id: str
    dataset_id: str
    bias_score: float
    risk_level: RiskLevel
    metrics: List[FairnessMetric]
    proxy_results: List[ProxyResult]
    intersectional_results: List[IntersectionalResult]
    fingerprint: FingerprintData
    shap_values: Dict[str, float]
    impact_estimate: Dict[str, Any]
    proxy_heatmap: Dict[str, Dict[str, float]]
    bias_score_breakdown: Optional[BiasScoreBreakdown] = None
    drift_results: List[DriftResult] = []


class NarrateRequest(BaseModel):
    analysis_id: str
    domain: Domain
    language: Language = Language.english
    organization_name: Optional[str] = "your organization"
    role: Optional[str] = "professional"


class NarrateResponse(BaseModel):
    executive_headline: str
    risk_badge: RiskLevel
    paragraph1: str
    paragraph2: str
    paragraph3: str
    action_steps: List[str]
    language: str


class StoryRequest(BaseModel):
    analysis_id: str
    protected_attr: str


class StoryProfile(BaseModel):
    name: str
    age: int
    education: str
    experience: int
    skills: List[str]
    protected_value: str
    protected_attr: str
    extra_fields: Dict[str, Any]


class StoryResponse(BaseModel):
    profile_a: StoryProfile
    profile_b: StoryProfile
    score_a: float
    score_b: float
    decision_a: str
    decision_b: str
    story_text: str
    difference_attr: str


class MitigationTechnique(str, Enum):
    reweighing = "reweighing"
    disparate_impact_remover = "disparate_impact_remover"
    threshold_adjustment = "threshold_adjustment"
    adversarial_debiasing = "adversarial_debiasing"
    fairlearn_threshold = "fairlearn_threshold"
    fairlearn_eg = "fairlearn_eg"


class MitigateRequest(BaseModel):
    analysis_id: str
    techniques: List[MitigationTechnique]
    repair_level: float = Field(default=0.8, ge=0.0, le=1.0)
    thresholds: Optional[Dict[str, float]] = None


class TradeoffPoint(BaseModel):
    fairness_improvement: float
    accuracy_loss: float
    setting: Dict[str, Any]
    label: str


class MitigateResponse(BaseModel):
    mitigation_id: str
    before_metrics: List[FairnessMetric]
    after_metrics: List[FairnessMetric]
    before_score: float
    after_score: float
    improvement: float
    tradeoff_curve: List[TradeoffPoint]
    score_journey: List[Dict[str, Any]]
    prescribed_steps: List[str]


class SimulateRequest(BaseModel):
    analysis_id: str
    group_representation: float = Field(default=0.5, ge=0.0, le=1.0)
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    privileged_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    unprivileged_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    dataset_size_multiplier: float = Field(default=1.0, ge=0.1, le=5.0)
    repair_level: float = Field(default=0.0, ge=0.0, le=1.0)


class SimulateResponse(BaseModel):
    simulated_bias_score: float
    simulated_metrics: List[FairnessMetric]
    group_outcome_rates: Dict[str, float]
    insight_text: str


class ReportRequest(BaseModel):
    analysis_id: str
    organization_name: str
    analyst_name: str
    domain: Domain
    mitigation_id: Optional[str] = None


class MonitorBatchItem(BaseModel):
    label: str
    timestamp: str


class MonitorResponse(BaseModel):
    time_series: List[Dict[str, Any]]
    drift_score: float
    alerts: List[Dict[str, Any]]
    recommendations: List[str]


class ImpactRequest(BaseModel):
    analysis_id: str
    decisions_per_year: int
    years_running: int


class ImpactResponse(BaseModel):
    affected_total: int
    affected_per_month: float
    bias_gap: float
    narrative: str
    city_comparison: str
