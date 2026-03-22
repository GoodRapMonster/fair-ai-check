import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Radar, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler,
  CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend,
} from 'chart.js';
import toast from 'react-hot-toast';
import { analyzeDataset, narrateAnalysis } from '../utils/api';

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function useCountUp(target, duration = 1500) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let start = null;
    const step = (timestamp) => {
      if (!start) start = timestamp;
      const progress = Math.min((timestamp - start) / duration, 1);
      setValue(parseFloat((progress * target).toFixed(3)));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration]);
  return value;
}

function getRiskClass(score) {
  if (score < 0.5) return 'critical';
  if (score < 0.65) return 'high';
  if (score < 0.8) return 'medium';
  if (score < 0.9) return 'low';
  return 'pass';
}

// ---- Section A: Bias Score Hero ----
function BiasScoreHero({ score, riskLevel, impactCount }) {
  const animScore = useCountUp(score);
  const riskClass = getRiskClass(score);

  return (
    <div className="score-hero">
      <div className="score-label">OVERALL FAIRNESS SCORE</div>
      <motion.div
        className={`score-number score-${riskClass}`}
        initial={{ scale: 0.5, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 100 }}
      >
        {animScore.toFixed(3)}
      </motion.div>

      <div style={{ position: 'relative', maxWidth: 400, margin: '0.5rem auto' }}>
        <div className="progress-bar-track" style={{ height: 12 }}>
          <motion.div
            className="progress-bar-fill"
            style={{ background: score < 0.6 ? 'var(--red)' : score < 0.8 ? 'var(--amber)' : 'var(--teal)' }}
            initial={{ width: 0 }}
            animate={{ width: `${score * 100}%` }}
            transition={{ duration: 1.5 }}
          />
        </div>
        <div style={{ position: 'absolute', left: '80%', top: -20, transform: 'translateX(-50%)' }}>
          <div style={{ fontSize: '0.65rem', color: 'var(--amber)', fontWeight: 700, whiteSpace: 'nowrap' }}>▼ 0.80 EEOC / EU AI Act</div>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', marginTop: '1rem', flexWrap: 'wrap' }}>
        <span className={`badge badge-${riskClass}`} style={{ fontSize: '0.85rem', padding: '6px 16px' }}>
          {riskLevel} RISK
        </span>
        {impactCount > 0 && (
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            ≈ <strong style={{ color: 'var(--red)' }}>{impactCount.toLocaleString()}</strong> people affected/year
          </span>
        )}
      </div>
    </div>
  );
}

// ---- Section B: Bias Fingerprint ----
function BiasFingerprint({ fingerprint }) {
  if (!fingerprint) return null;
  const labels = ['Gender', 'Race', 'Age', 'Proxy Risk', 'Intersectional', 'Provenance', 'Severity', 'Spread'];
  const values = [
    fingerprint.gender_bias, fingerprint.race_bias, fingerprint.age_bias,
    fingerprint.proxy_risk, fingerprint.intersectional, fingerprint.provenance,
    fingerprint.severity, fingerprint.spread,
  ];

  const data = {
    labels,
    datasets: [{
      label: 'Fairness Score',
      data: values,
      backgroundColor: 'rgba(29, 158, 117, 0.15)',
      borderColor: '#1D9E75',
      borderWidth: 2,
      pointBackgroundColor: values.map(v => v < 0.7 ? '#E24B4A' : '#1D9E75'),
      pointRadius: 5,
    }],
  };

  const options = {
    responsive: true,
    animation: { duration: 1500 },
    scales: {
      r: {
        min: 0, max: 1,
        ticks: { display: false, stepSize: 0.2 },
        grid: { color: 'rgba(0,0,0,0.07)' },
        pointLabels: { font: { size: 11, weight: '600', family: 'Inter' }, color: '#374151' },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => `${ctx.label}: ${ctx.raw.toFixed(3)} (${ctx.raw < 0.7 ? '⚠️ Below threshold' : '✓ OK'})`,
        },
      },
    },
  };

  return (
    <div style={{ maxWidth: 380, margin: '0 auto' }}>
      <Radar data={data} options={options} />
      <p style={{ textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8 }}>
        Outer edge = perfectly fair. Inner = biased. Shape is unique to this dataset.
      </p>
    </div>
  );
}

// ---- Section C: Metrics Table ----
function MetricsTable({ metrics }) {
  const [expanded, setExpanded] = useState(null);
  return (
    <table className="metric-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Score</th>
          <th>Threshold</th>
          <th>Status</th>
          <th>What it means</th>
        </tr>
      </thead>
      <tbody>
        {metrics.map((m, i) => (
          <React.Fragment key={m.name}>
            <tr style={{ cursor: 'pointer' }} onClick={() => setExpanded(expanded === i ? null : i)}>
              <td style={{ fontWeight: 600 }}>{m.name}</td>
              <td style={{ fontWeight: 700, color: m.score < m.threshold ? 'var(--red)' : 'var(--teal)', fontVariantNumeric: 'tabular-nums' }}>
                {m.score.toFixed(4)}
              </td>
              <td style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>≥ {m.threshold}</td>
              <td><span className={`badge badge-${m.status.toLowerCase()}`}>{m.status}</span></td>
              <td style={{ fontSize: '0.8rem', color: 'var(--text-muted)', maxWidth: 200 }}>
                {m.description.substring(0, 60)}{m.description.length > 60 ? '...' : ''} <span style={{ color: 'var(--navy)' }}>{expanded === i ? '▲' : '▼'}</span>
              </td>
            </tr>
            {expanded === i && (
              <tr className="metric-expand-row">
                <td colSpan={5}>
                  <strong>Full description:</strong> {m.description}<br />
                  <strong>Math:</strong> <code style={{ fontFamily: 'monospace', background: 'rgba(0,0,0,0.05)', padding: '1px 4px', borderRadius: 3 }}>{m.math_explanation}</code>
                </td>
              </tr>
            )}
          </React.Fragment>
        ))}
      </tbody>
    </table>
  );
}

// ---- Section D: Proxy Heatmap ----
function ProxyHeatmap({ heatmap, protectedAttrs }) {
  const [tooltip, setTooltip] = useState(null);
  if (!heatmap || Object.keys(heatmap).length === 0) return <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>No proxy data available.</p>;

  const features = Object.keys(heatmap).slice(0, 12);
  const attrs = protectedAttrs;

  const getColor = (val) => {
    if (val >= 0.8) return '#E24B4A';
    if (val >= 0.5) return '#EF9F27';
    if (val >= 0.3) return '#f5a623';
    if (val >= 0.1) return '#ffe0b2';
    return '#f5f5f0';
  };

  return (
    <div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'separate', borderSpacing: 3, fontSize: '0.78rem' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '4px 8px', color: 'var(--text-muted)', fontWeight: 600, minWidth: 120 }}>Feature</th>
              {attrs.map(a => <th key={a} style={{ padding: '4px 8px', color: 'var(--text-muted)', fontWeight: 600 }}>{a}</th>)}
            </tr>
          </thead>
          <tbody>
            {features.map(feat => (
              <tr key={feat}>
                <td style={{ padding: '4px 8px', fontWeight: 500, fontSize: '0.8rem' }}>{feat}</td>
                {attrs.map(attr => {
                  const val = heatmap[feat]?.[attr] ?? 0;
                  return (
                    <td key={attr} style={{ padding: 3 }}>
                      <div
                        className="heatmap-cell"
                        style={{ width: 70, height: 30, background: getColor(val), display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.72rem', fontWeight: 600, color: val > 0.4 ? 'white' : '#555' }}
                        onMouseEnter={() => setTooltip({ feat, attr, val })}
                        onMouseLeave={() => setTooltip(null)}
                      >
                        {val > 0.05 ? (val * 100).toFixed(0) + '%' : '—'}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {tooltip && (
        <div style={{ marginTop: 12, padding: '10px 14px', background: '#fff7f7', border: '1px solid #fca5a5', borderRadius: 8, fontSize: '0.85rem' }}>
          <strong>'{tooltip.feat}'</strong> is <strong>{(tooltip.val * 100).toFixed(0)}%</strong> correlated with <strong>'{tooltip.attr}'</strong>
          {tooltip.val >= 0.7
            ? ' — it is acting as a hidden proxy. Recommend: remove or apply Disparate Impact Remover.'
            : tooltip.val >= 0.4
            ? ' — moderate proxy risk. Consider transforming this feature.'
            : ' — low proxy risk. Monitor during training.'}
        </div>
      )}
      <div style={{ display: 'flex', gap: 16, marginTop: 12, fontSize: '0.75rem', flexWrap: 'wrap' }}>
        {[['#ffe0b2','Low risk (10–30%)'], ['#f5a623','Medium risk (30–50%)'], ['#EF9F27','High risk (50–80%)'], ['#E24B4A','Proxy confirmed (>80%)']].map(([c,l]) => (
          <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 14, height: 14, background: c, borderRadius: 3 }} />
            <span style={{ color: 'var(--text-muted)' }}>{l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---- Section E: Intersectional Table ----
function IntersectionalTable({ results }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="metric-table">
        <thead>
          <tr>
            <th>Subgroup</th>
            <th style={{ textAlign: 'center' }}>Size</th>
            <th style={{ textAlign: 'center' }}>Positive Rate</th>
            <th style={{ textAlign: 'center' }}>vs Overall</th>
            <th style={{ textAlign: 'center' }}>Severity</th>
            <th>Flag</th>
          </tr>
        </thead>
        <tbody>
          {results.slice(0, 15).map((r, i) => (
            <tr key={i} style={{ background: r.is_invisible_bias ? '#fff8f0' : 'white' }}>
              <td style={{ fontWeight: r.is_invisible_bias ? 700 : 400, fontSize: '0.85rem' }}>{r.subgroup}</td>
              <td style={{ textAlign: 'center', fontSize: '0.85rem' }}>{r.size}</td>
              <td style={{ textAlign: 'center', fontWeight: 600, color: r.positive_rate < r.overall_rate * 0.8 ? 'var(--red)' : 'var(--text)' }}>
                {(r.positive_rate * 100).toFixed(1)}%
              </td>
              <td style={{ textAlign: 'center', fontSize: '0.85rem', color: r.gap < -0.05 ? 'var(--red)' : 'var(--text-muted)' }}>
                {r.gap >= 0 ? '+' : ''}{(r.gap * 100).toFixed(1)}%
              </td>
              <td style={{ textAlign: 'center' }}>
                <div style={{ width: 60, height: 8, background: 'var(--bg-gray)', borderRadius: 99, margin: '0 auto', overflow: 'hidden' }}>
                  <div style={{ width: `${r.severity * 100}%`, height: '100%', background: r.severity > 0.6 ? 'var(--red)' : r.severity > 0.3 ? 'var(--amber)' : 'var(--teal)', borderRadius: 99 }} />
                </div>
              </td>
              <td>
                {r.is_invisible_bias && (
                  <span style={{ fontSize: '0.65rem', background: '#fff3cd', color: '#856404', padding: '2px 6px', borderRadius: 99, fontWeight: 700, whiteSpace: 'nowrap' }}>
                    ⚠️ INVISIBLE BIAS
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---- Section F: Narration Card ----
const LANGUAGES = ['English', 'Hindi', 'Spanish', 'French', 'Arabic', 'Portuguese', 'German', 'Mandarin'];

function NarrationCard({ analysisId, domain, biasScore }) {
  const [narration, setNarration] = useState(null);
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('English');

  const fetchNarration = async (lang) => {
    setLoading(true);
    try {
      const res = await narrateAnalysis({ analysis_id: analysisId, domain, language: lang });
      setNarration(res.data);
    } catch (e) {
      toast.error('Narration failed. Check your GEMINI_API_KEY.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { if (analysisId) fetchNarration(language); }, [analysisId]);

  const handleLang = (lang) => {
    setLanguage(lang);
    fetchNarration(lang);
  };

  if (!analysisId) return null;

  return (
    <div className="narration-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '1rem' }}>
        <div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>✨ AI NARRATION (via Gemini)</div>
          {narration && <span className={`badge badge-${narration.risk_badge?.toLowerCase()}`}>{narration.risk_badge} RISK</span>}
        </div>
        <div className="lang-selector">
          {LANGUAGES.map(l => (
            <button key={l} className={`lang-btn ${language === l ? 'active' : ''}`} onClick={() => handleLang(l)}>{l}</button>
          ))}
        </div>
      </div>

      {loading && (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <div className="spinner" style={{ borderColor: 'rgba(12,68,124,0.2)', borderTopColor: 'var(--navy)', margin: '0 auto 1rem' }} />
          Generating {language} narration with Gemini...
        </div>
      )}

      {narration && !loading && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <p className="narration-headline">📋 {narration.executive_headline}</p>
          <p className="narration-p"><strong>What was found:</strong> {narration.paragraph1}</p>
          <p className="narration-p"><strong>Root causes:</strong> {narration.paragraph2}</p>
          <p className="narration-p"><strong>What to do:</strong> {narration.paragraph3}</p>
          {narration.action_steps?.length > 0 && (
            <ul className="action-steps">
              {narration.action_steps.map((step, i) => (
                <li key={i}><span className="action-step-num">{i+1}</span>{step}</li>
              ))}
            </ul>
          )}
        </motion.div>
      )}

      <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border)' }}>
        <Link to="/story" className="btn btn-teal btn-sm">📖 Go to Story Mode →</Link>
      </div>
    </div>
  );
}

// ---- MAIN DASHBOARD ----
export default function Dashboard() {
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scaleInput, setScaleInput] = useState(10000);

  useEffect(() => {
    const config = sessionStorage.getItem('fairsight_config');
    if (!config) {
      // Load demo data
      loadDemoData();
      return;
    }
    runAnalysis(JSON.parse(config));
  }, []);

  const loadDemoData = async () => {
    // Try to load synthetic dataset as demo
    try {
      const { loadBuiltinDataset } = await import('../utils/api');
      const upload = await loadBuiltinDataset('synthetic_hiring');
      const config = {
        dataset_id: upload.data.dataset_id,
        outcome_col: 'hired',
        protected_attrs: ['gender'],
        domain: 'hiring',
        privileged_groups: { gender: 1 },
      };
      sessionStorage.setItem('fairsight_config', JSON.stringify(config));
      await runAnalysis(config);
    } catch (e) {
      setError('No dataset found. Please upload a dataset first.');
    }
  };

  const runAnalysis = async (config) => {
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeDataset(config);
      setAnalysis(res.data);
      // Store analysis_id for downstream pages
      sessionStorage.setItem('fairsight_analysis', JSON.stringify(res.data));
    } catch (e) {
      setError(e.response?.data?.detail || 'Analysis failed. Check that the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="page-header"><h1>Dashboard</h1><p>Running bias analysis...</p></div>
        {[1,2,3,4].map(i => (
          <div key={i} className="card" style={{ marginBottom: '1rem' }}>
            <div className="skeleton skeleton-title" />
            <div className="skeleton skeleton-text" style={{ width: '80%' }} />
            <div className="skeleton skeleton-chart" />
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="page-header"><h1>Dashboard</h1></div>
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚠️</div>
          <h3 style={{ color: 'var(--red)' }}>{error}</h3>
          <Link to="/upload" className="btn btn-primary" style={{ marginTop: '1rem' }}>← Upload Dataset</Link>
        </div>
      </div>
    );
  }

  if (!analysis) return null;

  const impactCount = analysis.impact_estimate
    ? Math.round((analysis.impact_estimate.bias_gap || 0.2) * scaleInput)
    : 0;

  return (
    <div className="page-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div className="page-header" style={{ margin: 0 }}>
          <h1 style={{ margin: 0 }}>Bias Analysis Dashboard</h1>
          <p style={{ margin: 0 }}>Analysis ID: {analysis.analysis_id?.substring(0, 8)}...</p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <Link to="/mitigate" className="btn btn-teal btn-sm">🔧 Mitigate Bias</Link>
          <Link to="/report" className="btn btn-outline btn-sm">📄 Generate Report</Link>
        </div>
      </div>

      {/* SECTION A: Bias Score Hero */}
      <div className="card">
        <BiasScoreHero score={analysis.bias_score} riskLevel={analysis.risk_level} impactCount={impactCount} />
        <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Decisions/year:</span>
            <input type="number" value={scaleInput} onChange={e => setScaleInput(Number(e.target.value))}
              style={{ width: 100, padding: '4px 8px', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.85rem' }} />
          </div>
        </div>
      </div>

      {/* SECTION B: Bias Fingerprint */}
      <div className="card" style={{ marginTop: '1.25rem' }}>
        <div className="card-header">
          <div>
            <h3 className="card-title">🕸️ Bias Fingerprint</h3>
            <p className="card-subtitle">8-dimensional radar chart — unique to this dataset</p>
          </div>
        </div>
        <BiasFingerprint fingerprint={analysis.fingerprint} />
      </div>

      {/* SECTION C: Metrics Table */}
      <div className="card" style={{ marginTop: '1.25rem' }}>
        <div className="card-header">
          <h3 className="card-title">📊 All Fairness Metrics</h3>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {analysis.metrics.filter(m => m.status === 'PASS').length}/{analysis.metrics.length} passing · Click row to expand
          </span>
        </div>
        <MetricsTable metrics={analysis.metrics} />
      </div>

      {/* SECTION D: Proxy Heatmap */}
      <div className="card" style={{ marginTop: '1.25rem' }}>
        <div className="card-header">
          <div>
            <h3 className="card-title">🔍 Proxy Danger Map</h3>
            <p className="card-subtitle">Feature correlation with protected attributes — click a cell for details</p>
          </div>
        </div>
        <ProxyHeatmap heatmap={analysis.proxy_heatmap} protectedAttrs={JSON.parse(sessionStorage.getItem('fairsight_config') || '{}').protected_attrs || []} />
        {analysis.proxy_results?.slice(0, 3).map((p, i) => (
          <div key={i} style={{ marginTop: '0.5rem', fontSize: '0.8rem', padding: '6px 10px', background: p.correlation >= 0.7 ? '#fde8e8' : '#fef3e2', borderRadius: 6 }}>
            {p.recommendation}
          </div>
        ))}
      </div>

      {/* SECTION E: Intersectional */}
      <div className="card" style={{ marginTop: '1.25rem' }}>
        <div className="card-header">
          <div>
            <h3 className="card-title">⚡ Intersectional Bias</h3>
            <p className="card-subtitle">Bias that only appears at subgroup intersections — missed by standard tools</p>
          </div>
          <span style={{ fontSize: '0.75rem', background: '#fff3cd', color: '#856404', padding: '3px 8px', borderRadius: 99, fontWeight: 600 }}>
            {analysis.intersectional_results?.filter(r => r.is_invisible_bias).length} invisible bias patterns
          </span>
        </div>
        <IntersectionalTable results={analysis.intersectional_results || []} />
      </div>

      {/* SECTION F: AI Narration */}
      <div className="card" style={{ marginTop: '1.25rem' }}>
        <div className="card-header">
          <h3 className="card-title">🤖 AI Narration</h3>
        </div>
        <NarrationCard
          analysisId={analysis.analysis_id}
          domain={JSON.parse(sessionStorage.getItem('fairsight_config') || '{}').domain || 'hiring'}
          biasScore={analysis.bias_score}
        />
      </div>

      {/* SHAP Feature Importance */}
      {analysis.shap_values && Object.keys(analysis.shap_values).length > 0 && (
        <div className="card" style={{ marginTop: '1.25rem' }}>
          <div className="card-header">
            <h3 className="card-title">🔬 Feature Importance (SHAP)</h3>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(analysis.shap_values).slice(0, 8).map(([feat, val]) => (
              <div key={feat}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 3 }}>
                  <span style={{ fontWeight: 600 }}>{feat}</span>
                  <span style={{ color: 'var(--text-muted)' }}>{(val * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bar-track">
                  <motion.div className="progress-bar-fill" style={{ background: 'var(--navy)' }}
                    initial={{ width: 0 }} animate={{ width: `${val * 100}%` }} transition={{ duration: 1, delay: 0.1 }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
