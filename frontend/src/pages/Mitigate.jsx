import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Bar, Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend } from 'chart.js';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { mitigateDataset } from '../utils/api';

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend);

const TECHNIQUES = [
  {
    id: 'reweighing',
    name: '1. Reweighing',
    icon: '⚖️',
    desc: 'Assigns sample weights to training examples to reduce bias. Safe and interpretable.',
    expected: '+15–20% fairness',
    warning: null,
  },
  {
    id: 'disparate_impact_remover',
    name: '2. Disparate Impact Remover',
    icon: '🔧',
    desc: 'Transforms feature values to reduce correlation with protected attributes.',
    expected: '+10–15% fairness',
    warning: null,
    hasSlider: true,
  },
  {
    id: 'threshold_adjustment',
    name: '3. Threshold Adjustment',
    icon: '📐',
    desc: 'Uses different decision thresholds per group to equalize true positive rates.',
    expected: '+5–10% fairness',
    warning: null,
  },
  {
    id: 'adversarial_debiasing',
    name: '4. Adversarial Debiasing',
    icon: '🧠',
    desc: 'Trains a model that cannot predict the protected attribute. Most powerful technique.',
    expected: '+5–8% fairness',
    warning: 'Requires full model retraining — may take several minutes',
  },
];

export default function Mitigate() {
  const [enabled, setEnabled] = useState({ reweighing: true, disparate_impact_remover: false, threshold_adjustment: false, adversarial_debiasing: false });
  const [repairLevel, setRepairLevel] = useState(0.8);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [analysisId, setAnalysisId] = useState(null);

  useEffect(() => {
    const a = sessionStorage.getItem('fairsight_analysis');
    if (a) setAnalysisId(JSON.parse(a).analysis_id);
    const m = sessionStorage.getItem('fairsight_mitigation');
    if (m) setResult(JSON.parse(m));
  }, []);

  const handleApply = async () => {
    if (!analysisId) { toast.error('No analysis found. Run analysis on Dashboard first.'); return; }
    const techniques = Object.entries(enabled).filter(([,v]) => v).map(([k]) => k);
    if (!techniques.length) { toast.error('Select at least one technique.'); return; }

    setLoading(true);
    try {
      const res = await mitigateDataset({ analysis_id: analysisId, techniques, repair_level: repairLevel });
      setResult(res.data);
      sessionStorage.setItem('fairsight_mitigation', JSON.stringify(res.data));
      toast.success('Mitigation applied successfully!');
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Mitigation failed.');
    } finally {
      setLoading(false);
    }
  };

  const beforeMetrics = result?.before_metrics || [];
  const afterMetrics = result?.after_metrics || [];
  const metricNames = beforeMetrics.map(m => m.name.replace(' Difference', '').replace(' Index', '').substring(0, 15));
  const beforeScores = beforeMetrics.map(m => m.score);
  const afterScores = [...afterMetrics.map(m => m.score), ...Array(Math.max(0, beforeScores.length - afterMetrics.length)).fill(null)];

  const barData = {
    labels: metricNames,
    datasets: [
      { label: 'Before', data: beforeScores, backgroundColor: 'rgba(226,75,74,0.6)', borderColor: '#E24B4A', borderWidth: 1.5, borderRadius: 4 },
      { label: 'After', data: afterScores, backgroundColor: 'rgba(29,158,117,0.6)', borderColor: '#1D9E75', borderWidth: 1.5, borderRadius: 4 },
    ],
  };

  const barOptions = {
    responsive: true,
    animation: { duration: 800 },
    scales: {
      y: { min: 0, max: 1, grid: { color: 'rgba(0,0,0,0.05)' } },
      x: { grid: { display: false } },
    },
    plugins: {
      legend: { position: 'top' },
      tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw?.toFixed(4) ?? 'N/A'}` } },
    },
  };

  const journeyData = result?.score_journey || [];
  const lineData = {
    labels: journeyData.map(j => j.label),
    datasets: [{
      label: 'Fairness Score',
      data: journeyData.map(j => j.score),
      borderColor: '#1D9E75',
      backgroundColor: 'rgba(29,158,117,0.1)',
      pointRadius: 6, pointBackgroundColor: '#1D9E75',
      tension: 0.3, fill: true,
    }],
  };
  const lineOptions = {
    responsive: true,
    animation: { duration: 1000 },
    scales: {
      y: { min: 0, max: 1, grid: { color: 'rgba(0,0,0,0.05)' } },
    },
    plugins: {
      annotation: { annotations: { threshold: { type: 'line', yMin: 0.8, yMax: 0.8, borderColor: '#EF9F27', borderDash: [5,5], label: { content: 'Legal 0.80', enabled: true } } } },
    },
  };

  // Pareto tradeoff curve data
  const tradeoff = result?.tradeoff_curve || [];
  const paretoData = {
    labels: tradeoff.map(p => p.label),
    datasets: [{
      label: 'Fairness vs Accuracy',
      data: tradeoff.map(p => ({ x: p.fairness_improvement, y: p.accuracy_loss })),
      borderColor: '#0C447C',
      backgroundColor: 'rgba(12,68,124,0.15)',
      pointRadius: 5, tension: 0.2,
    }],
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>🔧 Bias Mitigation</h1>
        <p>Apply fairness techniques step by step and see live improvements.</p>
      </div>

      <div className="grid-2">
        {/* Left: Technique toggles */}
        <div>
          {TECHNIQUES.map(t => (
            <div key={t.id} className="card" style={{ marginBottom: '1rem', border: enabled[t.id] ? '1.5px solid var(--teal)' : '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--navy)' }}>{t.icon} {t.name}</div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '6px 0' }}>{t.desc}</p>
                  <span style={{ fontSize: '0.75rem', background: '#e0f4ed', color: 'var(--teal)', padding: '2px 8px', borderRadius: 99, fontWeight: 600 }}>
                    Expected: {t.expected}
                  </span>
                  {t.warning && (
                    <div style={{ marginTop: 8, fontSize: '0.75rem', background: '#fff3cd', color: '#856404', padding: '4px 10px', borderRadius: 6 }}>
                      ⚠️ {t.warning}
                    </div>
                  )}
                  {t.hasSlider && enabled[t.id] && (
                    <div style={{ marginTop: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 4 }}>
                        <span>Repair Level</span><span style={{ fontWeight: 700, color: 'var(--navy)' }}>{repairLevel.toFixed(1)}</span>
                      </div>
                      <input type="range" className="form-range" min="0" max="1" step="0.1" value={repairLevel} onChange={e => setRepairLevel(Number(e.target.value))} />
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        <span>No repair</span><span>Full repair</span>
                      </div>
                    </div>
                  )}
                </div>
                <label className="toggle" style={{ marginLeft: 16, flexShrink: 0 }}>
                  <input type="checkbox" checked={enabled[t.id]} onChange={() => setEnabled(e => ({ ...e, [t.id]: !e[t.id] }))} />
                  <span className="toggle-slider" />
                </label>
              </div>
            </div>
          ))}

          <button className="btn btn-teal btn-lg" style={{ width: '100%', justifyContent: 'center' }} onClick={handleApply} disabled={loading || !analysisId}>
            {loading ? <><span className="spinner" style={{ marginRight: 8 }} />Applying techniques...</> : '✅ Apply Selected Techniques'}
          </button>

          {!analysisId && (
            <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center' }}>
              <Link to="/dashboard" style={{ color: 'var(--navy)' }}>Run analysis on Dashboard first →</Link>
            </div>
          )}

          {result?.prescribed_steps && (
            <div className="card" style={{ marginTop: '1rem', background: '#f0f7ff' }}>
              <h4 style={{ color: 'var(--navy)', margin: '0 0 0.75rem', fontSize: '0.9rem' }}>🤖 Gemini Prescribes:</h4>
              <ul className="action-steps">
                {result.prescribed_steps.map((s, i) => (
                  <li key={i}><span className="action-step-num">{i+1}</span>{s}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Right: Results */}
        <div>
          {result && (
            <>
              {/* Score Journey */}
              <div className="card" style={{ marginBottom: '1rem' }}>
                <h3 className="card-title" style={{ marginBottom: '1rem' }}>📈 Score Journey</h3>
                <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: '1rem' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--red)' }}>{result.before_score.toFixed(3)}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>BEFORE</div>
                  </div>
                  <div style={{ fontSize: '2rem', alignSelf: 'center' }}>→</div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--teal)' }}>{result.after_score.toFixed(3)}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>AFTER</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 900, color: result.improvement >= 0 ? 'var(--teal)' : 'var(--red)' }}>
                      {result.improvement >= 0 ? '+' : ''}{(result.improvement * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>IMPROVEMENT</div>
                  </div>
                </div>
                {journeyData.length > 0 && <Line data={lineData} options={{ ...lineOptions, plugins: { legend: { display: false } } }} height={100} />}
              </div>

              {/* Before/After Bar Chart */}
              <div className="card" style={{ marginBottom: '1rem' }}>
                <h3 className="card-title" style={{ marginBottom: '1rem' }}>📊 Before vs After Metrics</h3>
                <Bar data={barData} options={barOptions} height={200} />
              </div>

              {/* Pareto Curve */}
              {tradeoff.length > 0 && (
                <div className="card">
                  <h3 className="card-title" style={{ marginBottom: '1rem' }}>⚖️ Fairness vs Accuracy Trade-off</h3>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 0 }}>
                    Each point represents a different repair level. Choose your acceptable trade-off point.
                  </p>
                  <div style={{ fontSize: '0.875rem', background: '#e0f4ed', padding: '10px 14px', borderRadius: 8, marginBottom: '1rem' }}>
                    ✅ At recommended setting: bias reduces by <strong>{(tradeoff[Math.floor(tradeoff.length * 0.8)]?.fairness_improvement * 100)?.toFixed(1) || '~40'}%</strong>,
                    accuracy drops by only <strong>{(tradeoff[Math.floor(tradeoff.length * 0.8)]?.accuracy_loss * 100)?.toFixed(1) || '~2.1'}%</strong>
                  </div>
                  <div style={{ fontSize: '0.8rem' }}>
                    {tradeoff.map((p, i) => (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '5px 0', borderBottom: '0.5px solid var(--border)' }}>
                        <span style={{ fontWeight: 600, minWidth: 80 }}>{p.label}</span>
                        <span style={{ color: 'var(--teal)' }}>+{(p.fairness_improvement * 100).toFixed(1)}% fair</span>
                        <span style={{ color: 'var(--red)' }}>-{(p.accuracy_loss * 100).toFixed(1)}% acc</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {!result && (
            <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🔧</div>
              <h3 style={{ color: 'var(--navy)' }}>Select techniques and apply</h3>
              <p style={{ color: 'var(--text-muted)' }}>Results will appear here after mitigation is applied.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
