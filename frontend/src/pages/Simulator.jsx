import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import toast from 'react-hot-toast';
import { simulateBias, calculateImpact } from '../utils/api';

export default function Simulator() {
  const [analysis, setAnalysis] = useState(null);
  const [representation, setRepresentation] = useState(50); // % of privileged group
  const [repairLevel, setRepairLevel] = useState(0); // 0 to 1
  const [simResult, setSimResult] = useState(null);
  const [impactResult, setImpactResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scaleFactor, setScaleFactor] = useState(10000); // population size

  useEffect(() => {
    const data = sessionStorage.getItem('fairsight_analysis');
    if (data) setAnalysis(JSON.parse(data));
  }, []);

  useEffect(() => {
    if (analysis) runSimulation();
  }, [representation, repairLevel, analysis]); // Debouncing might be better in prod, but fine for demo

  // Debounce helper
  const [timer, setTimer] = useState(null);
  const runSimulation = () => {
    if (timer) clearTimeout(timer);
    setTimer(setTimeout(async () => {
      setLoading(true);
      try {
        const [sim, impact] = await Promise.all([
          simulateBias({
            analysis_id: analysis.analysis_id,
            group_representations: { privileged: representation / 100, unprivileged: 1 - (representation / 100) },
            repair_level: repairLevel
          }),
          calculateImpact({
            analysis_id: analysis.analysis_id,
            population_scale: scaleFactor,
            repair_level: repairLevel
          })
        ]);
        setSimResult(sim.data);
        setImpactResult(impact.data);
      } catch (e) {
        // Silent fail for smooth dragging
      } finally {
        setLoading(false);
      }
    }, 300));
  };

  useEffect(() => {
    if (analysis) runSimulation();
  }, [scaleFactor]);

  if (!analysis) {
    return (
      <div className="page-container" style={{ textAlign: 'center', paddingTop: '4rem' }}>
        <h3 style={{ color: 'var(--navy)' }}>No data to simulate</h3>
        <p style={{ color: 'var(--text-muted)' }}>Run an analysis on the Dashboard first.</p>
      </div>
    );
  }

  const score = simResult ? simResult.simulated_bias_score : analysis.bias_score;
  const metrics = simResult ? simResult.simulated_metrics : analysis.metrics;

  const chartData = {
    labels: metrics?.map(m => m.name.split(' ')[0]) || [],
    datasets: [{
      label: 'Fairness (1.0 is Perfect)',
      data: metrics?.map(m => Math.max(0, 1 - Math.abs(m.score > 1 ? 1/m.score : m.score - 1))) || [],
      borderColor: score < 0.8 ? '#E24B4A' : '#1D9E75',
      backgroundColor: score < 0.8 ? 'rgba(226,75,74,0.1)' : 'rgba(29,158,117,0.1)',
      fill: true,
      tension: 0.4
    }]
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>🎛️ Live Policy Simulator</h1>
        <p>Drag the sliders to see how hiring rules and demographic shifts change bias in real time.</p>
      </div>

      <div className="simulator-panel">
        {/* Sliders */}
        <div className="card">
          <h3 className="card-title" style={{ marginBottom: '1.5rem' }}>Control Panel</h3>

          <div className="slider-group">
            <div className="slider-header">
              <span className="slider-label">Demographic Split (Privileged %)</span>
              <span className="slider-value">{representation}%</span>
            </div>
            <input type="range" className="form-range" min="10" max="90" value={representation} onChange={e => setRepresentation(Number(e.target.value))} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
              <span>10% Advantaged</span><span>90% Advantaged</span>
            </div>
          </div>

          <div className="slider-group" style={{ marginTop: '2rem' }}>
            <div className="slider-header">
              <span className="slider-label">Algorithm Repair Level</span>
              <span className="slider-value">{Math.round(repairLevel * 100)}%</span>
            </div>
            <input type="range" className="form-range" min="0" max="1" step="0.05" value={repairLevel} onChange={e => setRepairLevel(Number(e.target.value))} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
              <span>Biased (Original)</span><span>Fully Repaired</span>
            </div>
            {repairLevel > 0.8 && <div style={{ fontSize: '0.75rem', color: 'var(--amber)', marginTop: 8 }}>⚠️ High repair may reduce baseline accuracy.</div>}
          </div>

          <div className="slider-group" style={{ marginTop: '2rem' }}>
            <div className="slider-header">
              <span className="slider-label">Annual Application Volume</span>
              <span className="slider-value">{scaleFactor.toLocaleString()}</span>
            </div>
            <input type="range" className="form-range" min="1000" max="100000" step="1000" value={scaleFactor} onChange={e => setScaleFactor(Number(e.target.value))} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
              <span>Small Co</span><span>Enterprise</span>
            </div>
          </div>
        </div>

        {/* Live Output */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Big Score */}
          <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <h3 className="card-title">Simulated Fairness Score</h3>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Legal threshold: 0.80</div>
            </div>
            <div style={{ position: 'relative' }}>
              <div style={{ fontSize: '3.5rem', fontWeight: 900, color: score < 0.8 ? 'var(--red)' : 'var(--teal)', lineHeight: 1, letterSpacing: -2 }}>
                {score.toFixed(2)}
              </div>
              <motion.div
                initial={false}
                animate={{ opacity: loading ? 1 : 0 }}
                style={{ position: 'absolute', top: -10, right: -15 }}
              >
                <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2, borderColor: 'transparent', borderTopColor: 'var(--navy)' }} />
              </motion.div>
            </div>
          </div>

          {/* Real People Impact */}
          {impactResult && (
            <div className="card" style={{ background: 'linear-gradient(135deg, #fff7f7, #fdf2f2)', border: '1px solid #fca5a5' }}>
              <h3 className="card-title" style={{ color: 'var(--red)', display: 'flex', alignItems: 'center', gap: 6 }}>
                🧑‍🤝‍🧑 Human Impact
              </h3>
              <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--red)', marginTop: 8, marginBottom: 4 }}>
                {impactResult.people_impacted.toLocaleString()}{' '}
                <span style={{ fontSize: '1rem', fontWeight: 600, color: '#c0392b' }}>people unfairly rejected</span>
              </div>
              <p style={{ fontSize: '0.9rem', color: 'var(--text)', margin: '0 0 12px', opacity: 0.9 }}>
                {impactResult.impact_narrative}
              </p>
              {repairLevel > 0 && (
                <div style={{ fontSize: '0.8rem', background: '#e0f4ed', color: 'var(--teal)', padding: '6px 10px', borderRadius: 6, display: 'inline-block', fontWeight: 600 }}>
                  🛡️ Saved {Math.floor(impactResult.people_impacted * (repairLevel / (1-repairLevel || 0.1))).toLocaleString()} people by repairing up to {Math.round(repairLevel*100)}%
                </div>
              )}
            </div>
          )}

          {/* Mini Chart */}
          <div className="card">
            <h3 className="card-title" style={{ marginBottom: '1rem' }}>Metric Breakdown</h3>
            <div style={{ height: 180 }}>
               <Line data={chartData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 1 } } }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
