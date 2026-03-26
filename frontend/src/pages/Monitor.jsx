import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import { getAnalyses } from '../utils/api';

export default function Monitor() {
  const [data, setData] = useState({ dates: [], scores: [], alerts: [] });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await getAnalyses();
        const history = [...res.data].reverse(); // Sort by oldest first for the chart
        
        const dates = history.map(h => new Date(h.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' }));
        const scores = history.map(h => h.bias_score);
        
        // Use bias_score_breakdown fields if available, otherwise just warn on score
        const alerts = history
          .filter(h => h.bias_score < 0.8)
          .map(h => ({
            date: new Date(h.timestamp).toLocaleDateString(),
            message: `Alert: Fairness score (${h.bias_score.toFixed(2)}) is below the 0.80 legal threshold for ${h.domain}.`
          }));

        setData({ dates, scores, alerts });
      } catch (e) {
        console.error("Failed to fetch history:", e);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const chartData = {
    labels: data.dates,
    datasets: [{
      label: 'Fairness Score Over Time',
      data: data.scores,
      borderColor: '#0C447C',
      backgroundColor: 'rgba(12,68,124,0.1)',
      pointBackgroundColor: data.scores.map(s => s < 0.8 ? '#E24B4A' : '#1D9E75'),
      pointRadius: 6,
      fill: true,
      tension: 0.3
    }]
  };

  const chartOptions = {
    responsive: true,
    scales: {
      y: { min: 0.5, max: 1.0, grid: { color: 'rgba(0,0,0,0.05)' } }
    },
    plugins: {
      annotation: { annotations: { line1: { type: 'line', yMin: 0.8, yMax: 0.8, borderColor: '#EF9F27', borderDash: [5, 5], label: { content: 'Legal Threshold', enabled: true } } } }
    }
  };

  if (loading) {
    return (
      <div className="page-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
        <div className="spinner" style={{ borderColor: 'rgba(12,68,124,0.1)', borderTopColor: 'var(--navy)' }} />
      </div>
    );
  }

  const latestScore = data.scores[data.scores.length - 1] || 1.0;
  const isBiased = latestScore < 0.8;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>📡 Continuous Monitoring</h1>
        <p>Track fairness metrics over time and alert on data drift in production models.</p>
      </div>

      {/* Dynamic alert banner */}
      {isBiased && (
        <div style={{ background: '#fdf2f2', border: '1px solid #fca5a5', padding: '1rem 1.5rem', borderRadius: 'var(--radius-lg)', marginBottom: '1.5rem', display: 'flex', alignItems: 'flex-start', gap: 12 }}>
          <div style={{ fontSize: '1.5rem' }}>🚨</div>
          <div>
            <h3 style={{ margin: '0 0 4px', color: 'var(--red)', fontSize: '1.1rem' }}>Active Risk Alert</h3>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text)' }}>
              Your latest analysis fairness score ({latestScore.toFixed(2)}) has fallen below the 0.80 legal threshold. 
              Immediate mitigation and model re-audit is strongly advised.
            </p>
          </div>
        </div>
      )}

      {data.scores.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1.5rem' }}>📊</div>
          <h2>No History Yet</h2>
          <p>Run your first analysis to start tracking fairness trends.</p>
          <button className="btn btn-primary" onClick={() => window.location.hash = '/upload'}>Go to Upload</button>
        </div>
      ) : (
        <div className="grid-2" style={{ gridTemplateColumns: '2fr 1fr' }}>
          <div className="card">
            <h3 className="card-title" style={{ marginBottom: '1rem' }}>Fairness Trajectory</h3>
            <Line data={chartData} options={chartOptions} height={150} />
          </div>

          <div className="card">
            <h3 className="card-title" style={{ marginBottom: '1rem' }}>Alert Log</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {data.alerts.length === 0 ? (
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No critical alerts detected in the tracking history.</p>
              ) : (
                data.alerts.map((a, i) => (
                  <div key={i} style={{ padding: '0.75rem', background: 'var(--bg-gray)', borderLeft: '3px solid var(--red)', borderRadius: '0 6px 6px 0', fontSize: '0.85rem' }}>
                    <strong style={{ display: 'block', color: 'var(--navy)', marginBottom: 4 }}>{a.date}</strong>
                    {a.message}
                  </div>
                ))
              )}
            </div>

            <button className="btn btn-outline" style={{ marginTop: '1.5rem', width: '100%', justifyContent: 'center' }}>
              Connect Production API
            </button>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: 8 }}>
              (Requires backend integration with model serving infrastructure)
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
