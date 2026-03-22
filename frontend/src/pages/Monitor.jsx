import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Line } from 'react-chartjs-2';

const MOCK_MONITOR_DATA = {
  dates: ['Jan 1', 'Feb 1', 'Mar 1', 'Apr 1', 'May 1', 'Jun 1', 'Jul 1', 'Aug 1'],
  scores: [0.85, 0.84, 0.82, 0.81, 0.79, 0.77, 0.75, 0.72],
  alerts: [
    { date: 'May 15', message: 'Warning: Disparate impact for Age fell below 0.80 threshold. Re-train recommended.' },
    { date: 'Jul 20', message: 'Critical: Significant data drift detected in "Income" feature distribution.' }
  ]
};

export default function Monitor() {
  const [data] = useState(MOCK_MONITOR_DATA);

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

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>📡 Continuous Monitoring</h1>
        <p>Track fairness metrics over time and alert on data drift in production models.</p>
      </div>

      {/* Mock alert banner */}
      <div style={{ background: '#fdf2f2', border: '1px solid #fca5a5', padding: '1rem 1.5rem', borderRadius: 'var(--radius-lg)', marginBottom: '1.5rem', display: 'flex', alignItems: 'flex-start', gap: 12 }}>
        <div style={{ fontSize: '1.5rem' }}>🚨</div>
        <div>
          <h3 style={{ margin: '0 0 4px', color: 'var(--red)', fontSize: '1.1rem' }}>Active Drift Alert</h3>
          <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text)' }}>
            Your simulated production model's fairness score has degraded from 0.85 to 0.72 over the last 90 days. Retraining with mitigated data is strongly advised.
          </p>
        </div>
      </div>

      <div className="grid-2" style={{ gridTemplateColumns: '2fr 1fr' }}>
        <div className="card">
          <h3 className="card-title" style={{ marginBottom: '1rem' }}>Fairness Trajectory</h3>
          <Line data={chartData} options={chartOptions} height={150} />
        </div>

        <div className="card">
          <h3 className="card-title" style={{ marginBottom: '1rem' }}>Alert Log</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {data.alerts.map((a, i) => (
              <div key={i} style={{ padding: '0.75rem', background: 'var(--bg-gray)', borderLeft: `3px solid ${a.message.includes('Critical') ? 'var(--red)' : 'var(--amber)'}`, borderRadius: '0 6px 6px 0', fontSize: '0.85rem' }}>
                <strong style={{ display: 'block', color: 'var(--navy)', marginBottom: 4 }}>{a.date}</strong>
                {a.message}
              </div>
            ))}
          </div>

          <button className="btn btn-outline" style={{ marginTop: '1.5rem', width: '100%', justifyContent: 'center' }}>
            Connect Production API
          </button>
          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: 8 }}>
            (Requires backend integration with model serving infrastructure)
          </p>
        </div>
      </div>
    </div>
  );
}
