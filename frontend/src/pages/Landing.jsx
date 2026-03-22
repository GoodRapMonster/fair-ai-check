import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

const SCORE_DEMO = [0.39, 0.52, 0.65, 0.73, 0.81, 0.88, 0.91];

function AnimatedScore() {
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    if (idx < SCORE_DEMO.length - 1) {
      const t = setTimeout(() => setIdx(i => i + 1), 900);
      return () => clearTimeout(t);
    }
  }, [idx]);

  const score = SCORE_DEMO[idx];
  const color = score < 0.6 ? '#E24B4A' : score < 0.8 ? '#EF9F27' : '#1D9E75';

  return (
    <div className="live-demo-card">
      <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.8rem', marginBottom: 8, letterSpacing: 1 }}>LIVE BIAS REMEDIATION DEMO</p>
      <motion.div
        key={idx}
        initial={{ scale: 0.85, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        style={{ fontSize: '4rem', fontWeight: 900, color, letterSpacing: -2, lineHeight: 1.1 }}
      >
        {score.toFixed(2)}
      </motion.div>
      <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.85rem', marginTop: 8 }}>
        {score < 0.6 ? '🔴 CRITICAL — applying reweighing...' :
         score < 0.8 ? '🟡 IMPROVING — running DIR...' : '🟢 FAIR — above legal threshold'}
      </p>
      <div style={{ background: 'rgba(255,255,255,0.1)', borderRadius: 99, height: 6, marginTop: 12, overflow: 'hidden' }}>
        <motion.div
          style={{ height: '100%', borderRadius: 99, background: color }}
          animate={{ width: `${score * 100}%` }}
          transition={{ duration: 0.6 }}
        />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
        <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.4)' }}>0 (biased)</span>
        <span style={{ fontSize: '0.7rem', color: '#EF9F27' }}>▲ 0.80 legal</span>
        <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.4)' }}>1.0 (fair)</span>
      </div>
    </div>
  );
}

export default function Landing() {
  const navigate = useNavigate();

  const domains = [
    { icon: '👔', name: 'Hiring', desc: 'Resume screening, interview scoring, promotion decisions', dataset: 'synthetic_hiring' },
    { icon: '🏦', name: 'Lending', desc: 'Loan approvals, credit scoring, interest rate setting', dataset: 'synthetic_lending' },
    { icon: '🏥', name: 'Medical', desc: 'Triage prioritization, treatment allocation, diagnosis AI', dataset: 'synthetic_medical' },
  ];

  const stats = [
    { num: '42,000+', label: 'People wrongly rejected by biased hiring AI per year' },
    { num: '87%', label: 'Of AI audits find significant demographic disparities' },
    { num: '$23B', label: 'In economic harm from algorithmic discrimination annually' },
    { num: '0.8', label: 'EEOC legal threshold for disparate impact ratio' },
  ];

  const steps = [
    { n: '1', title: 'Upload Dataset', desc: 'Drop your CSV, Excel, or JSON file. We auto-detect protected attributes.' },
    { n: '2', title: 'Detect Bias', desc: '7 fairness metrics computed. Proxy variables and intersectional bias revealed.' },
    { n: '3', title: 'Mitigate', desc: 'Apply reweighing, DIR, threshold adjustment. See before/after in real time.' },
    { n: '4', title: 'Download Report', desc: 'PDF compliance certificate citing EEOC, EU AI Act, and ECOA regulations.' },
  ];

  return (
    <div>
      {/* Hero */}
      <section className="hero-section">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
          <h1 className="hero-title">
            Catch AI discrimination<br />
            <span className="highlight">before it hurts real people</span>
          </h1>
          <p className="hero-subtitle">
            Upload a biased dataset. Get a plain-English report. Fix the bias.
            Download a compliance certificate. All in one tool — free.
          </p>
          <div className="hero-actions">
            <Link to="/upload" className="btn btn-teal btn-lg">⬆️ Upload Your Dataset</Link>
            <Link to="/dashboard" className="btn btn-outline btn-lg" style={{ color: 'white', borderColor: 'rgba(255,255,255,0.4)' }}>
              View Demo Dashboard
            </Link>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.3 }}>
          <AnimatedScore />
        </motion.div>

        {/* Domain cards */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}>
          <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.8rem', marginTop: '2rem', marginBottom: '1rem', letterSpacing: 1 }}>
            LOAD A SAMPLE DATASET →
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', maxWidth: 700, margin: '0 auto' }}>
            {domains.map(d => (
              <div key={d.name} className="domain-card" onClick={() => navigate('/upload?preset=' + d.dataset)}>
                <div className="domain-icon">{d.icon}</div>
                <h3>{d.name}</h3>
                <p>{d.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* Stats */}
      <section className="stats-banner">
        <div className="stats-grid">
          {stats.map(s => (
            <motion.div
              key={s.num}
              className="stat-item"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <h3>{s.num}</h3>
              <p>{s.label}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="how-it-works">
        <h2>How FairSight Works</h2>
        <div className="steps-grid">
          {steps.map((s, i) => (
            <motion.div
              key={s.n}
              className="step-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
            >
              <div className="step-number">{s.n}</div>
              <h3>{s.title}</h3>
              <p>{s.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Final CTA */}
      <section style={{ background: 'linear-gradient(135deg, #0C447C, #0a3560)', color: 'white', padding: '4rem 2rem', textAlign: 'center' }}>
        <h2 style={{ fontSize: '2rem', fontWeight: 800, margin: '0 0 1rem' }}>
          Ready to audit your AI?
        </h2>
        <p style={{ opacity: 0.75, marginBottom: '2rem', maxWidth: 500, margin: '0 auto 2rem' }}>
          Free to use. No account required for demo. Start with a synthetic dataset in 10 seconds.
        </p>
        <Link to="/upload" className="btn btn-teal btn-lg">Get Started Free →</Link>
      </section>
    </div>
  );
}
