import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { generateStory } from '../utils/api';

function ScoreBar({ score, label, decision }) {
  const color = decision === 'APPROVED' ? 'var(--teal)' : 'var(--red)';
  return (
    <div style={{ textAlign: 'center' }}>
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 80, delay: 0.5 }}
        style={{ fontSize: '4rem', fontWeight: 900, color, letterSpacing: -3 }}
      >
        {score.toFixed(2)}
      </motion.div>
      <motion.span
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className={`badge ${decision === 'APPROVED' ? 'badge-pass' : 'badge-fail'}`}
        style={{ fontSize: '1rem', padding: '8px 20px', marginTop: 8 }}
      >
        {decision}
      </motion.span>
    </div>
  );
}

function ProfileCard({ profile, score, decision, flip }) {
  if (!profile) return null;
  const isApproved = decision === 'APPROVED';

  const fields = [
    ['Education', profile.education],
    ['Experience', `${profile.experience} years`],
    ['Skills', (profile.skills || []).join(', ')],
    ...Object.entries(profile.extra_fields || {}).slice(0, 3),
    [profile.protected_attr, <span key="p" className="profile-field-value changed">{profile.protected_value}</span>],
  ];

  return (
    <motion.div
      className="profile-card"
      initial={{ x: flip ? 80 : -80, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 80 }}
    >
      <div className={`profile-card-header ${decision === 'APPROVED' ? 'approved' : decision === 'REJECTED' ? 'rejected' : 'pending'}`}>
        <div className="profile-name">{profile.name}</div>
        <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>Age {profile.age}</div>
        <div style={{ marginTop: '1rem' }}>
          <ScoreBar score={score} label="AI Score" decision={decision} />
        </div>
      </div>
      <div className="profile-body">
        {fields.map(([label, value], i) => (
          <div key={i} className="profile-field">
            <span className="profile-field-label">{label}</span>
            <span className={`profile-field-value ${label === profile.protected_attr ? 'changed' : ''}`}>
              {typeof value === 'string' ? value : value}
            </span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

export default function StoryMode() {
  const [story, setStory] = useState(null);
  const [loading, setLoading] = useState(false);
  const [protectedAttr, setProtectedAttr] = useState('');
  const [availableAttrs, setAvailableAttrs] = useState([]);

  useEffect(() => {
    const config = sessionStorage.getItem('fairsight_config');
    const analysisData = sessionStorage.getItem('fairsight_analysis');
    let c = {};
    if (config) {
      c = JSON.parse(config);
      setAvailableAttrs(c.protected_attrs || []);
      setProtectedAttr(c.protected_attrs?.[0] || '');
    }
    if (!analysisData) return;
    const analysis = JSON.parse(analysisData);
    if (analysis.analysis_id && c?.protected_attrs?.[0]) {
      loadStory(analysis.analysis_id, c.protected_attrs[0]);
    }
  }, []);

  const loadStory = async (analysisId, attr) => {
    if (!analysisId) return;
    setLoading(true);
    try {
      const res = await generateStory({ analysis_id: analysisId, protected_attr: attr });
      setStory(res.data);
    } catch (e) {
      toast.error('Story generation failed. Check analysis is complete.');
    } finally {
      setLoading(false);
    }
  };

  const handleAttrChange = (attr) => {
    setProtectedAttr(attr);
    const analysis = sessionStorage.getItem('fairsight_analysis');
    if (analysis) loadStory(JSON.parse(analysis).analysis_id, attr);
  };

  const handleGoToDashboardFirst = () => {
    // Attempt with whatever analysis is available
    const data = sessionStorage.getItem('fairsight_analysis');
    if (data) {
      const a = JSON.parse(data);
      loadStory(a.analysis_id, protectedAttr || availableAttrs[0]);
    }
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>📖 Bias Story Mode</h1>
        <p>Watch discrimination happen — same qualifications, different outcome.</p>
      </div>

      {/* Controls */}
      {availableAttrs.length > 0 && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <div>
              <label className="form-label" style={{ marginBottom: 4 }}>Protected attribute to vary:</label>
              <select className="form-control form-select" style={{ width: 200 }} value={protectedAttr} onChange={e => handleAttrChange(e.target.value)}>
                {availableAttrs.map(a => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
            <button className="btn btn-primary" onClick={handleGoToDashboardFirst} disabled={loading}>
              {loading ? <><span className="spinner" />Generating...</> : '🔄 Regenerate Story'}
            </button>
            {!sessionStorage.getItem('fairsight_analysis') && (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                No analysis found. <Link to="/upload" style={{ color: 'var(--navy)' }}>Upload a dataset first →</Link>
              </div>
            )}
          </div>
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center', padding: '4rem' }}>
          <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3, borderColor: 'rgba(12,68,124,0.2)', borderTopColor: 'var(--navy)', margin: '0 auto 1rem' }} />
          <p style={{ color: 'var(--text-muted)' }}>Generating counterfactual profiles and AI narration...</p>
        </div>
      )}

      {story && !loading && (
        <AnimatePresence>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            {/* Headline */}
            <div style={{ textAlign: 'center', padding: '1.5rem', background: 'linear-gradient(135deg, #fff7f7, #fff)', border: '1px solid #fca5a5', borderRadius: 'var(--radius-lg)', marginBottom: '1.5rem' }}>
              <h2 style={{ color: 'var(--navy)', margin: '0 0 8px', fontSize: '1.3rem' }}>
                The ONLY difference between these two people is their <span style={{ color: 'var(--red)' }}>{story.difference_attr}</span>
              </h2>
              <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '0.9rem' }}>
                Identical qualifications. Identical experience. Identical skills. Different algorithm outcome.
              </p>
            </div>

            {/* Profile Cards */}
            <div className="grid-2" style={{ gap: '2rem', marginBottom: '1.5rem' }}>
              <ProfileCard profile={story.profile_a} score={story.score_a} decision={story.decision_a} flip={false} />
              <ProfileCard profile={story.profile_b} score={story.score_b} decision={story.decision_b} flip={true} />
            </div>

            {/* Score Gap visualization */}
            <div className="card" style={{ marginBottom: '1.5rem', textAlign: 'center' }}>
              <h3 style={{ color: 'var(--navy)', margin: '0 0 1rem' }}>AI Score Gap</h3>
              <div style={{ display: 'flex', align: 'center', gap: '1rem', maxWidth: 500, margin: '0 auto' }}>
                <div style={{ flex: 1, textAlign: 'right' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{story.profile_a?.name}</span>
                  <div style={{ height: 12, background: 'var(--teal)', borderRadius: '99px 0 0 99px', marginTop: 4, width: `${story.score_a * 100}%`, marginLeft: 'auto' }} />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', fontWeight: 900, color: 'var(--navy)', fontSize: '1.1rem', whiteSpace: 'nowrap' }}>
                  Δ {Math.abs(story.score_a - story.score_b).toFixed(2)}
                </div>
                <div style={{ flex: 1, textAlign: 'left' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{story.profile_b?.name}</span>
                  <div style={{ height: 12, background: 'var(--red)', borderRadius: '0 99px 99px 0', marginTop: 4, width: `${story.score_b * 100}%` }} />
                </div>
              </div>
            </div>

            {/* AI Story Narration */}
            {story.story_text && (
              <div className="narration-card">
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10 }}>✨ GEMINI AI STORY NARRATION</div>
                <p style={{ fontSize: '1rem', lineHeight: 1.8, color: 'var(--text)', margin: 0, fontStyle: 'italic' }}>
                  "{story.story_text}"
                </p>
              </div>
            )}

            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
              <Link to="/mitigate" className="btn btn-teal">🔧 Fix This Bias →</Link>
              <Link to="/report" className="btn btn-outline">📄 Add to Report</Link>
            </div>
          </motion.div>
        </AnimatePresence>
      )}

      {!story && !loading && !sessionStorage.getItem('fairsight_analysis') && (
        <div className="card" style={{ textAlign: 'center', padding: '4rem' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📖</div>
          <h3 style={{ color: 'var(--navy)' }}>No analysis available yet</h3>
          <p style={{ color: 'var(--text-muted)' }}>Upload a dataset and run bias analysis first to see the Story Mode.</p>
          <Link to="/upload" className="btn btn-primary" style={{ marginTop: '1rem' }}>Upload Dataset →</Link>
        </div>
      )}
    </div>
  );
}
