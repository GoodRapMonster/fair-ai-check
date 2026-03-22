import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { generateReport } from '../utils/api';

export default function Report() {
  const [analysisId, setAnalysisId] = useState(null);
  const [domain, setDomain] = useState('hiring');
  const [loading, setLoading] = useState(false);
  const [companyName, setCompanyName] = useState('Demo Corp');
  const [modelName, setModelName] = useState('Production HR Screener v1');

  useEffect(() => {
    const config = sessionStorage.getItem('fairsight_config');
    const a = sessionStorage.getItem('fairsight_analysis');
    if (config) setDomain(JSON.parse(config).domain || 'hiring');
    if (a) setAnalysisId(JSON.parse(a).analysis_id);
  }, []);

  const handleDownload = async () => {
    if (!analysisId) { toast.error('Run analysis on Dashboard first.'); return; }
    setLoading(true);
    try {
      const payload = {
        analysis_id: analysisId,
        company_name: companyName,
        model_name: modelName,
        domain: domain,
        include_mitigation: !!sessionStorage.getItem('fairsight_mitigation')
      };
      const res = await generateReport(payload);
      
      // Blob download
      const url = window.URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `FairSight_Audit_${modelName.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success('Report downloaded!');
    } catch (e) {
      toast.error('Failed to generate PDF report.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container" style={{ maxWidth: 800 }}>
      <div className="page-header" style={{ textAlign: 'center' }}>
        <h1>📄 Compliance Certificate</h1>
        <p>Generate a board-ready PDF report detailing bias findings, mitigations, and legal standing.</p>
      </div>

      <div className="card" style={{ padding: '2.5rem' }}>
        {!analysisId && (
          <div style={{ background: '#fff3cd', color: '#856404', padding: '1rem', borderRadius: 8, marginBottom: '2rem', fontSize: '0.9rem', textAlign: 'center' }}>
            ⚠️ <strong>No active analysis found.</strong> You must upload a dataset and run an analysis before generating a report.
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Company Name</label>
            <input className="form-control" value={companyName} onChange={e => setCompanyName(e.target.value)} />
          </div>
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Model/Dataset Name</label>
            <input className="form-control" value={modelName} onChange={e => setModelName(e.target.value)} />
          </div>
        </div>

        <div style={{ background: 'var(--bg-gray)', padding: '1.5rem', borderRadius: 'var(--radius)', border: '1px solid var(--border)', marginBottom: '2.5rem' }}>
          <h4 style={{ margin: '0 0 1rem', color: 'var(--navy)' }}>Report Contents ({domain} regulations applied)</h4>
          <ul style={{ margin: 0, paddingLeft: '1.25rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            <li style={{ marginBottom: 6 }}>Executive Summary (Gemini generated)</li>
            <li style={{ marginBottom: 6 }}>7-Metric Fairness Audit Results</li>
            <li style={{ marginBottom: 6 }}>Proxy Variable Risk Assessment</li>
            <li style={{ marginBottom: 6 }}>Intersectional Bias Analysis</li>
            {sessionStorage.getItem('fairsight_mitigation') && (
              <li style={{ color: 'var(--teal)', fontWeight: 600 }}>MITIGATION LOG INCLUDED: Proof of proactive bias repair</li>
            )}
            <li style={{ marginTop: 12 }}>
              <strong>Regulatory alignment:</strong>{' '}
              {domain === 'hiring' ? 'EEOC Title VII, NYC Local Law 144'
              : domain === 'lending' ? 'ECOA, Fair Housing Act'
              : 'EU AI Act, White House Blueprint for AI Bill of Rights'}
            </li>
          </ul>
        </div>

        <button
          className="btn btn-primary btn-lg"
          style={{ width: '100%', justifyContent: 'center', padding: '1.25rem' }}
          disabled={!analysisId || loading}
          onClick={handleDownload}
        >
          {loading ? (
            <><span className="spinner" style={{ marginRight: 10 }} /> Generating PDF...</>
          ) : (
            <><span style={{ fontSize: '1.2rem', marginRight: 8 }}>📥</span> Download Official Compliance PDF</>
          )}
        </button>

        <p style={{ textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1rem' }}>
          Generated via ReportLab. Includes digital timestamp and audit fingerprint.
        </p>
      </div>
    </div>
  );
}
