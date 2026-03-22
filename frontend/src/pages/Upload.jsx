import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { uploadDataset, loadBuiltinDataset, getBuiltinDatasets } from '../utils/api';

const DOMAINS = ['hiring', 'lending', 'medical', 'criminal_justice', 'other'];

export default function Upload() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [step, setStep] = useState(1); // 1=upload, 2=configure, 3=preview
  const [loading, setLoading] = useState(false);
  const [uploadData, setUploadData] = useState(null);
  const [builtinDatasets, setBuiltinDatasets] = useState([]);
  const [domain, setDomain] = useState('hiring');
  const [outcomeCol, setOutcomeCol] = useState('');
  const [selectedProtected, setSelectedProtected] = useState([]);
  const [privilegedGroups, setPrivilegedGroups] = useState({});

  useEffect(() => {
    getBuiltinDatasets().then(r => setBuiltinDatasets(r.data)).catch(() => {});
    const preset = searchParams.get('preset');
    if (preset) handleLoadBuiltin(preset);
  }, []);

  const handleLoadBuiltin = async (name) => {
    setLoading(true);
    try {
      const res = await loadBuiltinDataset(name);
      setUploadData(res.data);
      setSelectedProtected(res.data.detected_protected_attrs.map(a => a.column));
      const domainMap = { synthetic_hiring: 'hiring', synthetic_lending: 'lending', synthetic_medical: 'medical', compas: 'criminal_justice', adult: 'hiring', german: 'lending' };
      setDomain(domainMap[name] || 'hiring');
      setStep(2);
      toast.success('Dataset loaded!');
    } catch (e) {
      toast.error('Failed to load dataset.');
    } finally {
      setLoading(false);
    }
  };

  const onDrop = useCallback(async (files) => {
    const file = files[0];
    if (!file) return;
    setLoading(true);
    try {
      const res = await uploadDataset(file);
      setUploadData(res.data);
      setSelectedProtected(res.data.detected_protected_attrs.map(a => a.column));
      setStep(2);
      toast.success(`Uploaded ${res.data.rows.toLocaleString()} rows!`);
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Upload failed.');
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'text/csv': ['.csv'], 'application/json': ['.json'], 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] },
    maxFiles: 1,
  });

  const handleAnalyze = () => {
    if (!outcomeCol) { toast.error('Please select an outcome column.'); return; }
    if (selectedProtected.length === 0) { toast.error('Please select at least one protected attribute.'); return; }
    // Store in session for dashboard
    sessionStorage.setItem('fairsight_config', JSON.stringify({
      dataset_id: uploadData.dataset_id,
      outcome_col: outcomeCol,
      protected_attrs: selectedProtected,
      domain,
      privileged_groups: privilegedGroups,
    }));
    navigate('/dashboard');
  };

  const toggleProtected = (col) => {
    setSelectedProtected(p => p.includes(col) ? p.filter(x => x !== col) : [...p, col]);
  };

  const setPrivileged = (attr, val) => {
    setPrivilegedGroups(g => ({ ...g, [attr]: isNaN(val) ? val : Number(val) }));
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Upload Dataset</h1>
        <p>Upload your data or choose a built-in sample to get started immediately.</p>
      </div>

      {/* Step indicators */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: '2rem', maxWidth: 500 }}>
        {['Upload', 'Configure', 'Analyze'].map((label, i) => (
          <React.Fragment key={label}>
            <div style={{ textAlign: 'center' }}>
              <div className={`step-dot ${step > i+1 ? 'complete' : step === i+1 ? 'active' : ''}`}>{step > i+1 ? '✓' : i+1}</div>
              <div className="step-label">{label}</div>
            </div>
            {i < 2 && <div className={`step-connector ${step > i+1 ? 'active' : ''}`} style={{ flex: 1, height: 2, background: step > i+1 ? 'var(--teal)' : 'var(--border)', marginBottom: 20 }} />}
          </React.Fragment>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {step === 1 && (
          <motion.div key="step1" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            {/* Drag and Drop */}
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              <div className="dropzone-icon">{loading ? '⏳' : '📂'}</div>
              <h3>{isDragActive ? 'Drop it here!' : 'Drag & drop your dataset'}</h3>
              <p>Supports CSV, Excel (.xlsx), and JSON · Max recommended: 500k rows</p>
              {!loading && <button className="btn btn-primary" style={{ marginTop: '1rem' }}>Or Browse Files</button>}
              {loading && <div className="spinner" style={{ margin: '1rem auto', borderColor: 'rgba(12,68,124,0.2)', borderTopColor: 'var(--navy)' }} />}
            </div>

            {/* Built-in datasets */}
            <div className="card" style={{ marginTop: '2rem' }}>
              <div className="card-header">
                <div>
                  <h3 className="card-title">Built-in Sample Datasets</h3>
                  <p className="card-subtitle">Click to load instantly — no upload needed</p>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                {builtinDatasets.map(ds => (
                  <div key={ds.id} className="card" style={{ cursor: 'pointer', transition: 'all 0.2s', border: '1px solid var(--border)' }}
                    onClick={() => handleLoadBuiltin(ds.id)}
                    onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--navy)'}
                    onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
                  >
                    <div style={{ fontWeight: 700, color: 'var(--navy)', marginBottom: 6 }}>{ds.name}</div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 8 }}>{ds.description.substring(0, 100)}...</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--red)', fontWeight: 600, background: '#fde8e8', padding: '3px 8px', borderRadius: 99, display: 'inline-block' }}>
                      ⚠️ {ds.known_bias.substring(0, 45)}...
                    </div>
                    <div style={{ marginTop: 8, fontSize: '0.75rem', color: 'var(--text-muted)' }}>{ds.rows.toLocaleString()} rows · {ds.domain}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: '1rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>Or generate a synthetic dataset:</p>
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
                  {['synthetic_hiring', 'synthetic_lending', 'synthetic_medical'].map(name => (
                    <button key={name} className="btn btn-outline btn-sm" onClick={() => handleLoadBuiltin(name)}>
                      {name.replace('synthetic_', '').charAt(0).toUpperCase() + name.replace('synthetic_', '').slice(1)} (synthetic)
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {step === 2 && uploadData && (
          <motion.div key="step2" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
            <div className="grid-2">
              {/* Left — Configure */}
              <div>
                <div className="card">
                  <div className="card-header"><h3 className="card-title">Dataset Info</h3></div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.875rem' }}>
                    <div>
                      <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 600 }}>FILE</div>
                      <div style={{ fontWeight: 700 }}>{uploadData.filename}</div>
                    </div>
                    <div>
                      <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 600 }}>ROWS</div>
                      <div style={{ fontWeight: 700 }}>{uploadData.rows.toLocaleString()}</div>
                    </div>
                    <div>
                      <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 600 }}>COLUMNS</div>
                      <div style={{ fontWeight: 700 }}>{uploadData.columns.length}</div>
                    </div>
                  </div>
                </div>

                <div className="card" style={{ marginTop: '1rem' }}>
                  <h3 className="card-title" style={{ marginBottom: '1rem' }}>Configuration</h3>
                  <div className="form-group">
                    <label className="form-label">Domain</label>
                    <select className="form-control form-select" value={domain} onChange={e => setDomain(e.target.value)}>
                      {DOMAINS.map(d => <option key={d} value={d}>{d.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}</option>)}
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Outcome Column (what we're predicting)</label>
                    <select className="form-control form-select" value={outcomeCol} onChange={e => setOutcomeCol(e.target.value)}>
                      <option value="">— Select outcome column —</option>
                      {uploadData.columns.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                </div>

                <div className="card" style={{ marginTop: '1rem' }}>
                  <h3 className="card-title" style={{ marginBottom: '1rem' }}>Protected Attributes</h3>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 0 }}>Auto-detected below. Toggle to include/exclude.</p>
                  {uploadData.detected_protected_attrs.length === 0 && (
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No protected attributes auto-detected. Select manually:</p>
                  )}
                  {uploadData.columns.map(col => {
                    const detected = uploadData.detected_protected_attrs.find(a => a.column === col);
                    const isSelected = selectedProtected.includes(col);
                    return (
                      <div key={col} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 0', borderBottom: '0.5px solid var(--border)' }}>
                        <div>
                          <span style={{ fontWeight: isSelected ? 600 : 400, fontSize: '0.875rem' }}>{col}</span>
                          {detected && <span style={{ marginLeft: 8, fontSize: '0.7rem', background: '#e0f4ed', color: 'var(--teal)', padding: '2px 7px', borderRadius: 99, fontWeight: 600 }}>
                            {detected.detected_type} · {Math.round(detected.confidence * 100)}%
                          </span>}
                        </div>
                        <label className="toggle">
                          <input type="checkbox" checked={isSelected} onChange={() => toggleProtected(col)} />
                          <span className="toggle-slider" />
                        </label>
                      </div>
                    );
                  })}
                </div>

                {/* Privileged group values */}
                {selectedProtected.length > 0 && (
                  <div className="card" style={{ marginTop: '1rem' }}>
                    <h3 className="card-title" style={{ marginBottom: '1rem' }}>Privileged Group Values</h3>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 0 }}>Which value = the advantaged group? (e.g. Male=1)</p>
                    {selectedProtected.map(attr => (
                      <div key={attr} className="form-group">
                        <label className="form-label">{attr} — privileged value</label>
                        <input className="form-control" placeholder={`e.g. 1 or "Male"`}
                          onChange={e => setPrivileged(attr, e.target.value)} />
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Right — Preview */}
              <div>
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">Data Preview (first 10 rows)</h3>
                  </div>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem' }}>
                      <thead>
                        <tr>
                          {uploadData.columns.map(c => (
                            <th key={c} style={{ padding: '8px 10px', background: 'var(--bg-gray)', borderBottom: '1px solid var(--border)', textAlign: 'left', whiteSpace: 'nowrap', fontWeight: 600, color: selectedProtected.includes(c) ? 'var(--navy)' : 'var(--text-muted)' }}>
                              {c}{selectedProtected.includes(c) && ' 🛡️'}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {uploadData.preview_rows.map((row, i) => (
                          <tr key={i}>
                            {uploadData.columns.map(c => (
                              <td key={c} style={{ padding: '7px 10px', borderBottom: '0.5px solid var(--border)', whiteSpace: 'nowrap' }}>{String(row[c] ?? '')}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <button className="btn btn-primary btn-lg" style={{ width: '100%', marginTop: '1.25rem', justifyContent: 'center' }} onClick={handleAnalyze}>
                  🔍 Analyze for Bias →
                </button>
                <button className="btn btn-outline" style={{ width: '100%', marginTop: '0.5rem', justifyContent: 'center' }} onClick={() => setStep(1)}>
                  ← Back to Upload
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
