import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({ baseURL: API_BASE });

export const uploadDataset = (file) => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/upload', form);
};

export const loadBuiltinDataset = (name) => {
  const form = new FormData();
  form.append('dataset_id_name', name);
  return api.post('/load-builtin', form);
};

export const getBuiltinDatasets = () => api.get('/builtin-datasets');

export const analyzeDataset = (payload) => api.post('/analyze', payload);

export const narrateAnalysis = (payload) => api.post('/narrate', payload);

export const generateStory = (payload) => api.post('/story', payload);

export const mitigateDataset = (payload) => api.post('/mitigate', payload);

export const simulateBias = (payload) => api.post('/simulate', payload);

export const calculateImpact = (payload) => api.post('/impact', payload);

export const generateReport = (payload) =>
  api.post('/report/generate', payload, { responseType: 'blob' });

export default api;
