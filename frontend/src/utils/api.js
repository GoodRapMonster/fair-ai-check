import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({ baseURL: API_BASE });

api.interceptors.request.use(config => {
  const token = localStorage.getItem('fairsight_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, error => Promise.reject(error));

export const login = (username, password) => {
  const params = new URLSearchParams();
  params.append('username', username);
  params.append('password', password);
  return api.post('/auth/login', params, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
  });
};

export const register = (username, password) => api.post('/auth/register', { username, password });

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
export const getAnalyses = () => api.get('/analyses');

export const narrateAnalysis = (payload) => api.post('/narrate', payload);

export const generateStory = (payload) => api.post('/story', payload);

export const mitigateDataset = (payload) => api.post('/mitigate', payload);

export const simulateBias = (payload) => api.post('/simulate', payload);

export const calculateImpact = (payload) => api.post('/impact', payload);

export const generateReport = (payload) =>
  api.post('/report/generate', payload, { responseType: 'blob' });

export default api;
