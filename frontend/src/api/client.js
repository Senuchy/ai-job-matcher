import axios from 'axios';

const API_BASE = '/api';

const client = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

const uploadClient = axios.create({
  baseURL: API_BASE,
});

// ─── 岗位 ───
export const getJobs = (params) => client.get('/jobs', { params });
export const getJobDetail = (id) => client.get(`/jobs/${id}`);
export const createJob = (data) => client.post('/jobs', data);
export const importJobsExcel = (file) => {
  const form = new FormData();
  form.append('file', file);
  return uploadClient.post('/jobs/import-excel', form);
};
export const deleteJobs = (ids) => client.delete('/jobs', { data: ids });

// ─── 简历 ───
export const getResumes = (params) => client.get('/resumes', { params });
export const createResume = (data) => client.post('/resumes', data);
export const importResumesExcel = (file) => {
  const form = new FormData();
  form.append('file', file);
  return uploadClient.post('/resumes/import-excel', form);
};
export const deleteResumes = (ids) => client.delete('/resumes', { data: ids });

// ─── 匹配 ───
export const matchResumeToJobs = (data) => client.post('/match/resume-to-jobs', data);
export const matchResumeUpload = (file) => {
  const form = new FormData();
  form.append('file', file);
  return uploadClient.post('/match/resume-to-jobs/upload', form);
};
export const matchJobToCandidates = (jobId) =>
  client.post('/match/job-to-candidates', { job_id: jobId });

export default client;