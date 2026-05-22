import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import MatchResumeToJobs from './pages/MatchResumeToJobs'
import JobLibrary from './pages/JobLibrary'
import ResumeLibrary from './pages/ResumeLibrary'
import JobToCandidates from './pages/JobToCandidates'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/match" replace />} />
          <Route path="/match" element={<MatchResumeToJobs />} />
          <Route path="/jobs" element={<JobLibrary />} />
          <Route path="/resumes" element={<ResumeLibrary />} />
          <Route path="/reverse" element={<JobToCandidates />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}