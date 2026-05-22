import { useState, useEffect } from 'react'
import { matchJobToCandidates, getJobDetail, getJobs } from '../api/client'

export default function JobToCandidates() {
  const [jobs, setJobs] = useState([])
  const [jobId, setJobId] = useState('')
  const [jobTitle, setJobTitle] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState(0)

  // 加载岗位列表
  useEffect(() => {
    getJobs().then(({ data }) => setJobs(data)).catch(() => {})
  }, [])

  // 从 sessionStorage 读取从岗位库跳转过来的选中岗位
  useEffect(() => {
    const stored = sessionStorage.getItem('selectedJobId')
    if (stored) {
      setJobId(stored)
      sessionStorage.removeItem('selectedJobId')
      // 自动触发匹配
      handleMatch(stored)
    }
  }, [])

  const handleMatch = async (id) => {
    const targetId = id || jobId
    if (!targetId) {
      setError('请选择或输入岗位ID')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)
    const t0 = Date.now()
    try {
      const detail = await getJobDetail(parseInt(targetId))
      setJobTitle(detail.data.title)
      const { data } = await matchJobToCandidates(parseInt(targetId))
      setResult(data)
      setElapsed(((Date.now() - t0) / 1000).toFixed(1))
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const scoreColor = (score) => {
    if (score >= 7) return 'badge-green'
    if (score >= 5) return 'badge-yellow'
    return 'badge-red'
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">🔄 岗位匹配候选人</h1>
        <p className="text-sm text-gray-500 mt-1">选择一个岗位，AI 将从简历库中为你推荐最匹配的候选人</p>
      </div>

      {/* 选择岗位 */}
      <div className="card">
        <div className="flex flex-col sm:flex-row gap-3">
          <select
            className="input flex-1"
            value={jobId}
            onChange={(e) => {
              setJobId(e.target.value)
              setResult(null)
              setJobTitle('')
            }}
          >
            <option value="">-- 从岗位库选择 --</option>
            {jobs.map((j) => (
              <option key={j.id} value={j.id}>
                {j.company_name ? `${j.company_name} · ` : ''}{j.title} {j.location ? `(${j.location})` : ''} — ID:{j.id}
              </option>
            ))}
          </select>
          <button onClick={() => handleMatch()} disabled={loading || !jobId} className="btn-primary px-8">
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                匹配中...
              </>
            ) : '🔍 开始匹配候选人'}
          </button>
        </div>
        {jobTitle && (
          <div className="mt-3 flex items-center gap-2">
            <span className="badge-blue">当前岗位</span>
            <span className="font-medium text-gray-800">{jobTitle}</span>
            {elapsed > 0 && <span className="text-xs text-gray-400 ml-auto">耗时 {elapsed}s</span>}
          </div>
        )}
        {error && (
          <div className="mt-3 p-3 bg-red-50 text-red-700 rounded-lg text-sm border border-red-200">
            ❌ {error}
          </div>
        )}
      </div>

      {/* 结果 */}
      {result && (
        <div className="space-y-4">
          {/* Top3 */}
          {result.top_matches?.length > 0 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-3">🏆 Top3 候选人（AI 严格评分）</h2>
              <div className="space-y-3">
                {result.top_matches.map((cand, idx) => (
                  <div key={cand.resume_id} className="card hover:shadow-md transition">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-lg font-bold text-gray-900">#{idx + 1}</span>
                          <h3 className="text-lg font-bold text-gray-900">👤 {cand.name}</h3>
                        </div>
                        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-sm text-gray-500">
                          {cand.phone && <span>📞 {cand.phone}</span>}
                          {cand.email && <span>✉️ {cand.email}</span>}
                          {cand.education && <span>🎓 {cand.education}</span>}
                        </div>
                        <div className="flex flex-wrap gap-2 mt-2">
                          <span className={scoreColor(cand.score)}>
                            🎯 AI评分 {cand.score}/10
                          </span>
                          <span className="badge-blue">
                            📊 相似度 {(cand.similarity * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                      <p className="text-sm font-medium text-gray-700">📌 匹配理由</p>
                      <p className="text-sm text-gray-600 mt-1">{cand.reason}</p>
                    </div>

                    {cand.dimensions && (
                      <details className="mt-3 group">
                        <summary className="text-sm text-primary-600 cursor-pointer hover:text-primary-700 font-medium">
                          🔍 查看详细维度分析
                        </summary>
                        <pre className="mt-2 text-xs text-gray-600 bg-gray-50 p-3 rounded-lg whitespace-pre-wrap leading-relaxed">
                          {cand.dimensions}
                        </pre>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 其他候选人 */}
          {result.other_matches?.length > 0 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-3">📋 其他备选候选人（仅向量相似度）</h2>
              <div className="card divide-y divide-gray-100">
                {result.other_matches.map((c) => (
                  <div key={c.resume_id} className="py-3 first:pt-0 last:pb-0 flex items-center justify-between gap-4">
                    <div className="min-w-0">
                      <span className="font-medium text-gray-800">👤 {c.name}</span>
                      <span className="text-xs text-gray-400 ml-2">ID:{c.resume_id}</span>
                      <div className="text-xs text-gray-500 mt-0.5">
                        {c.phone && <span className="mr-3">📞 {c.phone}</span>}
                        {c.email && <span>✉️ {c.email}</span>}
                      </div>
                    </div>
                    <span className="badge-blue shrink-0">
                      {(c.similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.top_matches?.length === 0 && result.other_matches?.length === 0 && (
            <div className="card text-center py-12 text-gray-400">
              <div className="text-4xl mb-3">🔍</div>
              <p>暂无匹配结果</p>
              <p className="text-sm mt-1">请先在简历库中添加候选人简历</p>
            </div>
          )}
        </div>
      )}

      {/* 未选择时的提示 */}
      {!result && !loading && !error && (
        <div className="card text-center py-12 text-gray-400">
          <div className="text-4xl mb-3">🎯</div>
          <p>选择一个岗位开始匹配</p>
          <p className="text-sm mt-1">也可以从岗位库页面直接点击「匹配候选人」跳转</p>
        </div>
      )}
    </div>
  )
}