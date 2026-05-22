import { useState, useEffect } from 'react'
import { matchResumeToJobs, matchResumeUpload, getResumes } from '../api/client'

export default function MatchResumeToJobs() {
  const [mode, setMode] = useState('text') // text | file | select
  const [resumeText, setResumeText] = useState('')
  const [resumeFile, setResumeFile] = useState(null)
  const [selectedResumeId, setSelectedResumeId] = useState('')
  const [resumeList, setResumeList] = useState([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (mode === 'select' && resumeList.length === 0) {
      getResumes().then(({ data }) => setResumeList(data)).catch(() => {})
    }
  }, [mode])

  const handleSubmit = async () => {
    setLoading(true)
    setError('')
    setResult(null)
    const t0 = Date.now()
    try {
      let response
      if (mode === 'text') {
        if (!resumeText.trim()) throw new Error('请输入简历文本')
        response = await matchResumeToJobs({ resume_text: resumeText })
      } else if (mode === 'file') {
        if (!resumeFile) throw new Error('请选择文件')
        response = await matchResumeUpload(resumeFile)
      } else {
        if (!selectedResumeId) throw new Error('请选择简历')
        response = await matchResumeToJobs({ resume_id: parseInt(selectedResumeId) })
      }
      setResult(response.data)
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
        <h1 className="text-2xl font-bold text-gray-900">📄 简历智能匹配岗位</h1>
        <p className="text-sm text-gray-500 mt-1">上传或粘贴简历，AI 将为你推荐最匹配的岗位</p>
      </div>

      {/* 输入区域 */}
      <div className="card">
        <div className="flex flex-wrap gap-4 mb-5">
          {[
            { key: 'text', label: '✏️ 粘贴文本', desc: '直接粘贴简历内容' },
            { key: 'file', label: '📁 上传文件', desc: '支持 PDF / DOCX' },
            { key: 'select', label: '📇 简历库选择', desc: '从已有简历中选取' },
          ].map((m) => (
            <label
              key={m.key}
              className={`flex-1 min-w-[180px] cursor-pointer border-2 rounded-xl p-4 transition-all ${
                mode === m.key
                  ? 'border-primary-500 bg-primary-50 shadow-sm'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <input
                type="radio"
                name="mode"
                value={m.key}
                checked={mode === m.key}
                onChange={() => setMode(m.key)}
                className="sr-only"
              />
              <div className="font-medium text-sm">{m.label}</div>
              <div className="text-xs text-gray-400 mt-0.5">{m.desc}</div>
            </label>
          ))}
        </div>

        {mode === 'text' && (
          <textarea
            rows={8}
            className="input resize-none"
            placeholder="粘贴简历文本...&#10;&#10;例如：&#10;姓名：张三&#10;工作经历：5年 Java 开发经验...&#10;技能：Spring Boot, MySQL, Redis..."
            value={resumeText}
            onChange={(e) => setResumeText(e.target.value)}
          />
        )}

        {mode === 'file' && (
          <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-primary-400 transition">
            <input
              type="file"
              accept=".pdf,.docx"
              id="resume-file"
              className="hidden"
              onChange={(e) => setResumeFile(e.target.files[0])}
            />
            <label htmlFor="resume-file" className="cursor-pointer">
              <div className="text-4xl mb-2">📎</div>
              {resumeFile ? (
                <div>
                  <p className="font-medium text-primary-600">{resumeFile.name}</p>
                  <p className="text-xs text-gray-400 mt-1">点击重新选择</p>
                </div>
              ) : (
                <div>
                  <p className="font-medium text-gray-600">点击选择文件</p>
                  <p className="text-xs text-gray-400 mt-1">支持 PDF、DOCX 格式</p>
                </div>
              )}
            </label>
          </div>
        )}

        {mode === 'select' && (
          <select
            className="input"
            value={selectedResumeId}
            onChange={(e) => setSelectedResumeId(e.target.value)}
          >
            <option value="">-- 请选择简历 --</option>
            {resumeList.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name} {r.phone ? `(${r.phone})` : ''} — ID:{r.id}
              </option>
            ))}
            {resumeList.length === 0 && <option disabled>暂无简历，请先在简历库中添加</option>}
          </select>
        )}

        <div className="flex items-center gap-3 mt-5">
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="btn-primary px-8"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                匹配中...
              </>
            ) : '🔍 开始匹配'}
          </button>
          {elapsed > 0 && <span className="text-xs text-gray-400">耗时 {elapsed}s</span>}
        </div>

        {error && (
          <div className="mt-3 p-3 bg-red-50 text-red-700 rounded-lg text-sm border border-red-200">
            ❌ {error}
          </div>
        )}
      </div>

      {/* 结果展示 */}
      {result && (
        <div className="space-y-4">
          {/* Top 匹配 */}
          {result.top_matches?.length > 0 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-3">🏆 推荐岗位（AI 严格评分 Top3）</h2>
              <div className="space-y-3">
                {result.top_matches.map((job, idx) => (
                  <div key={job.job_id} className="card hover:shadow-md transition">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-lg font-bold text-gray-900">#{idx + 1}</span>
                          <h3 className="text-lg font-bold text-gray-900 truncate">{job.title}</h3>
                        </div>
                        <div className="flex flex-wrap gap-2 mt-2">
                          <span className={scoreColor(job.score)}>
                            🎯 AI评分 {job.score}/10
                          </span>
                          <span className="badge-blue">
                            📊 相似度 {(job.similarity * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                      <p className="text-sm font-medium text-gray-700">📌 匹配理由</p>
                      <p className="text-sm text-gray-600 mt-1">{job.reason}</p>
                    </div>

                    {job.dimensions && (
                      <details className="mt-3 group">
                        <summary className="text-sm text-primary-600 cursor-pointer hover:text-primary-700 font-medium">
                          🔍 查看详细维度分析
                        </summary>
                        <pre className="mt-2 text-xs text-gray-600 bg-gray-50 p-3 rounded-lg whitespace-pre-wrap leading-relaxed">
                          {job.dimensions}
                        </pre>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 其他备选 */}
          {result.other_matches?.length > 0 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-3">📋 其他备选岗位（仅向量相似度）</h2>
              <div className="card divide-y divide-gray-100">
                {result.other_matches.map((job) => (
                  <div key={job.job_id} className="py-3 first:pt-0 last:pb-0 flex items-center justify-between">
                    <div>
                      <span className="font-medium text-gray-800">{job.title}</span>
                      <span className="text-xs text-gray-400 ml-2">ID:{job.job_id}</span>
                    </div>
                    <span className="badge-blue">
                      {(job.similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.top_matches?.length === 0 && result.other_matches?.length === 0 && (
            <div className="card text-center py-12 text-gray-400">
              <div className="text-4xl mb-3">🔍</div>
              <p>暂无匹配结果，请先在岗位库中添加岗位</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}