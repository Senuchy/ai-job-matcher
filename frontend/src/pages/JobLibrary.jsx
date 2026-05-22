import { useState, useEffect, useCallback } from 'react'
import { getJobs, createJob, importJobsExcel, deleteJobs } from '../api/client'
import { useNavigate } from 'react-router-dom'

export default function JobLibrary() {
  const navigate = useNavigate()
  const [jobs, setJobs] = useState([])
  const [filters, setFilters] = useState({ company: '', platform: '', location: '' })
  const [selectedIds, setSelectedIds] = useState([])
  const [showForm, setShowForm] = useState(false)
  const [showImport, setShowImport] = useState(false)
  const [loading, setLoading] = useState(false)
  const [importing, setImporting] = useState(false)
  const [newJob, setNewJob] = useState({
    title: '', jd_text: '', company_name: '', platform: '',
    department: '', location: '', core_business: '', candidate_profile: ''
  })

  const loadJobs = useCallback(async () => {
    setLoading(true)
    try {
      const params = {}
      if (filters.company) params.company = filters.company
      if (filters.platform) params.platform = filters.platform
      if (filters.location) params.location = filters.location
      const { data } = await getJobs(params)
      setJobs(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [filters])

  useEffect(() => { loadJobs() }, [loadJobs])

  const handleCreate = async () => {
    if (!newJob.title.trim() || !newJob.jd_text.trim()) {
      alert('岗位名称和JD为必填项')
      return
    }
    try {
      await createJob(newJob)
      setShowForm(false)
      setNewJob({ title: '', jd_text: '', company_name: '', platform: '', department: '', location: '', core_business: '', candidate_profile: '' })
      loadJobs()
    } catch (err) {
      alert('创建失败: ' + (err.response?.data?.detail || err.message))
    }
  }

  const handleImportExcel = async (file) => {
    if (!file) return
    setImporting(true)
    try {
      const { data } = await importJobsExcel(file)
      alert(`导入完成！成功 ${data.success} / 共 ${data.total} 条`)
      loadJobs()
    } catch (err) {
      alert('导入失败: ' + (err.response?.data?.detail || err.message))
    } finally {
      setImporting(false)
    }
  }

  const handleDelete = async () => {
    if (selectedIds.length === 0) return
    if (!confirm(`确定删除选中的 ${selectedIds.length} 个岗位吗？`)) return
    try {
      await deleteJobs(selectedIds)
      setSelectedIds([])
      loadJobs()
    } catch (err) {
      alert('删除失败')
    }
  }

  const toggleSelect = (id) => {
    setSelectedIds((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])
  }

  const toggleSelectAll = () => {
    if (selectedIds.length === jobs.length) setSelectedIds([])
    else setSelectedIds(jobs.map((j) => j.id))
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">📁 岗位库管理</h1>
          <p className="text-sm text-gray-500 mt-1">管理所有岗位信息，支持筛选、录入、导入和批量删除</p>
        </div>
        <span className="badge-blue text-sm">{jobs.length} 个岗位</span>
      </div>

      {/* 筛选栏 */}
      <div className="card">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <input className="input" placeholder="🔍 公司名称" value={filters.company}
            onChange={(e) => setFilters({ ...filters, company: e.target.value })} />
          <input className="input" placeholder="🔍 平台" value={filters.platform}
            onChange={(e) => setFilters({ ...filters, platform: e.target.value })} />
          <input className="input" placeholder="🔍 工作地点" value={filters.location}
            onChange={(e) => setFilters({ ...filters, location: e.target.value })} />
        </div>
      </div>

      {/* 操作栏 */}
      <div className="flex flex-wrap gap-2">
        <button onClick={() => { setShowForm(!showForm); setShowImport(false) }}
          className="btn-success">＋ 手动录入</button>
        <button onClick={() => { setShowImport(!showImport); setShowForm(false) }}
          className="btn-secondary">📊 Excel 导入</button>
        {selectedIds.length > 0 && (
          <button onClick={handleDelete} className="btn-danger">
            🗑️ 删除选中 ({selectedIds.length})
          </button>
        )}
        {jobs.length > 0 && (
          <label className="btn-secondary cursor-pointer ml-auto">
            <input type="checkbox" className="mr-1" checked={selectedIds.length === jobs.length && jobs.length > 0}
              onChange={toggleSelectAll} />
            全选
          </label>
        )}
      </div>

      {/* 手动录入表单 */}
      {showForm && (
        <div className="card border-primary-200 bg-primary-50/30">
          <h3 className="font-bold text-gray-800 mb-4">📝 新增岗位</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">岗位名称 *</label>
              <input className="input" placeholder="如：高级Java开发" value={newJob.title}
                onChange={(e) => setNewJob({ ...newJob, title: e.target.value })} />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">公司名称</label>
              <input className="input" placeholder="如：腾讯" value={newJob.company_name}
                onChange={(e) => setNewJob({ ...newJob, company_name: e.target.value })} />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">公司平台</label>
              <input className="input" placeholder="如：互联网" value={newJob.platform}
                onChange={(e) => setNewJob({ ...newJob, platform: e.target.value })} />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">部门</label>
              <input className="input" placeholder="如：技术部" value={newJob.department}
                onChange={(e) => setNewJob({ ...newJob, department: e.target.value })} />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">工作地点</label>
              <input className="input" placeholder="如：深圳" value={newJob.location}
                onChange={(e) => setNewJob({ ...newJob, location: e.target.value })} />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">核心业务</label>
              <input className="input" placeholder="如：社交产品" value={newJob.core_business}
                onChange={(e) => setNewJob({ ...newJob, core_business: e.target.value })} />
            </div>
            <div className="sm:col-span-2">
              <label className="text-xs text-gray-500 mb-1 block">岗位JD *</label>
              <textarea className="input resize-none" rows={4} placeholder="粘贴完整的岗位描述..."
                value={newJob.jd_text} onChange={(e) => setNewJob({ ...newJob, jd_text: e.target.value })} />
            </div>
            <div className="sm:col-span-2">
              <label className="text-xs text-gray-500 mb-1 block">人选画像</label>
              <textarea className="input resize-none" rows={2} placeholder="理想的候选人特征..."
                value={newJob.candidate_profile} onChange={(e) => setNewJob({ ...newJob, candidate_profile: e.target.value })} />
            </div>
          </div>
          <div className="flex gap-2 mt-4">
            <button onClick={handleCreate} className="btn-primary">💾 保存岗位</button>
            <button onClick={() => setShowForm(false)} className="btn-secondary">取消</button>
          </div>
        </div>
      )}

      {/* Excel 导入 */}
      {showImport && (
        <div className="card border-blue-200 bg-blue-50/30">
          <h3 className="font-bold text-gray-800 mb-2">📊 Excel 批量导入</h3>
          <p className="text-sm text-gray-500 mb-3">
            Excel 需包含列：<code className="bg-gray-200 px-1 rounded">公司名</code>{' '}
            <code className="bg-gray-200 px-1 rounded">公司平台</code>{' '}
            <code className="bg-gray-200 px-1 rounded">部门</code>{' '}
            <code className="bg-gray-200 px-1 rounded">工作地点</code>{' '}
            <code className="bg-gray-200 px-1 rounded">核心业务</code>{' '}
            <code className="bg-gray-200 px-1 rounded">岗位名</code>{' '}
            <code className="bg-gray-200 px-1 rounded">岗位JD</code>{' '}
            <code className="bg-gray-200 px-1 rounded">人选画像</code>
          </p>
          <label className="btn-secondary cursor-pointer inline-flex">
            {importing ? '⏳ 导入中...' : '📂 选择 Excel 文件'}
            <input type="file" accept=".xlsx,.xls" className="hidden"
              onChange={(e) => handleImportExcel(e.target.files[0])} disabled={importing} />
          </label>
          <button onClick={() => setShowImport(false)} className="btn-secondary ml-2">关闭</button>
        </div>
      )}

      {/* 岗位列表 */}
      {loading && <div className="text-center py-8 text-gray-400">⏳ 加载中...</div>}

      {!loading && jobs.length === 0 && (
        <div className="card text-center py-12 text-gray-400">
          <div className="text-4xl mb-3">📭</div>
          <p>暂无岗位数据</p>
          <p className="text-sm mt-1">点击「手动录入」或「Excel 导入」添加岗位</p>
        </div>
      )}

      <div className="space-y-2">
        {jobs.map((job) => (
          <div key={job.id} className={`card flex items-start gap-3 transition ${selectedIds.includes(job.id) ? 'ring-2 ring-primary-300 bg-primary-50/20' : ''}`}>
            <input type="checkbox" className="mt-1.5 shrink-0"
              checked={selectedIds.includes(job.id)}
              onChange={() => toggleSelect(job.id)} />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="font-bold text-gray-900">{job.title}</h3>
                <span className="text-xs text-gray-400">ID:{job.id}</span>
              </div>
              <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1 text-xs text-gray-500">
                {job.company_name && <span>🏢 {job.company_name}</span>}
                {job.platform && <span>📱 {job.platform}</span>}
                {job.location && <span>📍 {job.location}</span>}
              </div>
              <p className="text-sm text-gray-600 mt-2 line-clamp-2">{job.jd_text}</p>
            </div>
            <button
              onClick={() => {
                sessionStorage.setItem('selectedJobId', job.id)
                navigate('/reverse')
              }}
              className="btn-secondary text-xs shrink-0"
            >
              🔍 匹配候选人
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}