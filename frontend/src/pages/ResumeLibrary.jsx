import { useState, useEffect, useCallback } from 'react'
import { getResumes, createResume, importResumesExcel, deleteResumes } from '../api/client'

export default function ResumeLibrary() {
  const [resumes, setResumes] = useState([])
  const [filters, setFilters] = useState({ name: '', phone: '', email: '' })
  const [selectedIds, setSelectedIds] = useState([])
  const [showForm, setShowForm] = useState(false)
  const [showImport, setShowImport] = useState(false)
  const [loading, setLoading] = useState(false)
  const [importing, setImporting] = useState(false)
  const [newResume, setNewResume] = useState({
    name: '', text: '', phone: '', email: '', education: ''
  })

  const loadResumes = useCallback(async () => {
    setLoading(true)
    try {
      const params = {}
      if (filters.name) params.name = filters.name
      if (filters.phone) params.phone = filters.phone
      if (filters.email) params.email = filters.email
      const { data } = await getResumes(params)
      setResumes(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [filters])

  useEffect(() => { loadResumes() }, [loadResumes])

  const handleCreate = async () => {
    if (!newResume.name.trim() || !newResume.text.trim()) {
      alert('姓名和简历正文为必填项')
      return
    }
    try {
      await createResume(newResume)
      setShowForm(false)
      setNewResume({ name: '', text: '', phone: '', email: '', education: '' })
      loadResumes()
    } catch (err) {
      alert('创建失败: ' + (err.response?.data?.detail || err.message))
    }
  }

  const handleImportExcel = async (file) => {
    if (!file) return
    setImporting(true)
    try {
      const { data } = await importResumesExcel(file)
      alert(`导入完成！成功 ${data.success} / 共 ${data.total} 条`)
      loadResumes()
    } catch (err) {
      alert('导入失败: ' + (err.response?.data?.detail || err.message))
    } finally {
      setImporting(false)
    }
  }

  const handleDelete = async () => {
    if (selectedIds.length === 0) return
    if (!confirm(`确定删除选中的 ${selectedIds.length} 条简历吗？`)) return
    try {
      await deleteResumes(selectedIds)
      setSelectedIds([])
      loadResumes()
    } catch (err) {
      alert('删除失败')
    }
  }

  const toggleSelect = (id) => {
    setSelectedIds((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])
  }

  const toggleSelectAll = () => {
    if (selectedIds.length === resumes.length) setSelectedIds([])
    else setSelectedIds(resumes.map((r) => r.id))
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">📇 简历库管理</h1>
          <p className="text-sm text-gray-500 mt-1">管理所有候选人简历，支持筛选、录入、导入和批量删除</p>
        </div>
        <span className="badge-blue text-sm">{resumes.length} 份简历</span>
      </div>

      {/* 筛选栏 */}
      <div className="card">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <input className="input" placeholder="🔍 姓名" value={filters.name}
            onChange={(e) => setFilters({ ...filters, name: e.target.value })} />
          <input className="input" placeholder="🔍 电话" value={filters.phone}
            onChange={(e) => setFilters({ ...filters, phone: e.target.value })} />
          <input className="input" placeholder="🔍 邮箱" value={filters.email}
            onChange={(e) => setFilters({ ...filters, email: e.target.value })} />
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
        {resumes.length > 0 && (
          <label className="btn-secondary cursor-pointer ml-auto">
            <input type="checkbox" className="mr-1" checked={selectedIds.length === resumes.length && resumes.length > 0}
              onChange={toggleSelectAll} />
            全选
          </label>
        )}
      </div>

      {/* 手动录入表单 */}
      {showForm && (
        <div className="card border-primary-200 bg-primary-50/30">
          <h3 className="font-bold text-gray-800 mb-4">📝 新增简历</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">姓名 *</label>
              <input className="input" placeholder="候选人姓名" value={newResume.name}
                onChange={(e) => setNewResume({ ...newResume, name: e.target.value })} />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">电话</label>
              <input className="input" placeholder="手机号" value={newResume.phone}
                onChange={(e) => setNewResume({ ...newResume, phone: e.target.value })} />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">邮箱</label>
              <input className="input" placeholder="email@example.com" value={newResume.email}
                onChange={(e) => setNewResume({ ...newResume, email: e.target.value })} />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">学历背景</label>
              <input className="input" placeholder="如：北京大学·计算机科学·2020届" value={newResume.education}
                onChange={(e) => setNewResume({ ...newResume, education: e.target.value })} />
            </div>
            <div className="sm:col-span-2">
              <label className="text-xs text-gray-500 mb-1 block">简历正文 *</label>
              <textarea className="input resize-none" rows={6} placeholder="粘贴完整的简历内容，包括工作经历、技能、项目经验等..."
                value={newResume.text} onChange={(e) => setNewResume({ ...newResume, text: e.target.value })} />
            </div>
          </div>
          <div className="flex gap-2 mt-4">
            <button onClick={handleCreate} className="btn-primary">💾 保存简历</button>
            <button onClick={() => setShowForm(false)} className="btn-secondary">取消</button>
          </div>
        </div>
      )}

      {/* Excel 导入 */}
      {showImport && (
        <div className="card border-blue-200 bg-blue-50/30">
          <h3 className="font-bold text-gray-800 mb-2">📊 Excel 批量导入</h3>
          <p className="text-sm text-gray-500 mb-3">
            Excel 需包含列：<code className="bg-gray-200 px-1 rounded">姓名</code>{' '}
            <code className="bg-gray-200 px-1 rounded">电话</code>{' '}
            <code className="bg-gray-200 px-1 rounded">邮箱</code>{' '}
            <code className="bg-gray-200 px-1 rounded">学历背景</code>{' '}
            <code className="bg-gray-200 px-1 rounded">简历正文</code>
          </p>
          <label className="btn-secondary cursor-pointer inline-flex">
            {importing ? '⏳ 导入中...' : '📂 选择 Excel 文件'}
            <input type="file" accept=".xlsx,.xls" className="hidden"
              onChange={(e) => handleImportExcel(e.target.files[0])} disabled={importing} />
          </label>
          <button onClick={() => setShowImport(false)} className="btn-secondary ml-2">关闭</button>
        </div>
      )}

      {/* 简历列表 */}
      {loading && <div className="text-center py-8 text-gray-400">⏳ 加载中...</div>}

      {!loading && resumes.length === 0 && (
        <div className="card text-center py-12 text-gray-400">
          <div className="text-4xl mb-3">📭</div>
          <p>暂无简历数据</p>
          <p className="text-sm mt-1">点击「手动录入」或「Excel 导入」添加简历</p>
        </div>
      )}

      <div className="space-y-2">
        {resumes.map((r) => (
          <div key={r.id} className={`card flex items-start gap-3 transition ${selectedIds.includes(r.id) ? 'ring-2 ring-primary-300 bg-primary-50/20' : ''}`}>
            <input type="checkbox" className="mt-1.5 shrink-0"
              checked={selectedIds.includes(r.id)}
              onChange={() => toggleSelect(r.id)} />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="font-bold text-gray-900">👤 {r.name}</h3>
                <span className="text-xs text-gray-400">ID:{r.id}</span>
              </div>
              <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1 text-xs text-gray-500">
                {r.phone && <span>📞 {r.phone}</span>}
                {r.email && <span>✉️ {r.email}</span>}
                {r.education && <span>🎓 {r.education}</span>}
              </div>
              <p className="text-sm text-gray-600 mt-2 line-clamp-2">{r.text}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}