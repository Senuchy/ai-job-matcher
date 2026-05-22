import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useState } from 'react'

const navItems = [
  { path: '/match', label: '简历匹配岗位', icon: '📄', desc: '上传简历找岗位' },
  { path: '/jobs', label: '岗位库管理', icon: '📁', desc: '录入与管理岗位' },
  { path: '/resumes', label: '简历库管理', icon: '📇', desc: '录入与管理简历' },
  { path: '/reverse', label: '岗位匹配候选人', icon: '🔄', desc: '选岗位找人才' },
]

export default function Layout() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()

  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* 侧边栏 */}
      <aside className={`${collapsed ? 'w-16' : 'w-64'} bg-white border-r border-gray-200 flex flex-col transition-all duration-300 ease-in-out relative z-10`}>
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 h-16 border-b border-gray-100 shrink-0 relative">
          <span className="text-2xl">🤝</span>
          {!collapsed && <span className="font-bold text-lg text-gray-800 whitespace-nowrap">招聘AI助手</span>}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className={`absolute -right-3 top-5 z-30 bg-white rounded-full shadow-md border border-gray-200 w-6 h-6 flex items-center justify-center text-gray-500 hover:text-gray-700 transition-all`}
            title={collapsed ? '展开' : '收起'}
          >
            {collapsed ? '→' : '←'}
          </button>
        </div>

        {/* 导航 */}
        <nav className="flex-1 py-4 space-y-1 px-2">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-0 rounded-lg text-sm font-medium transition-all h-14 ${
                  isActive
                    ? 'bg-primary-50 text-primary-700 shadow-sm'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`}
                title={collapsed ? item.label : undefined}
              >
                <span className="text-lg shrink-0">{item.icon}</span>
                {!collapsed && (
                  <div className="min-w-0 flex-1">
                    <div className="truncate leading-tight">{item.label}</div>
                    <div className="text-xs text-gray-400 truncate leading-tight mt-0.5">{item.desc}</div>
                  </div>
                )}
              </NavLink>
            )
          })}
        </nav>

        {/* 底部 - 始终显示，但禁止折行，超出部分隐藏 */}
        <div className="px-4 py-3 border-t border-gray-100 text-xs text-gray-400 whitespace-nowrap overflow-hidden">
           v2.0  简历岗位匹配 AI 助手
        </div>
      </aside>

      {/* 主内容区 */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}