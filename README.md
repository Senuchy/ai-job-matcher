# AI岗位简历匹配助手

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

基于腾讯混元大模型的智能招聘匹配工具，通过向量检索与 AI 多维度评分，帮助高效匹配候选人与岗位，适用社招招聘（可能），碎碎念放在最底下。

目前还是一个MVP状态，什么都很简单，只是实现了ai调用的核心匹配功能，我称之为1.0.0核心版本。

## 功能亮点

-  **岗位管理**：支持岗位的录入、向量化存储、查看和删除（目前是本地存储，支持体量不大的个人用户）
-  **简历解析**：支持 PDF/Word 文件上传，也可直接粘贴纯文本（缓冲很慢，还在调整）
-  **向量初筛**：基于混元 Embedding 模型计算语义相似度。（调用）
-  **AI 严格评分**：五维度（工作经验/核心技能/教育背景/项目成就/稳定性）综合评估（提示词严格设定了一部分评分参考，可自行优化）
-  **并发优化**：Top3 岗位并发调用 AI，大幅缩短等待时间（没评估并发快了夺少）

## 🛠 技术栈

| 层级 | 技术 |
|------|------|
| 前端界面 | Streamlit |
| 向量嵌入 | 腾讯混元 Embedding API |
| AI 评分 | 腾讯混元 Chat API (hunyuan-2.0-instruct) |
| 数据库 | SQLite3 |
| 文件解析 | PyPDF2、python-docx |
| 并发处理 | ThreadPoolExecutor |

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 腾讯云 API 密钥（需开通混元大模型服务）
- TokenHub API 密钥

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/你的用户名/hunter-ai-matcher.git
cd hunter-ai-matcher
```
2. 安装依赖
```bash
pip install -r requirements.txt
```
3. 配置环境变量
```bash
cp .env.example .env
#编辑.env文件，填入你的API密钥
```
4. 启动应用
```bash
streamlit run app.py
```
## 使用说明
1. 在左侧边栏录入岗位名称和职位描述
2. 选择上传简历文件或直接粘贴简历文本
3. 点击匹配后，AI 将对向量相似度最高的前3 个岗位进行五维度严格评分
4. 点击结果卡片可展开查看详细维度分析

##  项目架构
```text
用户界面 (Streamlit)
    ├── 岗位管理 (SQLite CRUD)
    ├── 简历输入 (文件上传 / 文本粘贴)
    └── 匹配引擎
         ├── 文本提取 → Embedding API → 向量
         ├── 向量相似度排序 → Top5 初筛
         └── AI 并发评分 (Top3)
              └── 结果展示 (评分/理由/维度分析)
```
##  待办事项
- 支持简历历史存储与复用
- 添加批量导入岗位功能
- 支持自定义评分维度权重
- ···
- 还在思考，应该会重新做简历库和岗位库部分，更换数据库技术栈之类的，可能还会分开招聘端和个人求职端，再分别更新可能需求。

## 碎碎念
其实暂时还没帮自己工作什么，暂时是只在看不懂简历（招聘端），以及纠结自己投什么岗位（求职端）的时候，突发奇想想要做一个帮帮忙。但是优化功能可能还需要走一段路程。需要解决的问题还有很多，比如如何花最少的token给出最精准的答案/减少时间/服务器部署否/数据库租用否...之类的

##  贡献
欢迎提交 Issue 和 Pull Request！

##  许可证
本项目采用 MIT License 开源。