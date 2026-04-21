import streamlit as st
import sqlite3
import numpy as np
import os
import PyPDF2
from docx import Document
import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- 页面配置 ---
st.set_page_config(page_title="AI匹配助手", layout="wide")
st.title("🤝 AI匹配助手（AI智能评分·严格版）")

# --- 初始化腾讯云 Embedding 客户端（用于向量）---
def get_embedding(text):
    secret_id = os.environ.get("TENCENT_SECRET_ID")
    secret_key = os.environ.get("TENCENT_SECRET_KEY")
    if not secret_id or not secret_key:
        st.error("请设置环境变量 TENCENT_SECRET_ID 和 TENCENT_SECRET_KEY")
        st.stop()
    
    cred = credential.Credential(secret_id, secret_key)
    httpProfile = HttpProfile()
    httpProfile.endpoint = "hunyuan.tencentcloudapi.com"
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    client = hunyuan_client.HunyuanClient(cred, "", clientProfile)
    
    req = models.GetEmbeddingRequest()
    truncated_text = text[:1024]  # 混元限制 1024 token
    req.Input = truncated_text
    resp = client.GetEmbedding(req)
    embedding = resp.Data[0].Embedding
    return embedding

# --- 初始化 TokenHub 客户端（用于 AI 评分）---
tokenhub_api_key = os.environ.get("TOKENHUB_API_KEY")
if tokenhub_api_key:
    chat_client = OpenAI(
        api_key=tokenhub_api_key,
        base_url="https://tokenhub.tencentmaas.com/v1",
        timeout=15.0  # 适当延长超时，避免并发时超时
    )
else:
    chat_client = None
    st.warning("未设置 TOKENHUB_API_KEY，将只显示向量相似度，无法使用 AI 评分。")

# --- 余弦相似度 ---
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- 解析简历文件 ---
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text
    return text

# --- 智能文本截断函数（保留头尾关键信息）---
def smart_truncate(text, max_len=3000):
    """取前70%和后30%内容，保证开头和结尾的关键经历不丢失"""
    if len(text) <= max_len:
        return text
    head_len = int(max_len * 0.7)
    tail_len = max_len - head_len
    return text[:head_len] + "\n...(中间内容省略)...\n" + text[-tail_len:]

# --- AI 评分函数（调用混元 chat 模型，严格评估）---
def evaluate_match_with_ai(job_title, jd_text, resume_text):
    if not chat_client:
        return None, "未配置 AI 评分服务", ""
    
    # 智能截取，适当增加长度以保留更多细节
    jd_preview = smart_truncate(jd_text, 3200)
    resume_preview = smart_truncate(resume_text, 3200)
    
    # 【核心修改】严格的五维度评估提示词
    prompt = f"""你是一位资深猎头顾问，需要严格、客观地评估候选人与职位的匹配度。评分必须基于事实，不可主观臆断，高分必须满足硬性要求。

请按照以下维度逐项分析，最后给出综合评分（0-10分，保留1位小数，10分仅限完美匹配且无任何短板）：

【职位名称】{job_title}
【职位描述（关键部分）】
{jd_preview}

【候选人简历（关键部分）】
{resume_preview}

### 评估维度与权重（共100%）：
1. **工作经验匹配度（40%）**：行业领域、职位层级、工作年限是否匹配？职责是否高度相关？缺失关键经验需大幅扣分。
2. **核心技能匹配度（35%）**：职位要求的技术栈、工具、语言能力、证书等是否具备？每缺一项扣分。
3. **教育背景与资质（10%）**：学历专业是否符合？有相关认证加分，不符则扣分。
4. **项目/成就含金量（10%）**：候选人项目复杂度、成果影响力是否达到职位要求？
5. **稳定性与职业路径（5%）**：跳槽频率是否合理？职业规划是否与职位方向一致？

### 严格评分指引：
- **9-10分**：所有硬性要求完全满足，且有多项超出预期，可直接录用。
- **7-8.9分**：核心要求基本满足，仅有非关键技能缺失或行业稍有偏差。
- **5-6.9分**：部分匹配，但存在明显短板（如年限不足、关键技能缺失），需面试深挖。
- **3-4.9分**：匹配度低，勉强触及岗位边缘要求。
- **0-2.9分**：基本不匹配，方向差异大。

### 输出格式（严格遵守，便于程序解析）：
【维度分析】
- 工作经验：<具体匹配点与差距>
- 核心技能：<具体匹配点与差距>
- 教育背景：<评价>
- 项目成就：<评价>
- 稳定性：<评价>

【综合评分】X.X
【评分理由】<一句话总结最核心的匹配或硬伤，30字内>
"""
    
    try:
        response = chat_client.chat.completions.create(
            model="hunyuan-2.0-instruct-20251111",  # 保持不变
            messages=[
                {"role": "system", "content": "你是专业猎头，严格遵循评估指引，只输出指定格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,   # 保持不变
            stream=False
        )
        result = response.choices[0].message.content.strip()
        
        # 解析新格式
        score = None
        reason = "解析失败"
        dimensions = ""
        
        # 先尝试按新格式解析
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith('【综合评分】'):
                try:
                    score = float(line.replace('【综合评分】', '').strip())
                except:
                    score = 0.0
            elif line.startswith('【评分理由】'):
                reason = line.replace('【评分理由】', '').strip()
            elif line.startswith('【维度分析】') or line.startswith('- 工作经验') or line.startswith('- 核心技能'):
                dimensions += line + "\n"
        
        # 如果新格式解析失败，回退到旧格式（兼容）
        if score is None:
            for line in result.split('\n'):
                if '评分：' in line or '评分:' in line:
                    try:
                        score = float(line.split('：')[-1].split(':')[-1].strip())
                    except:
                        score = 0.0
                elif '理由：' in line or '理由:' in line:
                    reason = line.split('：')[-1].split(':')[-1].strip()
            if score is None:
                score = 0.0
                reason = result[:100]
        
        # 返回分数、理由和维度分析（可用于前端展示）
        return score, reason, dimensions
    except Exception as e:
        return None, f"调用失败：{str(e)}", ""

# --- SQLite 初始化 ---
conn = sqlite3.connect('hunter.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS jobs
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              title TEXT,
              jd_text TEXT,
              embedding TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS resumes
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT,
              text TEXT,
              embedding TEXT)''')
conn.commit()

# --- 侧边栏：岗位管理（新增删除功能）---
with st.sidebar:
    st.header("📋 岗位管理")
    job_title = st.text_input("岗位名称")
    jd_text = st.text_area("职位描述 (JD)", height=200)
    
    if st.button("➕ 录入岗位"):
        if job_title and jd_text:
            with st.spinner("生成岗位向量中..."):
                embedding = get_embedding(jd_text)
                c.execute("INSERT INTO jobs (title, jd_text, embedding) VALUES (?, ?, ?)",
                          (job_title, jd_text, json.dumps(embedding)))
                conn.commit()
            st.success(f"岗位 [{job_title}] 已录入！")
            st.rerun()
        else:
            st.error("请填写完整信息")
    
    st.divider()
    st.subheader("📌 已录入岗位")
    jobs = c.execute("SELECT id, title FROM jobs").fetchall()
    if not jobs:
        st.info("暂无岗位，请先录入")
    else:
        for job_id, title in jobs:
            # 每行显示岗位名称和删除按钮
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"- {title}")
            with col2:
                # 为每个删除按钮设置唯一的 key，避免冲突
                if st.button("🗑️", key=f"del_{job_id}", help=f"删除 {title}"):
                    c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                    conn.commit()
                    st.success(f"岗位 [{title}] 已删除")
                    st.rerun()

# --- 主区域：简历匹配 ---
st.header("📄 简历匹配")

# 【新增】选择简历输入方式
input_mode = st.radio(
    "请选择简历输入方式：",
    ("📁 上传文件 (PDF/Word)", "✏️ 直接粘贴简历文本"),
    horizontal=True
)

resume_text = None

if input_mode == "📁 上传文件 (PDF/Word)":
    uploaded_file = st.file_uploader("上传候选人简历", type=['pdf', 'docx'])
    if uploaded_file is not None:
        resume_text = extract_text_from_file(uploaded_file)
        if not resume_text:
            st.error("无法提取文本，请检查文件格式")
            st.stop()
else:
    resume_text = st.text_area(
        "请在此粘贴简历内容（纯文本）",
        height=300,
        placeholder="例如：\n姓名：张三\n工作经历：...\n技能：..."
    )
    if resume_text:
        # 简单清理多余空白
        resume_text = resume_text.strip()

# 只有获得了简历文本才继续执行匹配
if resume_text:
    st.success("简历内容已就绪！正在匹配...")
    
    with st.spinner("生成简历向量..."):
        resume_embedding = get_embedding(resume_text)
    
    # 获取所有岗位
    jobs = c.execute("SELECT id, title, jd_text, embedding FROM jobs").fetchall()
    if not jobs:
        st.warning("暂无岗位，请先在左侧录入")
        st.stop()
    
    # 1. 向量相似度初筛
    vector_results = []
    for job_id, title, jd_text, emb_json in jobs:
        job_embedding = json.loads(emb_json)
        sim = cosine_similarity(resume_embedding, job_embedding)
        vector_results.append((job_id, title, jd_text, sim))
    vector_results.sort(key=lambda x: x[3], reverse=True)
    top_candidates = vector_results[:5]  # 取前5个候选，但AI仅评前3
    
    # 2. AI 深度评分（如果可用）
    if chat_client:
        st.info("🤖 AI 正在严格评估匹配度，请稍候...")
        
        # 只对 Top3 进行 AI 评分
        top3_candidates = top_candidates[:3]
        
        # 定义并发任务函数
        def ai_score_job(job_tuple):
            job_id, title, jd_text, sim = job_tuple
            score, reason, dimensions = evaluate_match_with_ai(title, jd_text, resume_text)
            if score is None:
                # 如果AI调用失败，回退到向量相似度（归一化到0-10）
                fallback_score = int(sim * 10)
                return (title, jd_text, fallback_score, f"AI调用失败，回退分数 ({reason})", sim, "")
            else:
                return (title, jd_text, score, reason, sim, dimensions)
        
        ai_results = []
        start_time = time.time()
        
        # 并发执行评分
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交所有任务
            future_to_job = {executor.submit(ai_score_job, job): job for job in top3_candidates}
            
            # 显示进度（由于并发，进度条无法准确反映单个进度，改用状态文本）
            progress_bar = st.progress(0)
            completed = 0
            total = len(future_to_job)
            
            for future in as_completed(future_to_job):
                result = future.result()
                ai_results.append(result)
                completed += 1
                progress_bar.progress(completed / total)
        
        elapsed = time.time() - start_time
        st.caption(f"⏱️ AI 评分总耗时：{elapsed:.2f} 秒")
        
        # 按AI评分排序
        ai_results.sort(key=lambda x: x[2], reverse=True)
        
        st.subheader("🏆 推荐岗位（AI严格评分前3）")
        for title, jd_text, score, reason, sim, dimensions in ai_results:
            # 决定颜色标识
            if score >= 7:
                score_display = f":green[{score:.1f}/10]"
            elif score >= 5:
                score_display = f":orange[{score:.1f}/10]"
            else:
                score_display = f":red[{score:.1f}/10]"
            
            with st.expander(f"**{title}** — AI评分：{score:.1f}/10  (向量相似度：{sim:.1%})", expanded=True):
                st.markdown(f"**🎯 AI严格评分：** {score_display}")
                st.markdown(f"**📌 匹配理由：** {reason}")
                if dimensions:
                    with st.expander("🔍 查看详细维度分析"):
                        st.text(dimensions)
                st.markdown(f"**📄 职位描述摘要：** {jd_text[:200]}...")
        
        # 如果总岗位数>3，展示剩余岗位的向量相似度
        if len(vector_results) > 3:
            with st.expander("📋 其他备选岗位（仅向量相似度）"):
                for job_id, title, jd_text, sim in vector_results[3:]:
                    st.markdown(f"**{title}** — 向量相似度：{sim:.1%}")
                    st.caption(f"职位描述：{jd_text[:150]}...")
    else:
        # 无AI评分时，仅显示向量相似度
        st.subheader("📊 匹配结果（仅向量相似度）")
        for job_id, title, jd_text, sim in top_candidates[:3]:
            with st.expander(f"**{title}** 相似度：{sim:.2%}"):
                st.write(f"**职位描述摘要：** {jd_text[:300]}...")