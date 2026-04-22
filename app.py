import streamlit as st
import sqlite3
import numpy as np
import os
import PyPDF2
from docx import Document
import json
import pandas as pd
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- 页面配置 ---
st.set_page_config(page_title="ai-matcher-assistant", layout="wide")

# --- 初始化 Session State ---
if "page" not in st.session_state:
    st.session_state.page = "match"  # 'match', 'job_lib', 'resume_lib', 'job_to_candidates'
if "selected_job_for_match" not in st.session_state:
    st.session_state.selected_job_for_match = None

# --- 初始化腾讯云 Embedding 客户端 ---
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
    truncated_text = text[:1024]
    req.Input = truncated_text
    resp = client.GetEmbedding(req)
    embedding = resp.Data[0].Embedding
    return embedding

# --- 初始化 TokenHub 客户端（AI 评分）---
tokenhub_api_key = os.environ.get("TOKENHUB_API_KEY")
if tokenhub_api_key:
    chat_client = OpenAI(
        api_key=tokenhub_api_key,
        base_url="https://tokenhub.tencentmaas.com/v1",
        timeout=15.0
    )
else:
    chat_client = None

# --- 余弦相似度 ---
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- 解析简历文件（用于匹配页面）---
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

# --- 智能文本截断 ---
def smart_truncate(text, max_len=3000):
    if len(text) <= max_len:
        return text
    head_len = int(max_len * 0.7)
    tail_len = max_len - head_len
    return text[:head_len] + "\n...(中间内容省略)...\n" + text[-tail_len:]

# --- AI 评分函数（简历 -> 岗位）---
def evaluate_match_with_ai(job_title, jd_text, resume_text):
    if not chat_client:
        return None, "未配置 AI 评分服务", ""
    jd_preview = smart_truncate(jd_text, 3200)
    resume_preview = smart_truncate(resume_text, 3200)
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
            model="hunyuan-2.0-instruct-20251111",
            messages=[
                {"role": "system", "content": "你是专业猎头，严格遵循评估指引，只输出指定格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            stream=False
        )
        result = response.choices[0].message.content.strip()
        score = None
        reason = "解析失败"
        dimensions = ""
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
        return score, reason, dimensions
    except Exception as e:
        return None, f"调用失败：{str(e)}", ""

# --- SQLite 初始化（含字段扩展与迁移）---
conn = sqlite3.connect('hunter.db', check_same_thread=False)
c = conn.cursor()

# 岗位表
c.execute('''CREATE TABLE IF NOT EXISTS jobs
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              title TEXT,
              jd_text TEXT,
              embedding TEXT)''')
c.execute("PRAGMA table_info(jobs)")
existing_job_cols = [col[1] for col in c.fetchall()]
job_fields = {
    "company_name": "TEXT",
    "platform": "TEXT",
    "department": "TEXT",
    "location": "TEXT",
    "core_business": "TEXT",
    "candidate_profile": "TEXT"
}
for field, dtype in job_fields.items():
    if field not in existing_job_cols:
        c.execute(f"ALTER TABLE jobs ADD COLUMN {field} {dtype}")

# 简历表
c.execute('''CREATE TABLE IF NOT EXISTS resumes
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT,
              text TEXT,
              embedding TEXT)''')
c.execute("PRAGMA table_info(resumes)")
existing_resume_cols = [col[1] for col in c.fetchall()]
resume_fields = {
    "phone": "TEXT",
    "email": "TEXT",
    "education": "TEXT"
}
for field, dtype in resume_fields.items():
    if field not in existing_resume_cols:
        c.execute(f"ALTER TABLE resumes ADD COLUMN {field} {dtype}")

conn.commit()

# --- 侧边栏导航 ---
with st.sidebar:
    st.title("🤝 AI匹配助手（严格评分版）")
    st.divider()
    if st.button("📄 简历匹配岗位", use_container_width=True):
        st.session_state.page = "match"
        st.session_state.selected_job_for_match = None
        st.rerun()
    if st.button("📁 岗位库管理", use_container_width=True):
        st.session_state.page = "job_lib"
        st.rerun()
    if st.button("📇 简历库管理", use_container_width=True):
        st.session_state.page = "resume_lib"
        st.rerun()
    st.divider()
    if not chat_client:
        st.warning("未设置 TOKENHUB_API_KEY，AI评分不可用。")

# --- 页面渲染 ---

# ---------- 简历匹配岗位页面（原有逻辑）----------
if st.session_state.page == "match":
    st.title("📄 简历智能匹配岗位")
    input_mode = st.radio(
        "请选择简历输入方式：",
        ("📁 上传文件 (PDF/Word)", "✏️ 直接粘贴简历文本", "📇 从简历库选择"),
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
    elif input_mode == "✏️ 直接粘贴简历文本":
        resume_text = st.text_area(
            "请在此粘贴简历内容（纯文本）",
            height=300,
            placeholder="例如：\n姓名：张三\n工作经历：...\n技能：..."
        )
        if resume_text:
            resume_text = resume_text.strip()
    else:  # 从简历库选择
        resumes = c.execute("SELECT id, name, text FROM resumes ORDER BY id DESC").fetchall()
        if not resumes:
            st.warning("简历库暂无简历，请先录入")
            st.stop()
        resume_options = {f"{r[1]} (ID:{r[0]})": (r[0], r[2]) for r in resumes}
        selected_label = st.selectbox("选择简历", options=list(resume_options.keys()))
        if selected_label:
            resume_text = resume_options[selected_label][1]

    if resume_text:
        st.success("简历内容已就绪！")
        # 显示简历预览（可选）
        with st.expander("📄 简历内容预览"):
            st.text(resume_text[:500] + ("..." if len(resume_text) > 500 else ""))

        # 匹配按钮
        if st.button("🔍 开始匹配岗位", type="primary", use_container_width=True):
            with st.spinner("生成简历向量..."):
                resume_embedding = get_embedding(resume_text)

            jobs = c.execute("SELECT id, title, jd_text, embedding FROM jobs").fetchall()
            if not jobs:
                st.warning("暂无岗位，请先在岗位库中添加")
                st.stop()

            vector_results = []
            for job_id, title, jd_text, emb_json in jobs:
                if emb_json:
                    job_embedding = json.loads(emb_json)
                    sim = cosine_similarity(resume_embedding, job_embedding)
                    vector_results.append((job_id, title, jd_text, sim))
            vector_results.sort(key=lambda x: x[3], reverse=True)
            top_candidates = vector_results[:5]

            if chat_client:
                st.info("🤖 AI 正在严格评估匹配度，请稍候...")
                top3_candidates = top_candidates[:3]
                def ai_score_job(job_tuple):
                    job_id, title, jd_text, sim = job_tuple
                    score, reason, dimensions = evaluate_match_with_ai(title, jd_text, resume_text)
                    if score is None:
                        fallback_score = int(sim * 10)
                        return (title, jd_text, fallback_score, f"AI调用失败，回退分数 ({reason})", sim, "")
                    else:
                        return (title, jd_text, score, reason, sim, dimensions)
                ai_results = []
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_job = {executor.submit(ai_score_job, job): job for job in top3_candidates}
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
                ai_results.sort(key=lambda x: x[2], reverse=True)

                st.subheader("🏆 推荐岗位（AI严格评分前3）")
                for title, jd_text, score, reason, sim, dimensions in ai_results:
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

                if len(vector_results) > 3:
                    with st.expander("📋 其他备选岗位（仅向量相似度）"):
                        for job_id, title, jd_text, sim in vector_results[3:]:
                            st.markdown(f"**{title}** — 向量相似度：{sim:.1%}")
                            st.caption(f"职位描述：{jd_text[:150]}...")
            else:
                st.subheader("📊 匹配结果（仅向量相似度）")
                for job_id, title, jd_text, sim in top_candidates[:3]:
                    with st.expander(f"**{title}** 相似度：{sim:.2%}"):
                        st.write(f"**职位描述摘要：** {jd_text[:300]}...")
# ---------- 岗位库管理页面 ----------
elif st.session_state.page == "job_lib":
    st.title("📁 岗位库管理")

    # 筛选区域
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_company = st.text_input("🔍 按公司名称筛选")
        with col2:
            filter_platform = st.text_input("🔍 按平台筛选")
        with col3:
            filter_location = st.text_input("🔍 按工作地点筛选")

    st.divider()

    # 导入区域
    with st.expander("➕ 导入新岗位", expanded=False):
        tab1, tab2 = st.tabs(["📝 手动录入", "📊 Excel批量导入"])

        with tab1:
            with st.form("manual_job_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    company = st.text_input("公司名称")
                    platform = st.text_input("公司平台")
                    department = st.text_input("部门")
                    location = st.text_input("工作地点")
                    core_business = st.text_area("核心业务", height=100)
                with col_b:
                    title = st.text_input("岗位名称")
                    candidate_profile = st.text_area("人选画像", height=100)
                jd_text = st.text_area("岗位JD（详细描述）", height=200)
                submitted = st.form_submit_button("保存岗位")
                if submitted:
                    if not title or not jd_text:
                        st.error("岗位名称和JD为必填项")
                    else:
                        with st.spinner("生成向量中..."):
                            embedding = get_embedding(jd_text)
                            c.execute("""
                                INSERT INTO jobs 
                                (title, jd_text, embedding, company_name, platform, department, location, core_business, candidate_profile)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (title, jd_text, json.dumps(embedding), company, platform, department, location, core_business, candidate_profile))
                            conn.commit()
                        st.success("岗位已保存！")
                        st.rerun()

        with tab2:
            st.markdown("""
            **Excel 模板要求**：文件需包含以下列名（顺序不限）：
            `公司名`、`公司平台`、`部门`、`工作地点`、`核心业务`、`岗位名`、`岗位JD`、`人选画像`
            """)
            uploaded_excel = st.file_uploader("上传Excel文件", type=["xlsx", "xls"])
            if uploaded_excel:
                try:
                    df = pd.read_excel(uploaded_excel)
                    required_cols = ["公司名", "公司平台", "部门", "工作地点", "核心业务", "岗位名", "岗位JD", "人选画像"]
                    if not all(col in df.columns for col in required_cols):
                        st.error(f"Excel必须包含以下列：{', '.join(required_cols)}")
                    else:
                        st.dataframe(df.head(), use_container_width=True)
                        if st.button("确认导入", type="primary"):
                            success_count = 0
                            progress_bar = st.progress(0)
                            total = len(df)
                            for i, row in df.iterrows():
                                try:
                                    embedding = get_embedding(str(row["岗位JD"]))
                                    c.execute("""
                                        INSERT INTO jobs 
                                        (title, jd_text, embedding, company_name, platform, department, location, core_business, candidate_profile)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        row["岗位名"],
                                        row["岗位JD"],
                                        json.dumps(embedding),
                                        row.get("公司名", ""),
                                        row.get("公司平台", ""),
                                        row.get("部门", ""),
                                        row.get("工作地点", ""),
                                        row.get("核心业务", ""),
                                        row.get("人选画像", "")
                                    ))
                                    success_count += 1
                                except Exception as e:
                                    st.warning(f"第{i+2}行导入失败: {e}")
                                progress_bar.progress((i+1)/total)
                            conn.commit()
                            st.success(f"成功导入 {success_count} 条岗位")
                            st.rerun()
                except Exception as e:
                    st.error(f"读取Excel失败: {e}")

    # 查询并显示岗位列表（带筛选）
    query = "SELECT id, title, company_name, platform, location, jd_text FROM jobs WHERE 1=1"
    params = []
    if filter_company:
        query += " AND company_name LIKE ?"
        params.append(f"%{filter_company}%")
    if filter_platform:
        query += " AND platform LIKE ?"
        params.append(f"%{filter_platform}%")
    if filter_location:
        query += " AND location LIKE ?"
        params.append(f"%{filter_location}%")
    query += " ORDER BY id DESC"

    jobs_data = c.execute(query, params).fetchall()

    if not jobs_data:
        st.info("暂无岗位，请先导入")
    else:
        st.subheader(f"📋 岗位列表（共 {len(jobs_data)} 个）")

        # 多选删除
        job_options = {
            f"{job[2] or '未知公司'} - {job[1]} ({job[4] or '地点未填'}) [ID:{job[0]}]": job[0]
            for job in jobs_data
        }
        selected_labels = st.multiselect(
            "选择要删除的岗位（可多选）",
            options=list(job_options.keys()),
            key="delete_multiselect"
        )
        if selected_labels:
            if st.button("🗑️ 批量删除选中岗位", type="secondary"):
                ids_to_delete = [job_options[label] for label in selected_labels]
                c.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in ids_to_delete])
                conn.commit()
                st.success(f"已删除 {len(ids_to_delete)} 个岗位")
                st.rerun()

        st.divider()
        # 预览卡片
        for job in jobs_data:
            job_id, title, company, platform, location, jd_text = job
            company_display = company if company else "（未填公司）"
            platform_display = f" · {platform}" if platform else ""
            location_display = f" · 📍 {location}" if location else ""
            with st.expander(f"🏢 {company_display} — **{title}**{platform_display}{location_display}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**岗位ID:** {job_id}")
                with col2:
                    if st.button("🔍 匹配候选人", key=f"match_candidates_{job_id}"):
                        st.session_state.selected_job_for_match = job_id
                        st.session_state.page = "job_to_candidates"
                        st.rerun()
                st.markdown("**📄 岗位JD：**")
                st.text_area("JD详情", jd_text, height=200, key=f"jd_{job_id}", label_visibility="collapsed")
                extra = c.execute("SELECT department, core_business, candidate_profile FROM jobs WHERE id=?", (job_id,)).fetchone()
                if extra:
                    dept, core_biz, profile = extra
                    if dept or core_biz or profile:
                        st.markdown("**📌 附加信息：**")
                        if dept: st.caption(f"部门：{dept}")
                        if core_biz: st.caption(f"核心业务：{core_biz}")
                        if profile: st.caption(f"人选画像：{profile}")

# ---------- 简历库管理页面 ----------
elif st.session_state.page == "resume_lib":
    st.title("📇 简历库管理")

    # 筛选区域
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_name = st.text_input("🔍 按姓名筛选")
        with col2:
            filter_phone = st.text_input("🔍 按电话筛选")
        with col3:
            filter_email = st.text_input("🔍 按邮箱筛选")

    st.divider()

    # 导入区域
    with st.expander("➕ 导入新简历", expanded=False):
        tab1, tab2 = st.tabs(["📝 手动录入", "📊 Excel批量导入"])

        with tab1:
            with st.form("manual_resume_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    name = st.text_input("姓名")
                    phone = st.text_input("电话")
                    email = st.text_input("邮箱")
                with col_b:
                    education = st.text_input("学历背景（学校+专业+毕业年份）")
                resume_text = st.text_area("简历正文（详细工作经历、技能等）", height=250)
                submitted = st.form_submit_button("保存简历")
                if submitted:
                    if not name or not resume_text:
                        st.error("姓名和简历正文为必填项")
                    else:
                        with st.spinner("生成向量中..."):
                            embedding = get_embedding(resume_text)
                            c.execute("""
                                INSERT INTO resumes 
                                (name, text, embedding, phone, email, education)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (name, resume_text, json.dumps(embedding), phone, email, education))
                            conn.commit()
                        st.success("简历已保存！")
                        st.rerun()

        with tab2:
            st.markdown("""
            **Excel 模板要求**：文件需包含以下列名（顺序不限）：
            `姓名`、`电话`、`邮箱`、`学历背景`、`简历正文`
            """)
            uploaded_excel = st.file_uploader("上传Excel文件", type=["xlsx", "xls"], key="resume_excel")
            if uploaded_excel:
                try:
                    df = pd.read_excel(uploaded_excel)
                    required_cols = ["姓名", "电话", "邮箱", "学历背景", "简历正文"]
                    if not all(col in df.columns for col in required_cols):
                        st.error(f"Excel必须包含以下列：{', '.join(required_cols)}")
                    else:
                        st.dataframe(df.head(), use_container_width=True)
                        if st.button("确认导入", type="primary", key="confirm_resume_import"):
                            success_count = 0
                            progress_bar = st.progress(0)
                            total = len(df)
                            for i, row in df.iterrows():
                                try:
                                    embedding = get_embedding(str(row["简历正文"]))
                                    c.execute("""
                                        INSERT INTO resumes 
                                        (name, text, embedding, phone, email, education)
                                        VALUES (?, ?, ?, ?, ?, ?)
                                    """, (
                                        row["姓名"],
                                        row["简历正文"],
                                        json.dumps(embedding),
                                        row.get("电话", ""),
                                        row.get("邮箱", ""),
                                        row.get("学历背景", "")
                                    ))
                                    success_count += 1
                                except Exception as e:
                                    st.warning(f"第{i+2}行导入失败: {e}")
                                progress_bar.progress((i+1)/total)
                            conn.commit()
                            st.success(f"成功导入 {success_count} 条简历")
                            st.rerun()
                except Exception as e:
                    st.error(f"读取Excel失败: {e}")

    # 查询并显示简历列表
    query = "SELECT id, name, phone, email, education, text FROM resumes WHERE 1=1"
    params = []
    if filter_name:
        query += " AND name LIKE ?"
        params.append(f"%{filter_name}%")
    if filter_phone:
        query += " AND phone LIKE ?"
        params.append(f"%{filter_phone}%")
    if filter_email:
        query += " AND email LIKE ?"
        params.append(f"%{filter_email}%")
    query += " ORDER BY id DESC"

    resumes_data = c.execute(query, params).fetchall()

    if not resumes_data:
        st.info("暂无简历，请先导入")
    else:
        st.subheader(f"📋 简历列表（共 {len(resumes_data)} 个）")

        # 多选删除
        resume_options = {
            f"{r[1]} - {r[3] or '无电话'} - {r[4] or '无邮箱'} [ID:{r[0]}]": r[0]
            for r in resumes_data
        }
        selected_labels = st.multiselect(
            "选择要删除的简历（可多选）",
            options=list(resume_options.keys()),
            key="delete_resume_multiselect"
        )
        if selected_labels:
            if st.button("🗑️ 批量删除选中简历", type="secondary"):
                ids_to_delete = [resume_options[label] for label in selected_labels]
                c.executemany("DELETE FROM resumes WHERE id = ?", [(i,) for i in ids_to_delete])
                conn.commit()
                st.success(f"已删除 {len(ids_to_delete)} 条简历")
                st.rerun()

        st.divider()
        # 预览卡片
        for r in resumes_data:
            resume_id, name, phone, email, education, text = r
            with st.expander(f"👤 {name} — 📞 {phone or '未填'} — ✉️ {email or '未填'}"):
                st.markdown(f"**简历ID:** {resume_id}")
                if education:
                    st.markdown(f"**🎓 学历背景:** {education}")
                st.markdown("**📄 简历正文：**")
                st.text_area("简历详情", text, height=200, key=f"resume_{resume_id}", label_visibility="collapsed")

# ---------- 岗位匹配候选人页面（反向匹配）----------
elif st.session_state.page == "job_to_candidates":
    job_id = st.session_state.selected_job_for_match
    if not job_id:
        st.error("未选择岗位，请返回岗位库")
        st.stop()

    job = c.execute("SELECT title, jd_text, embedding FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not job:
        st.error("岗位不存在")
        st.stop()

    job_title, jd_text, job_emb_json = job
    job_embedding = json.loads(job_emb_json)

    st.title(f"🔍 为岗位匹配候选人")
    st.markdown(f"**岗位名称：** {job_title}")
    with st.expander("📄 岗位JD详情"):
        st.text(jd_text)

    # 检查简历库是否为空
    resumes_count = c.execute("SELECT COUNT(*) FROM resumes").fetchone()[0]
    if resumes_count == 0:
        st.warning("简历库暂无简历，请先录入")
        st.stop()

    # 匹配按钮
    if st.button("🔍 开始匹配候选人", type="primary", use_container_width=True):
        resumes = c.execute("SELECT id, name, text, embedding, phone, email, education FROM resumes").fetchall()

        with st.spinner("正在计算向量相似度..."):
            results = []
            for r in resumes:
                if r[3]:
                    resume_emb = json.loads(r[3])
                    sim = cosine_similarity(job_embedding, resume_emb)
                    results.append((r[0], r[1], r[2], sim, r[4], r[5], r[6]))
            results.sort(key=lambda x: x[3], reverse=True)
            top10 = results[:10]
            top3 = top10[:3]

        if chat_client:
            st.info("🤖 AI 正在严格评估候选人匹配度，请稍候...")
            def ai_score_candidate(cand):
                cid, name, text, sim, phone, email, edu = cand
                score, reason, dimensions = evaluate_match_with_ai(job_title, jd_text, text)
                if score is None:
                    fallback_score = int(sim * 10)
                    return (name, text, fallback_score, f"AI调用失败，回退分数 ({reason})", sim, phone, email, edu, "")
                else:
                    return (name, text, score, reason, sim, phone, email, edu, dimensions)

            ai_results = []
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_cand = {executor.submit(ai_score_candidate, c): c for c in top3}
                progress_bar = st.progress(0)
                completed = 0
                total = len(future_to_cand)
                for future in as_completed(future_to_cand):
                    result = future.result()
                    ai_results.append(result)
                    completed += 1
                    progress_bar.progress(completed / total)
            elapsed = time.time() - start_time
            st.caption(f"⏱️ AI 评分总耗时：{elapsed:.2f} 秒")
            ai_results.sort(key=lambda x: x[2], reverse=True)

            st.subheader("🏆 Top3 候选人（AI严格评分）")
            for name, text, score, reason, sim, phone, email, edu, dimensions in ai_results:
                if score >= 7:
                    score_display = f":green[{score:.1f}/10]"
                elif score >= 5:
                    score_display = f":orange[{score:.1f}/10]"
                else:
                    score_display = f":red[{score:.1f}/10]"
                with st.expander(f"👤 {name} — AI评分：{score:.1f}/10  (向量相似度：{sim:.1%})", expanded=True):
                    st.markdown(f"**📞 电话:** {phone or '未填'}  |  **✉️ 邮箱:** {email or '未填'}")
                    if edu:
                        st.markdown(f"**🎓 学历:** {edu}")
                    st.markdown(f"**🎯 AI严格评分：** {score_display}")
                    st.markdown(f"**📌 匹配理由：** {reason}")
                    if dimensions:
                        with st.expander("🔍 查看详细维度分析"):
                            st.text(dimensions)
                    st.markdown(f"**📄 简历摘要：** {text[:200]}...")

            if len(top10) > 3:
                with st.expander("📋 其他备选候选人（仅向量相似度 Top4-10）"):
                    for cand in top10[3:]:
                        cid, name, text, sim, phone, email, edu = cand
                        st.markdown(f"**{name}** — 向量相似度：{sim:.1%}  |  {phone or '无电话'}  |  {email or '无邮箱'}")
                        st.caption(f"简历摘要：{text[:150]}...")
        else:
            st.subheader("📊 匹配结果（仅向量相似度 Top10）")
            for cand in top10:
                cid, name, text, sim, phone, email, edu = cand
                with st.expander(f"👤 {name} — 向量相似度：{sim:.1%}"):
                    st.markdown(f"**📞 电话:** {phone or '未填'}  |  **✉️ 邮箱:** {email or '未填'}")
                    if edu:
                        st.markdown(f"**🎓 学历:** {edu}")
                    st.write(f"**简历摘要：** {text[:300]}...")

    if st.button("🔙 返回岗位库"):
        st.session_state.page = "job_lib"
        st.session_state.selected_job_for_match = None
        st.rerun()