import os
import json
import sqlite3
import hashlib
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import numpy as np
import pandas as pd
import PyPDF2
from docx import Document
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

# ------------------------- 配置与环境变量 -------------------------
TENCENT_SECRET_ID = os.environ.get("TENCENT_SECRET_ID")
TENCENT_SECRET_KEY = os.environ.get("TENCENT_SECRET_KEY")
TOKENHUB_API_KEY = os.environ.get("TOKENHUB_API_KEY")

if not TENCENT_SECRET_ID or not TENCENT_SECRET_KEY:
    raise ValueError("请设置环境变量 TENCENT_SECRET_ID 和 TENCENT_SECRET_KEY")
if not TOKENHUB_API_KEY:
    raise ValueError("请设置环境变量 TOKENHUB_API_KEY")

# ------------------------- 数据库 -------------------------
DB_PATH = "hunter.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    # 岗位表
    c.execute('''CREATE TABLE IF NOT EXISTS jobs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  jd_text TEXT,
                  embedding TEXT,
                  company_name TEXT,
                  platform TEXT,
                  department TEXT,
                  location TEXT,
                  core_business TEXT,
                  candidate_profile TEXT)''')
    # 简历表
    c.execute('''CREATE TABLE IF NOT EXISTS resumes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  text TEXT,
                  embedding TEXT,
                  phone TEXT,
                  email TEXT,
                  education TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ------------------------- 客户端初始化 -------------------------
@contextmanager
def get_tencent_embedding_client():
    cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
    httpProfile = HttpProfile()
    httpProfile.endpoint = "hunyuan.tencentcloudapi.com"
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    yield hunyuan_client.HunyuanClient(cred, "", clientProfile)

def get_chat_client():
    return OpenAI(api_key=TOKENHUB_API_KEY, base_url="https://tokenhub.tencentmaas.com/v1", timeout=15.0)

chat_client = get_chat_client()

# ------------------------- 辅助函数 -------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text: str) -> List[float]:
    with get_tencent_embedding_client() as client:
        req = models.GetEmbeddingRequest()
        req.Input = text[:1024]
        resp = client.GetEmbedding(req)
        return resp.Data[0].Embedding

def extract_text_from_file(content: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf':
        import io
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    elif ext == '.docx':
        import io
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("不支持的文件类型，仅支持 PDF 或 DOCX")

def smart_truncate(text: str, max_len: int = 3000) -> str:
    if len(text) <= max_len:
        return text
    head_len = int(max_len * 0.7)
    tail_len = max_len - head_len
    return text[:head_len] + "\n...(中间内容省略)...\n" + text[-tail_len:]

def evaluate_match_with_ai(job_title: str, jd_text: str, resume_text: str):
    """返回 (score, reason, dimensions)"""
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

# ------------------------- Pydantic 模型 -------------------------
class JobCreate(BaseModel):
    title: str
    jd_text: str
    company_name: Optional[str] = ""
    platform: Optional[str] = ""
    department: Optional[str] = ""
    location: Optional[str] = ""
    core_business: Optional[str] = ""
    candidate_profile: Optional[str] = ""

class ResumeCreate(BaseModel):
    name: str
    text: str
    phone: Optional[str] = ""
    email: Optional[str] = ""
    education: Optional[str] = ""

class MatchResumeRequest(BaseModel):
    resume_text: Optional[str] = None
    resume_id: Optional[int] = None

class MatchJobRequest(BaseModel):
    job_id: int

# ------------------------- FastAPI 应用 -------------------------
app = FastAPI(title="简历岗位匹配 AI 助手 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请改为具体前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------- 岗位 API -------------------------
@app.get("/api/jobs")
def get_jobs(
    company: str = "",
    platform: str = "",
    location: str = ""
):
    conn = get_db()
    c = conn.cursor()
    query = "SELECT id, title, company_name, platform, location, jd_text FROM jobs WHERE 1=1"
    params = []
    if company:
        query += " AND company_name LIKE ?"
        params.append(f"%{company}%")
    if platform:
        query += " AND platform LIKE ?"
        params.append(f"%{platform}%")
    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    query += " ORDER BY id DESC"
    rows = c.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    conn = get_db()
    c = conn.cursor()
    row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "岗位不存在")
    return dict(row)

@app.post("/api/jobs")
def create_job(job: JobCreate):
    try:
        embedding = get_embedding(job.jd_text)
    except Exception as e:
        raise HTTPException(500, f"生成向量失败: {str(e)}")
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO jobs
        (title, jd_text, embedding, company_name, platform, department, location, core_business, candidate_profile)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (job.title, job.jd_text, json.dumps(embedding), job.company_name, job.platform,
          job.department, job.location, job.core_business, job.candidate_profile))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return {"id": new_id, "message": "岗位创建成功"}

@app.post("/api/jobs/import-excel")
async def import_jobs_excel(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(400, "请上传 Excel 文件")
    content = await file.read()
    df = pd.read_excel(pd.io.BytesIO(content))
    required = ["公司名", "公司平台", "部门", "工作地点", "核心业务", "岗位名", "岗位JD", "人选画像"]
    if not all(col in df.columns for col in required):
        raise HTTPException(400, f"Excel 必须包含列: {', '.join(required)}")
    conn = get_db()
    c = conn.cursor()
    success = 0
    for _, row in df.iterrows():
        try:
            jd_text = str(row["岗位JD"])
            embedding = get_embedding(jd_text)
            c.execute("""
                INSERT INTO jobs
                (title, jd_text, embedding, company_name, platform, department, location, core_business, candidate_profile)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (row["岗位名"], jd_text, json.dumps(embedding),
                  row.get("公司名", ""), row.get("公司平台", ""), row.get("部门", ""),
                  row.get("工作地点", ""), row.get("核心业务", ""), row.get("人选画像", "")))
            success += 1
        except Exception as e:
            print(f"导入失败: {e}")
    conn.commit()
    conn.close()
    return {"success": success, "total": len(df)}

@app.delete("/api/jobs")
def delete_jobs(ids: List[int]):
    conn = get_db()
    c = conn.cursor()
    c.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    conn.close()
    return {"deleted": len(ids)}

# ------------------------- 简历 API -------------------------
@app.get("/api/resumes")
def get_resumes(name: str = "", phone: str = "", email: str = ""):
    conn = get_db()
    c = conn.cursor()
    query = "SELECT id, name, phone, email, education, text FROM resumes WHERE 1=1"
    params = []
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if phone:
        query += " AND phone LIKE ?"
        params.append(f"%{phone}%")
    if email:
        query += " AND email LIKE ?"
        params.append(f"%{email}%")
    query += " ORDER BY id DESC"
    rows = c.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.post("/api/resumes")
def create_resume(resume: ResumeCreate):
    try:
        embedding = get_embedding(resume.text)
    except Exception as e:
        raise HTTPException(500, f"生成向量失败: {str(e)}")
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO resumes (name, text, embedding, phone, email, education)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (resume.name, resume.text, json.dumps(embedding), resume.phone, resume.email, resume.education))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return {"id": new_id, "message": "简历创建成功"}

@app.post("/api/resumes/import-excel")
async def import_resumes_excel(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(400, "请上传 Excel 文件")
    content = await file.read()
    df = pd.read_excel(pd.io.BytesIO(content))
    required = ["姓名", "电话", "邮箱", "学历背景", "简历正文"]
    if not all(col in df.columns for col in required):
        raise HTTPException(400, f"Excel 必须包含列: {', '.join(required)}")
    conn = get_db()
    c = conn.cursor()
    success = 0
    for _, row in df.iterrows():
        try:
            resume_text = str(row["简历正文"])
            embedding = get_embedding(resume_text)
            c.execute("""
                INSERT INTO resumes (name, text, embedding, phone, email, education)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (row["姓名"], resume_text, json.dumps(embedding),
                  row.get("电话", ""), row.get("邮箱", ""), row.get("学历背景", "")))
            success += 1
        except Exception as e:
            print(f"导入失败: {e}")
    conn.commit()
    conn.close()
    return {"success": success, "total": len(df)}

@app.delete("/api/resumes")
def delete_resumes(ids: List[int]):
    conn = get_db()
    c = conn.cursor()
    c.executemany("DELETE FROM resumes WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    conn.close()
    return {"deleted": len(ids)}

# ------------------------- 匹配 API -------------------------
def _get_resume_text_from_request(req: MatchResumeRequest):
    if req.resume_text:
        return req.resume_text
    if req.resume_id:
        conn = get_db()
        c = conn.cursor()
        row = c.execute("SELECT text FROM resumes WHERE id = ?", (req.resume_id,)).fetchone()
        conn.close()
        if not row:
            raise HTTPException(404, "简历不存在")
        return row["text"]
    raise HTTPException(400, "请提供 resume_text 或 resume_id")

@app.post("/api/match/resume-to-jobs")
def match_resume_to_jobs(req: MatchResumeRequest):
    # 获取简历文本
    try:
        resume_text = _get_resume_text_from_request(req)
    except HTTPException as e:
        raise e
    # 生成简历向量
    try:
        resume_emb = get_embedding(resume_text)
    except Exception as e:
        raise HTTPException(500, f"生成简历向量失败: {str(e)}")

    conn = get_db()
    c = conn.cursor()
    jobs = c.execute("SELECT id, title, jd_text, embedding FROM jobs").fetchall()
    conn.close()
    if not jobs:
        return []  # 无岗位，返回空列表

    # 计算向量相似度
    results = []
    for job in jobs:
        if job["embedding"]:
            job_emb = json.loads(job["embedding"])
            sim = cosine_similarity(resume_emb, job_emb)
            results.append({
                "job_id": job["id"],
                "title": job["title"],
                "jd_text": job["jd_text"],
                "similarity": sim
            })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    top3 = results[:3]

    # 并发 AI 评分
    def ai_score(job_info):
        score, reason, dimensions = evaluate_match_with_ai(
            job_info["title"], job_info["jd_text"], resume_text
        )
        if score is None:
            score = int(job_info["similarity"] * 10)
            reason = f"AI调用失败，回退分数 ({reason})"
        return {
            "job_id": job_info["job_id"],
            "title": job_info["title"],
            "score": round(score, 1) if score else 0,
            "reason": reason,
            "similarity": job_info["similarity"],
            "dimensions": dimensions
        }

    ai_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(ai_score, job): job for job in top3}
        for future in as_completed(futures):
            ai_results.append(future.result())
    ai_results.sort(key=lambda x: x["score"], reverse=True)

    # 合并其余岗位（仅相似度）
    other_jobs = [
        {
            "job_id": j["job_id"],
            "title": j["title"],
            "similarity": j["similarity"],
            "score": None,
            "reason": None,
            "dimensions": None
        }
        for j in results[3:10]  # 最多返回10个
    ]

    return {
        "top_matches": ai_results,      # 前3个带AI评分
        "other_matches": other_jobs     # 后续仅相似度
    }

@app.post("/api/match/job-to-candidates")
def match_job_to_candidates(req: MatchJobRequest):
    conn = get_db()
    c = conn.cursor()
    job = c.execute("SELECT title, jd_text, embedding FROM jobs WHERE id = ?", (req.job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "岗位不存在")
    job_title, jd_text, job_emb_json = job["title"], job["jd_text"], job["embedding"]
    if not job_emb_json:
        raise HTTPException(400, "岗位没有向量数据，请重新保存")
    job_emb = json.loads(job_emb_json)

    resumes = c.execute("SELECT id, name, text, embedding, phone, email, education FROM resumes").fetchall()
    conn.close()
    if not resumes:
        return []

    # 向量相似度
    candidates = []
    for r in resumes:
        if r["embedding"]:
            resume_emb = json.loads(r["embedding"])
            sim = cosine_similarity(job_emb, resume_emb)
            candidates.append({
                "resume_id": r["id"],
                "name": r["name"],
                "text": r["text"],
                "similarity": sim,
                "phone": r["phone"] or "",
                "email": r["email"] or "",
                "education": r["education"] or ""
            })
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    top3 = candidates[:3]

    # 并发 AI 评分
    def ai_score_candidate(cand):
        score, reason, dimensions = evaluate_match_with_ai(job_title, jd_text, cand["text"])
        if score is None:
            score = int(cand["similarity"] * 10)
            reason = f"AI调用失败，回退分数 ({reason})"
        return {
            "resume_id": cand["resume_id"],
            "name": cand["name"],
            "score": round(score, 1) if score else 0,
            "reason": reason,
            "similarity": cand["similarity"],
            "phone": cand["phone"],
            "email": cand["email"],
            "education": cand["education"],
            "dimensions": dimensions
        }

    ai_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(ai_score_candidate, cand): cand for cand in top3}
        for future in as_completed(futures):
            ai_results.append(future.result())
    ai_results.sort(key=lambda x: x["score"], reverse=True)

    other_candidates = [
        {
            "resume_id": c["resume_id"],
            "name": c["name"],
            "similarity": c["similarity"],
            "phone": c["phone"],
            "email": c["email"],
            "education": c["education"],
            "score": None,
            "reason": None,
            "dimensions": None
        }
        for c in candidates[3:10]
    ]

    return {
        "top_matches": ai_results,
        "other_matches": other_candidates
    }

# ------------------------- 文件上传匹配（支持 PDF/Word） -------------------------
@app.post("/api/match/resume-to-jobs/upload")
async def match_resume_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.docx')):
        raise HTTPException(400, "仅支持 PDF 或 DOCX 文件")
    content = await file.read()
    try:
        resume_text = extract_text_from_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"文件解析失败: {str(e)}")
    if not resume_text:
        raise HTTPException(400, "未能从文件中提取文本")
    req = MatchResumeRequest(resume_text=resume_text)
    return match_resume_to_jobs(req)

# ------------------------- 启动入口 -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)