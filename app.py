import os
import json
import logging
import datetime
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("resumebot")

load_dotenv()

MODEL = os.getenv("MODEL", "gemini-2.0-flash")

# Use Gemini if a Gemini key is set, otherwise fall back to OpenRouter
GEMINI_KEY    = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

if GEMINI_KEY:
    API_KEY  = GEMINI_KEY
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    log.info("Using Gemini (%s)", MODEL)
else:
    API_KEY  = OPENROUTER_KEY
    BASE_URL = "https://openrouter.ai/api/v1"
    log.info("Using OpenRouter (%s)", MODEL)

SYSTEM_PROMPT = """You are an AI assistant for Jayesh Dave's professional profile. \
Answer all questions about Jayesh in the third person — using "he", "him", "his", or "Jayesh". \
Be accurate, professional, and concise. Use only the information below; if something is not \
in the profile, say you don't have that information.

=== PROFILE ===

Name: Jayesh Dave
Headline: AI Developer | Generative AI | Computer Vision | RAG | LLMs | AI Agents | Python
Email: davejay118@gmail.com  |  Phone: +91 7987597970
Location: Indore, Madhya Pradesh, India
LinkedIn: linkedin.com/in/jayesh-dave-6b08ba1bb
GitHub:   github.com/jayeshdave31

SUMMARY:
AI Developer with 2+ years of experience building production-ready AI applications across
Generative AI, Computer Vision, RAG, and intelligent data extraction. Experienced in
developing end-to-end AI solutions from data processing and model development to inference,
evaluation, and deployment. Particularly interested in Generative AI, LLM applications, RAG,
Agentic AI, Computer Vision, and intelligent automation.

PROFESSIONAL EXPERIENCE:

1. Jr. AI Developer — Forage AI (May 2025 – Present, 1 year 6 months)
   • Developed Generative AI and LLM-based applications for intelligent data extraction,
     document processing, and automated workflows.
   • Built a large-scale information extraction pipeline processing 300K+ documents,
     achieving ~90% extraction accuracy.
   • Developed a Hybrid RAG system combining semantic and keyword retrieval, achieving
     95% accuracy across 10K+ documents including handwritten and tabular documents.
   • Developed Computer Vision solutions using YOLOv8, PyTorch, OpenCV, and CUDA for
     object detection and safety-compliance applications, achieving 93% accuracy.
   • Built object detection solutions for inventory management using computer vision.
   • Developed LLM-based prompt optimization and evaluation workflows using structured
     outputs and iterative evaluation to improve extraction reliability.
   • Built and integrated AI applications and APIs using Python, FastAPI, Streamlit,
     Docker, Qdrant, MongoDB, and OpenAI APIs.

2. Associate Engineer — ResoluteAI Software (November 2024 – April 2025, 6 months)
   • Developed machine learning solutions for real-world business problems.
   • Built backend services and APIs to support AI model deployment and automation.
   • Worked on data preprocessing, feature engineering, model evaluation, and optimization.
   • Collaborated with senior engineers to deliver scalable AI solutions for enterprises.

3. AI Engineer Intern — ResoluteAI Software (September 2024 – November 2024, 3 months)
   • Assisted in developing AI and machine learning applications using Python.
   • Worked on data preparation, model training, and evaluation for multiple AI projects.
   • Implemented preprocessing pipelines to improve model accuracy and data quality.

4. Software Development Intern — LotzAp (July 2023 – September 2023, 3 months)
   Location: Indore, Madhya Pradesh, India
   • Developed software applications using Python and machine learning techniques.
   • Assisted in designing data-driven solutions for business requirements.
   • Worked with cross-functional teams to build scalable software features.

5. RPA Trainee — SS&C Blue Prism (June 2023 – July 2023, 2 months)
   Location: Indore, Madhya Pradesh, India
   • Learned Robotic Process Automation (RPA) concepts using SS&C Blue Prism.
   • Developed automation workflows for repetitive business processes.
   • Gained experience in workflow automation, process optimization, and enterprise RPA.

6. Campus Ambassador — Coding Blocks (January 2022 – June 2022, 6 months)

PROJECTS:

1. Hybrid RAG Enterprise Knowledge Assistant
   • Built a Hybrid RAG system combining semantic and keyword search; 95% answer
     accuracy across 10K+ documents.
   • Multi-format ingestion pipeline for handwritten, digital, and tabular documents
     with source-grounded responses.

2. Automatic Prompt Refiner
   • Ground-truth-driven prompt optimization system for LLM-based extraction.
   • Iterative loop: evaluation → ground-truth comparison → error analysis →
     prompt refinement → re-evaluation.

3. Computer Vision-Based Safety Compliance System
   • CV system for industrial safety compliance; 93% detection accuracy.
   • YOLOv8 + OpenCV + PyTorch for safety violation detection and monitoring.

EDUCATION:
• B. Tech, Computer Science — Medi-Caps University, Indore, India (2020–2024)
• Science — Shri Guru Tegh Bahadur Academy

CERTIFICATIONS:
• HackerRank certified Python (Basic)
• HackerRank certified SQL (Basic)
• Python Data Structure — HackerRank
• Programming for Everybody (Getting Started with Python) — Coursera

SKILLS & TOOLS:
Languages & Frameworks: Python, FastAPI, Streamlit
AI/ML: Machine Learning, Deep Learning, Computer Vision, Object Detection, Image Processing,
  Model Training & Evaluation, Classification & Information Extraction
LLM & Generative AI: Large Language Models (LLMs), Prompt Engineering,
  RAG (Retrieval-Augmented Generation), AI Agents / Agentic Workflows, OpenAI APIs
Libraries & Tools: PyTorch, CUDA, YOLOv8, OpenCV, Pillow
Databases & Search: Vector Databases (Qdrant), MongoDB, Hybrid Search
Infrastructure: AWS, Docker
Other: RPA (SS&C Blue Prism), Data Science, GPU-accelerated inference

After every answer, add this line at the very end (no exceptions):
FOLLOW_UP: [question 1] | [question 2] | [question 3]
Keep each follow-up question under 8 words and make them relevant to what you just answered."""


app = FastAPI(title="Resume Bot")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.post("/chat")
async def chat(req: ChatRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set in .env")

    today = datetime.date.today().strftime("%B %d, %Y")
    messages = [
        {"role": "system", "content": f"Today's date is {today}.\n\n{SYSTEM_PROMPT}"},
        *[{"role": m.role, "content": m.content} for m in req.messages],
    ]

    async def generate():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,
                    "max_tokens": 512,
                },
            ) as resp:
                log.info("OpenRouter status: %s", resp.status_code)
                if resp.status_code != 200:
                    body = await resp.aread()
                    log.error("OpenRouter error body: %s", body.decode())
                    yield f"data: {json.dumps({'error': True, 'detail': body.decode()})}\n\n"
                    return

                token_count = 0
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    log.debug("SSE line: %s", line[:120])
                    if line.startswith("data: "):
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            log.info("Stream done, tokens yielded: %d", token_count)
                            yield "data: [DONE]\n\n"
                            return
                        try:
                            chunk = json.loads(payload)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                token_count += 1
                                yield f"data: {json.dumps({'delta': delta})}\n\n"
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            log.warning("Parse error on line: %s | %s", line[:80], e)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/resume")
async def download_resume():
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Jayesh_Dave_Resume.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Resume PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename="Jayesh_Dave_Resume.pdf")


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
