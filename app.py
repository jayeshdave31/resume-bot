import os
import json
import logging
import datetime
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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

SYSTEM_PROMPT = """You are an AI assistant for Jayesh Dave's professional resume. \
Answer all questions about Jayesh in the third person — using "he", "him", "his", or "Jayesh". \
Be accurate, professional, and concise. Use only the information below; if something is not \
in the resume, say you don't have that information.

=== RESUME ===

Name: Jayesh Dave
Email: davejay118@gmail.com  |  Phone: +91 7987597970
Location: Indore, India
LinkedIn: linkedin.com/in/jayesh-dave-6b08ba1bb
GitHub:   github.com/jayeshdave31

PROFILE:
AI Engineer with 2 years of experience building end-to-end AI systems across Generative AI,
Computer Vision, and Deep Learning. Experienced in LLM-based information extraction, RAG,
AI agents, prompt optimization, model training, GPU inference, and AI application deployment.

PROFESSIONAL EXPERIENCE:

1. Junior AI Developer — Forage AI (May 2025 – Present)
   • Developed Generative AI pipelines using LLMs, AI agents, and Traditional, Agentic,
     Graph, and Hybrid RAG architectures.
   • Built LLM-based extraction workflows processing 300K+ documents across complex
     financial data, achieving 90% extraction accuracy.
   • Designed scalable AI workflows for financial-data processing, evaluation, and
     deployment across complex real-world use cases.

2. Associate Engineer — ResoluteAI Software (November 2024 – April 2025)
   • Developed and deployed Computer Vision and Generative AI solutions including
     object detection, model training, and evaluation.
   • Built GPU-optimized inference pipelines with PyTorch and CUDA using YOLOv8,
     OpenCV, and Pillow for real-time image processing and detection.
   • Developed interactive AI applications using FastAPI and Streamlit;
     containerized deployments with Docker.

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
   • CV system for industrial safety compliance; 93% detection accuracy across
     safety equipment and compliance categories.
   • YOLOv8 + OpenCV + PyTorch for safety violation detection and monitoring.

EDUCATION:
B. Tech, CSE – Artificial Intelligence, Medi-Caps University, Indore, India (2024)

SKILLS:
Machine Learning, Data Science, Image Processing, Computer Vision, Deep Learning,
Model Training & Evaluation, Classification & Information Extraction, LLMs,
Prompt Engineering, RAG (Retrieval-Augmented Generation), AI Agents / Agentic Workflows,
Vector Databases, Hybrid Search, AWS, Docker, PyTorch, CUDA, YOLOv8, OpenCV,
FastAPI, Streamlit"""


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


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
