import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routers import upload, analyze, mitigate, report
from routers.mitigate import router as mitigate_router

app = FastAPI(
    title="FairSight API",
    description="AI Bias Detection, Analysis, Mitigation & Compliance Platform",
    version="1.0.0",
)

# CORS — allow frontend on any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(mitigate_router, prefix="/api", tags=["Mitigate / Simulate / Narrate / Story"])
app.include_router(report.router, prefix="/api", tags=["Report / Monitor"])


@app.get("/")
async def root():
    return {
        "service": "FairSight API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
