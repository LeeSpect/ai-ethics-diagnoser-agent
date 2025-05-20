from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from typing import Dict
import datetime
from core.graph import app as langgraph_app, run_full_workflow_test
from core.states import ProjectState
from pydantic import BaseModel

app = FastAPI(
    title="AI 윤리성 리스크 진단 및 개선 권고 멀티 에이전트 시스템",
    description="AI 시스템의 윤리성 리스크를 진단하고 개선 권고를 제공하는 멀티 에이전트 기반 API 서버입니다.",
    version="0.1.0"
)

# CORS 허용 (프론트엔드 연동 대비)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI 윤리성 리스크 진단 및 개선 권고 시스템에 오신 것을 환영합니다!"}

class DiagnosisInput(BaseModel):
    service_name: str
    service_initial_info: str
    
from app.services.diagnosis_service import DiagnosisService

@app.post("/diagnosis")
async def diagnose(input_data: DiagnosisInput = Body(...)):
    """
    AI 서비스 윤리성 진단을 수행하고 결과 보고서를 반환합니다.
    """
    try:
        return await DiagnosisService.run_diagnosis(
            service_name=input_data.service_name,
            service_info=input_data.service_initial_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"진단 과정 중 오류 발생: {str(e)}")
