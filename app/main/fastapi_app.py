from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from fastapi.responses import FileResponse
from typing import Dict, List, Deque
import datetime
import asyncio
import logging
from collections import deque
from core.graph import app as langgraph_app, run_full_workflow_test
from core.states import ProjectState
from pydantic import BaseModel
import os

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

# --- WebSocket 로깅 기능 추가 시작 ---
log_queue: Deque[str] = deque(maxlen=100) 

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        for log_entry in list(log_queue):
            try:
                await websocket.send_text(log_entry)
            except Exception:
                pass # 이미 끊겼을 수 있음

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_log(self, message: str):
        # 오래된 로그를 큐의 앞쪽에서 제거 (maxlen에 의해 자동 처리되지만 명시적으로도 가능)
        # if len(log_queue) >= log_queue.maxlen:
        #     log_queue.popleft() # 가장 오래된 로그 제거
        # log_queue.append(message) # 새 로그 추가 - 핸들러에서 이미 추가함

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass # 연결 끊김 등 예외 처리 (필요시 여기서 disconnect 호출 고려)

manager = ConnectionManager()

class WebSocketLogHandler(logging.Handler):
    def __init__(self, ws_manager: ConnectionManager):
        super().__init__()
        self.ws_manager = ws_manager

    def emit(self, record):
        log_entry = self.format(record)
        if len(log_queue) == log_queue.maxlen: # 큐가 꽉 찼으면 가장 오래된 것 제거 (새 로그 공간 확보)
            log_queue.popleft()
        log_queue.append(log_entry)
        asyncio.create_task(self.ws_manager.broadcast_log(log_entry))

# 로그 포맷터 설정
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s (%(name)s:%(funcName)s)')

# 루트 로거 가져오기 및 핸들러 추가
# 기본적으로 uvicorn이 사용하는 로거 레벨은 INFO일 수 있음
# 좀 더 상세한 로그를 원하면 로거 레벨 조정 필요
root_logger = logging.getLogger()
# 루트 로거 레벨 설정 (필요에 따라 DEBUG 등으로 변경 가능)
# 이미 uvicorn 등에서 설정한 레벨이 있다면 이 설정이 덮어쓸 수 있음
# root_logger.setLevel(logging.INFO) 

# 핸들러 추가 전에 중복 방지
# if not any(isinstance(h, WebSocketLogHandler) for h in root_logger.handlers):
ws_log_handler = WebSocketLogHandler(manager)
ws_log_handler.setFormatter(log_formatter)
root_logger.addHandler(ws_log_handler)
# 기본 로깅 레벨이 WARNING일 수 있으므로, INFO 레벨 로그도 처리하도록 설정
# (만약 uvicorn 실행 시 --log-level info 등으로 이미 설정했다면 불필요)
if root_logger.level > logging.INFO or root_logger.level == logging.NOTSET: # NOTSET (0)은 부모 따름
    root_logger.setLevel(logging.INFO)


# 애플리케이션 자체 로그도 남기기
# 로깅을 사용할 모든 모듈 상단에 logger = logging.getLogger(__name__) 추가 후 사용
# logger.info("이것은 FastAPI 앱의 INFO 로그입니다.")
# logger.warning("이것은 FastAPI 앱의 WARNING 로그입니다.")

@app.websocket("/ws/logs")
async def websocket_log_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # 연결 후에는 서버가 일방적으로 로그를 보내므로, 클라이언트 메시지 대기는 최소화
    try:
        while True:
            # 연결 상태를 유지하기 위한 최소한의 작업
            await asyncio.sleep(10) # 10초 대기, 연결 유지 목적
    except WebSocketDisconnect:
        logging.info(f"Log WebSocket client {websocket.client} disconnected.") # 클라이언트 정보 로깅
    except Exception as e:
        # 연결 관련 예외 로깅 강화
        logging.error(f"Log WebSocket error with client {websocket.client}: {e}", exc_info=True)
    finally: # finally 블록으로 이동하여 어떤 경우든 disconnect가 호출되도록
        manager.disconnect(websocket)
        logging.info(f"Log WebSocket client {websocket.client} connection closed and disconnected from manager.")

# --- WebSocket 로깅 기능 추가 끝 ---

@app.get("/")
def root():
    # 이 로그는 이제 WebSocket으로도 전송될 것임
    logging.info("Root endpoint was accessed.")
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
    logging.info(f"Diagnosis request received for service: {input_data.service_name}")
    try:
        # DiagnosisService.run_diagnosis 내부에서도 logging.info 등으로 로그를 남기면
        # 해당 로그들이 WebSocket으로 스트리밍될 것입니다.
        # 여기서는 result가 ProjectState와 유사한 딕셔너리이며,
        # 'final_report_pdf_path' 키를 포함하고 있다고 가정합니다.
        result = await DiagnosisService.run_diagnosis(
            service_name=input_data.service_name,
            service_info=input_data.service_initial_info
        )
        logging.info(f"Diagnosis for {input_data.service_name} completed. Report type: {result.get('report_type') if isinstance(result, dict) else 'N/A'}")
        
        # PDF 경로가 결과에 포함되어 있다면, 다운로드 URL을 생성하거나 파일명 전달
        if isinstance(result, dict) and result.get("final_report_pdf_path"):
            pdf_path = result["final_report_pdf_path"]
            pdf_filename = os.path.basename(pdf_path)
            # 프론트엔드가 다운로드 URL을 직접 구성할 수 있도록 파일명만 전달하거나,
            # 혹은 다운로드 URL 자체를 만들어서 전달할 수 있습니다.
            # 여기서는 파일명을 전달하고, 프론트에서 /reports/download/{filename} 형태로 요청한다고 가정합니다.
            result["final_report_pdf_filename"] = pdf_filename
            # result["download_url"] = f"/reports/download/{pdf_filename}" # 이렇게도 가능

        return result
    except Exception as e:
        logging.error(f"Error during diagnosis for {input_data.service_name}: {str(e)}", exc_info=True) # exc_info=True로 트레이스백 포함
        raise HTTPException(status_code=500, detail=f"진단 과정 중 오류 발생: {str(e)}")

# 애플리케이션 종료 시 정리
# @app.on_event("shutdown")
# async def shutdown_event():
#     for ws in manager.active_connections[:]: # 복사본으로 순회
#         await ws.close(code=1000)
#     logging.info("Application shutdown: Closed all active WebSocket connections.")

# --- PDF 다운로드 엔드포인트 추가 ---
REPORTS_DIR = os.path.join(os.getcwd(), "outputs", "reports")

@app.get("/reports/download/{filename}")
async def download_report_pdf(filename: str):
    # 경로 조작 방지를 위해 filename이 순수 파일명인지 확인 (간단한 예시)
    # 실제 운영 환경에서는 더 강력한 검증 필요 (예: 허용된 문자만 포함, .. 사용 금지 등)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="잘못된 파일명입니다.")

    file_path = os.path.join(REPORTS_DIR, filename)
    logging.info(f"PDF 다운로드 요청: {file_path}")

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        logging.error(f"요청된 PDF 파일을 찾을 수 없음: {file_path}")
        raise HTTPException(status_code=404, detail="요청된 파일을 찾을 수 없습니다.")
    
    # 파일명에 한글 등이 포함될 수 있으므로, Content-Disposition 헤더에 적절히 인코딩하여 전달
    # (URL 인코딩된 파일명을 UTF-8로 디코딩 후 다시 인코딩)
    # from urllib.parse import unquote
    # original_filename = unquote(filename, encoding='utf-8')
    # headers = {{
    #     'Content-Disposition': f'attachment; filename="{original_filename}"; filename*=UTF-8\'\'{original_filename}'
    # }}
    # return FileResponse(file_path, media_type='application/pdf', headers=headers)
    # FastAPI의 FileResponse는 기본적으로 filename을 기반으로 Content-Disposition을 설정해줌.
    # 브라우저가 이를 잘 해석하도록 filename 파라미터에 원본 파일명을 전달.
    return FileResponse(file_path, media_type='application/pdf', filename=filename)
