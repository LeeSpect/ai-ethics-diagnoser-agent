# core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jhgan/ko-sroberta-multitask")

# ChromaDB 설정
# CHROMA_DB_PATH는 프로젝트 루트를 기준으로 설정하는 것이 일반적입니다.
# Poetry로 실행 시 현재 작업 디렉토리(CWD)가 프로젝트 루트이므로 "./resources/chromadb"는 보통 잘 작동합니다.
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(os.getcwd(), "resources", "chromadb"))
CHROMA_CHECKLIST_COLLECTION_NAME = os.getenv("CHROMA_CHECKLIST_COLLECTION_NAME", "ethics_checklist_collection")
CHROMA_PAST_CASES_COLLECTION_NAME = os.getenv("CHROMA_PAST_CASES_COLLECTION_NAME", "past_cases_collection")

# PDF 데이터 경로 (프로젝트 루트 기준)
PDF_DATA_PATH = os.getenv("PDF_DATA_PATH", os.path.join(os.getcwd(), "data", "2025_인공지능_윤리기준_실천을_위한_자율점검표(안).pdf"))

# PDF 파싱 청크 크기 및 중첩 설정
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "1000"))
PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "100"))

# 검색 시 반환할 문서 수
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))