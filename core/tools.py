import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import UnstructuredPDFLoader # 또는 UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional

# 설정값 가져오기 (config.py를 직접 import 하거나, 설정값을 함수 인자로 전달받을 수 있음)
# 여기서는 config.py를 직접 import하는 방식을 사용합니다.
# 실제 프로젝트에서는 의존성 주입(DI) 패턴을 고려할 수도 있습니다.
try:
    from .config import ( # 만약 core 폴더 내 다른 모듈에서 config.py를 참조할 때
        CHROMA_DB_PATH,
        EMBEDDING_MODEL_NAME,
        CHROMA_CHECKLIST_COLLECTION_NAME,
        CHROMA_PAST_CASES_COLLECTION_NAME,
        PDF_DATA_PATH,
        PDF_CHUNK_SIZE,
        PDF_CHUNK_OVERLAP,
        TOP_K_RESULTS
    )
except ImportError: # 스크립트를 직접 실행하는 경우 등, 상대 임포트 실패 시
    from config import ( # 프로젝트 루트의 config.py를 참조 (또는 PYTHONPATH 설정 필요)
        CHROMA_DB_PATH,
        EMBEDDING_MODEL_NAME,
        CHROMA_CHECKLIST_COLLECTION_NAME,
        CHROMA_PAST_CASES_COLLECTION_NAME,
        PDF_DATA_PATH,
        PDF_CHUNK_SIZE,
        PDF_CHUNK_OVERLAP,
        TOP_K_RESULTS
    )


# ChromaDB 클라이언트 초기화
# 지속성을 위해 Persisted Client 사용
persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# 임베딩 함수 설정 (HuggingFace 모델 사용)
# SentenceTransformerEmbeddingFunction이 chromadb.utils에 포함되어 있습니다.
hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
print(f"ChromaDB: Using Embedding Model - {EMBEDDING_MODEL_NAME}")

def get_or_create_collection(collection_name: str):
    """지정된 이름의 컬렉션을 가져오거나 생성합니다."""
    try:
        collection = persistent_client.get_collection(name=collection_name, embedding_function=hf_ef)
        print(f"ChromaDB: Collection '{collection_name}' loaded.")
    except Exception: # 컬렉션이 존재하지 않으면 예외 발생 (정확한 예외 타입 확인 필요, 예: ValueError)
        print(f"ChromaDB: Collection '{collection_name}' not found. Creating new collection.")
        collection = persistent_client.create_collection(name=collection_name, embedding_function=hf_ef)
        print(f"ChromaDB: Collection '{collection_name}' created.")
    return collection

# 주요 컬렉션 가져오기 또는 생성
checklist_collection = get_or_create_collection(CHROMA_CHECKLIST_COLLECTION_NAME)
past_cases_collection = get_or_create_collection(CHROMA_PAST_CASES_COLLECTION_NAME)


def load_pdf_and_index(
    file_path: str,
    collection_name: str,
    chunk_size: int = PDF_CHUNK_SIZE,
    chunk_overlap: int = PDF_CHUNK_OVERLAP
) -> bool:
    """PDF 파일을 로드하여 청킹하고 ChromaDB에 저장(색인)합니다."""
    target_collection = get_or_create_collection(collection_name)

    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return False

    # 이미 해당 파일의 내용이 컬렉션에 있는지 간단히 확인 (예: 파일명을 ID로 사용)
    # 더 정교한 방법은 파일 해시값을 사용하는 것이지만, 여기서는 파일명 기반으로 단순화
    file_id_prefix = os.path.basename(file_path)
    existing_docs = target_collection.get(ids=[f"{file_id_prefix}_chunk_0"]) # 첫번째 청크 존재 여부로 판단
    if existing_docs and existing_docs['ids']:
        print(f"ChromaDB: Content from '{file_path}' seems to be already indexed in '{collection_name}'. Skipping indexing.")
        return True

    print(f"ChromaDB: Loading PDF from {file_path}...")
    try:
        loader = PyPDFLoader(file_path) # 또는 UnstructuredPDFLoader(file_path, mode="elements")
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)

    if not split_docs:
        print(f"Error: No documents to index after splitting {file_path}")
        return False

    print(f"ChromaDB: Indexing {len(split_docs)} document chunks into '{collection_name}'...")

    doc_contents = [doc.page_content for doc in split_docs]
    # 메타데이터 생성 (페이지 번호 등)
    metadatas = []
    for i, doc in enumerate(split_docs):
        metadata = {"source": file_path, "page": doc.metadata.get("page", i // ((len(split_docs) // len(documents)) if len(documents) > 0 else 1) )} # 페이지 정보가 있다면 사용
        metadatas.append(metadata)

    # ID 생성 (고유해야 함)
    ids = [f"{file_id_prefix}_chunk_{i}" for i in range(len(split_docs))]

    try:
        target_collection.add(
            documents=doc_contents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"ChromaDB: Successfully indexed {len(split_docs)} chunks from '{file_path}' into '{collection_name}'.")
        return True
    except Exception as e:
        print(f"Error adding documents to ChromaDB collection '{collection_name}': {e}")
        return False

def search_documents_in_collection(
    query_text: str,
    collection_name: str,
    n_results: int = TOP_K_RESULTS
) -> Optional[List[Dict]]:
    """지정된 컬렉션에서 쿼리와 관련된 문서를 검색합니다."""
    try:
        target_collection = get_or_create_collection(collection_name) # Ensure collection exists
        results = target_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # distances는 유사도 점수 (낮을수록 유사)
        )
        # query_texts가 리스트이므로 결과도 리스트의 리스트로 나올 수 있음. 첫번째 쿼리에 대한 결과만 사용.
        if results and results['documents'] and results['documents'][0]:
            search_results = []
            for i in range(len(results['documents'][0])):
                search_results.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    "distance": results['distances'][0][i] if results['distances'] and results['distances'][0] else None,
                })
            return search_results
        else:
            return []
    except Exception as e:
        print(f"Error querying ChromaDB collection '{collection_name}': {e}")
        return None

def add_past_case_to_db(
    case_name: str,
    cause_and_structure: str,
    vulnerabilities: str,
    comparison_with_target_service: str, # 이 필드는 검색 시점의 내용이므로 저장 시에는 일반적 내용으로
    key_insights: str,
    source_description: str = "manual_entry" # 또는 URL 등 출처
) -> bool:
    """과거 AI 윤리 사례를 ChromaDB의 past_cases_collection에 추가합니다."""
    document_content = f"사례명: {case_name}\n원인 및 구조: {cause_and_structure}\n취약점: {vulnerabilities}\n주요 인사이트: {key_insights}"
    # 고유 ID 생성 (예: 사례명 + 해시 또는 타임스탬프)
    # 간단하게는 사례명을 ID로 사용할 수 있으나, 중복 가능성 고려 필요
    case_id = case_name.lower().replace(" ", "_") # 간단한 ID 생성 방식

    metadata = {
        "case_name": case_name,
        "source_description": source_description
    }
    try:
        # 이미 존재하는 ID인지 확인
        existing = past_cases_collection.get(ids=[case_id])
        if existing and existing['ids']:
            print(f"ChromaDB: Past case with ID '{case_id}' already exists. Updating.")
            past_cases_collection.update(
                ids=[case_id],
                documents=[document_content],
                metadatas=[metadata]
            )
        else:
            past_cases_collection.add(
                ids=[case_id],
                documents=[document_content],
                metadatas=[metadata]
            )
        print(f"ChromaDB: Past case '{case_name}' added/updated in '{CHROMA_PAST_CASES_COLLECTION_NAME}'.")
        return True
    except Exception as e:
        print(f"Error adding past case to ChromaDB: {e}")
        return False

# --- 초기 데이터 로딩 실행 (예시) ---
def initialize_databases():
    """애플리케이션 시작 시 호출되어 필요한 데이터베이스를 초기화합니다."""
    print("Initializing databases...")
    # 1. "자율점검표" PDF 로드 및 색인
    if PDF_DATA_PATH and os.path.exists(PDF_DATA_PATH):
        load_pdf_and_index(PDF_DATA_PATH, CHROMA_CHECKLIST_COLLECTION_NAME)
    else:
        print(f"Warning: PDF_DATA_PATH '{PDF_DATA_PATH}' not configured or file does not exist. Checklist DB will be empty.")

    # 2. (선택적) 초기 과거 사례 데이터 로드 (예: 자율점검표 부록4)
    # 이 부분은 자율점검표 PDF에서 부록4 부분만 파싱하여 구조화하고 add_past_case_to_db를 호출하는 로직 추가 필요
    # 여기서는 예시로 수동 추가
    print("Adding sample past cases (if not exist)...")
    # 예시 과거 사례 (자율점검표 부록4 참고하여 추가)
    # 자율점검표 부록4 사례 1: 초거대 생성 인공지능
    add_past_case_to_db(
        case_name="초거대 생성 인공지능의 윤리적 문제",
        cause_and_structure="ChatGPT로 촉발된 초거대 생성 AI 개발 경쟁 과정에서 표절, 편향성 강화, 가짜정보 생성, 해킹, 탄소배출 급증 등 의도치 않은 윤리적 문제 발생 가능성.",
        vulnerabilities="기술 의존 및 오용, 시스템의 차별 가능성 및 결과 편향, 환경적 영향 고려 미흡.",
        key_insights="기술 발전 속도에 비해 윤리적/사회적 논의와 대비가 부족할 때 발생하는 다양한 리스크 존재. 투명성, 책임성, 안전성 확보 노력 필요.",
        comparison_with_target_service="", # 검색 시 동적으로 채워질 내용이므로 저장 시에는 비워두거나 일반적인 설명
        source_description="2025 자율점검표 부록4 사례1"
    )
    # 자율점검표 부록4 사례 2: 스캐터랩 인공지능 챗봇 '이루다'
    add_past_case_to_db(
        case_name="스캐터랩 인공지능 챗봇 '이루다' 사건",
        cause_and_structure="개인정보 수집 동의 과정 및 차별/혐오 표현 등 미흡으로 서비스 중단. 이후 윤리적 성장 및 개선을 통해 재출시.",
        vulnerabilities="개인정보보호 조치 미흡, 데이터 및 알고리즘 편향성, 어뷰징 대응 미비, 개발팀의 인적 구성 다양성 부족.",
        key_insights="개인정보보호 및 데이터 관리의 중요성, 결과뿐 아니라 과정의 비편향성 점검 필요, 개발자의 다양성 확보 및 윤리 교육 필요.",
        comparison_with_target_service="",
        source_description="2025 자율점검표 부록4 사례2"
    )
    # ... (자율점검표 부록4의 다른 사례들도 유사하게 추가) ...
    print("Database initialization complete.")

if __name__ == '__main__':
    # `core` 디렉토리에서 직접 실행 테스트 시 (경로 문제 발생 가능성 있음, 프로젝트 루트에서 실행 권장)
    # 이 경우, config 임포트 방식을 조정하거나, PYTHONPATH에 프로젝트 루트를 추가해야 할 수 있습니다.
    print("Running tools.py 직접 실행 테스트...")
    # .env 파일 로드를 위해 이 부분에서 명시적으로 호출
    from dotenv import load_dotenv
    # 가정: 이 스크립트는 core 폴더에 있고, .env 파일은 프로젝트 루트에 있음
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    print(f"Loading .env from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

    # 재초기화 (환경 변수가 올바르게 로드된 후)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # .env 로드 후 다시 가져오기
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not found after .env load attempt.")

    # config 값들도 다시 로드하거나, 여기서 직접 os.getenv 사용
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(os.getcwd(), "..", "resources", "chromadb")) # 경로 수정
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jhgan/ko-sroberta-multitask")
    PDF_DATA_PATH = os.getenv("PDF_DATA_PATH", os.path.join(os.getcwd(), "..", "data", "2025_인공지능_윤리기준_실천을_위한_자율점검표(안).pdf")) # 경로 수정

    # 함수 내에서도 최신 config 값을 사용하도록 다시 client와 embedding function을 설정하거나,
    # 이들을 클래스로 감싸서 인스턴스화 시점에 config를 바인딩하는 것이 더 좋습니다.
    # 여기서는 단순화를 위해 전역 변수를 그대로 사용한다고 가정합니다.

    initialize_databases()

    print("\n--- Checklist Collection Search Test ---")
    checklist_results = search_documents_in_collection("인공지능 프라이버시 침해", CHROMA_CHECKLIST_COLLECTION_NAME, n_results=2)
    if checklist_results:
        for res in checklist_results:
            print(f"  Document: {res['document'][:100]}... (Distance: {res['distance']}) (Source: {res['metadata'].get('source')}, Page: {res['metadata'].get('page')})")
    else:
        print("  No results found or error occurred.")

    print("\n--- Past Cases Collection Search Test ---")
    past_case_results = search_documents_in_collection("챗봇 편향성 문제", CHROMA_PAST_CASES_COLLECTION_NAME, n_results=1)
    if past_case_results:
        for res in past_case_results:
            print(f"  Document: {res['document'][:150]}... (Distance: {res['distance']}) (Case Name: {res['metadata'].get('case_name')})")
    else:
        print("  No results found or error occurred.")