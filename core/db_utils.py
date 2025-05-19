# core/db_utils.py
import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Unstructured가 자체 청킹을 하므로, 필요에 따라 선택적 사용
from typing import List, Dict, Optional, Any
import hashlib # 파일 내용 변경 감지를 위한 해시

try:
    from .config import (
        CHROMA_DB_PATH,
        EMBEDDING_MODEL_NAME,
        CHROMA_CHECKLIST_COLLECTION_NAME,
        CHROMA_PAST_CASES_COLLECTION_NAME,
        PDF_DATA_PATH,
        PDF_CHUNK_SIZE, # Unstructured 사용 시 이 값의 의미가 달라질 수 있음
        PDF_CHUNK_OVERLAP, # Unstructured 사용 시 이 값의 의미가 달라질_ 수 있음
        TOP_K_RESULTS
    )
except ImportError: # 스크립트를 프로젝트 루트에서 python -m core.db_utils 등으로 실행 시
    from config import (
        CHROMA_DB_PATH,
        EMBEDDING_MODEL_NAME,
        CHROMA_CHECKLIST_COLLECTION_NAME,
        CHROMA_PAST_CASES_COLLECTION_NAME,
        PDF_DATA_PATH,
        PDF_CHUNK_SIZE,
        PDF_CHUNK_OVERLAP,
        TOP_K_RESULTS
    )

# --- ChromaDB 클라이언트 및 임베딩 함수 초기화 ---
try:
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        print(f"ChromaDB: Created directory for PersistentClient at {CHROMA_DB_PATH}")

    persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    print(f"ChromaDB: PersistentClient initialized at path: {CHROMA_DB_PATH}")

    hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        # device="cuda" # GPU 사용 시
    )
    print(f"ChromaDB: Using Embedding Model - {EMBEDDING_MODEL_NAME}")
except Exception as e:
    print(f"ChromaDB: Critical error during client/embedding function initialization: {e}")
    hf_ef = None # 초기화 실패 시 None
    persistent_client = None


def get_or_create_collection(collection_name: str) -> Optional[chromadb.api.models.Collection.Collection]:
    """지정된 이름의 컬렉션을 가져오거나 생성합니다. 초기화 실패 시 None 반환."""
    if not persistent_client or not hf_ef:
        print("ChromaDB: Client or embedding function not initialized. Cannot get/create collection.")
        return None
    try:
        collection = persistent_client.get_collection(name=collection_name, embedding_function=hf_ef)
        print(f"ChromaDB: Collection '{collection_name}' loaded.")
    except Exception:
        print(f"ChromaDB: Collection '{collection_name}' not found. Creating new collection.")
        try:
            collection = persistent_client.create_collection(name=collection_name, embedding_function=hf_ef)
            print(f"ChromaDB: Collection '{collection_name}' created.")
        except Exception as ce:
            print(f"ChromaDB: Failed to create collection '{collection_name}': {ce}")
            return None
    return collection

# --- 주요 컬렉션 초기화 ---
checklist_collection: Optional[chromadb.api.models.Collection.Collection] = None
past_cases_collection: Optional[chromadb.api.models.Collection.Collection] = None

if persistent_client and hf_ef:
    checklist_collection = get_or_create_collection(CHROMA_CHECKLIST_COLLECTION_NAME)
    past_cases_collection = get_or_create_collection(CHROMA_PAST_CASES_COLLECTION_NAME)
else:
    print("ChromaDB: Skipping collection initialization due to client/EF init failure.")


def _generate_file_hash(file_path: str) -> str:
    """파일 내용의 SHA256 해시를 생성합니다."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_pdf_with_unstructured_and_index(
    file_path: str,
    collection: Optional[chromadb.api.models.Collection.Collection],
    # UnstructuredPDFLoader는 자체적으로 문서를 요소별로 나누므로,
    # Langchain의 RecursiveCharacterTextSplitter를 추가로 사용할지 여부는 선택사항입니다.
    # 여기서는 Unstructured 결과를 직접 사용하거나, 필요시 추가 청킹하는 예시를 포함합니다.
    use_recursive_splitter: bool = False,
    chunk_size: int = PDF_CHUNK_SIZE,
    chunk_overlap: int = PDF_CHUNK_OVERLAP
) -> bool:
    """UnstructuredPDFLoader를 사용하여 PDF를 로드하고 ChromaDB에 저장(색인)합니다."""
    if not collection:
        print(f"ChromaDB: Collection is None. Cannot index documents for {file_path}.")
        return False
    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return False

    file_basename = os.path.basename(file_path)
    file_hash = _generate_file_hash(file_path)
    # 파일 해시를 메타데이터로 저장하여, 동일 파일 내용 재색인 방지
    # 또는 파일 해시를 ID의 일부로 사용하여 관리
    # 여기서는 간단히 이미 색인된 파일인지 확인 (컬렉션 메타데이터나 특정 ID로 확인)
    # 더 정확하게는, 색인된 문서의 source와 hash를 비교해야 함
    # 여기서는 파일명 기반으로 이미 해당 파일의 청크가 있는지 확인
    existing_docs = collection.get(where={"source_file_hash": file_hash}, limit=1)
    if existing_docs and existing_docs['ids']:
        print(f"ChromaDB: Content from '{file_basename}' (hash: {file_hash}) seems to be already indexed in '{collection.name}'. Skipping.")
        return True

    print(f"ChromaDB: Loading PDF using UnstructuredPDFLoader from {file_path}...")
    try:
        # mode="elements"는 PDF를 텍스트, 테이블, 이미지 등 요소별로 분리
        # mode="single"은 전체 문서를 단일 langchain Document로 로드
        # mode="paged"는 페이지별로 Document 생성
        # 여기서는 "elements" 또는 "paged"를 고려해볼 수 있습니다. "paged"가 시작하기 편합니다.
        loader = UnstructuredPDFLoader(file_path, mode="paged") # 또는 "elements"
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF {file_path} with UnstructuredPDFLoader: {e}")
        return False

    if not documents:
        print(f"Error: No documents extracted from {file_path} by UnstructuredPDFLoader.")
        return False

    split_docs_to_index: List[Any] = []
    if use_recursive_splitter:
        # UnstructuredLoader가 생성한 Document 리스트를 추가로 청킹
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for doc in documents: # loader.load()는 Document 리스트를 반환
            # Unstructured는 이미 페이지별 또는 요소별로 나눌 수 있으므로, 추가 청킹이 항상 필요한 것은 아님
            # 여기서는 각 Document의 page_content를 추가로 나눔
            texts = text_splitter.split_text(doc.page_content)
            for i, text_chunk in enumerate(texts):
                # 새로운 Document 객체를 만들거나, 기존 메타데이터와 결합
                # 여기서는 간단히 텍스트와 기본 메타데이터만 사용
                split_docs_to_index.append({
                    "text": text_chunk,
                    "metadata": {
                        "source": file_basename,
                        "original_page_number": doc.metadata.get("page_number", None), # Unstructured가 제공하는 메타데이터
                        "chunk_in_page": i,
                        "source_file_hash": file_hash
                    }
                })
        print(f"ChromaDB: Further split into {len(split_docs_to_index)} chunks using RecursiveCharacterTextSplitter.")
    else:
        # UnstructuredPDFLoader 결과를 직접 사용 (예: mode="paged" 사용 시 페이지별 Document)
        for doc in documents:
            split_docs_to_index.append({
                "text": doc.page_content,
                "metadata": {
                    "source": file_basename,
                    "page_number": doc.metadata.get("page_number", None), # Unstructured 메타데이터 활용
                    "source_file_hash": file_hash,
                    **(doc.metadata or {}) # 원본 메타데이터 추가
                }
            })
        print(f"ChromaDB: Using {len(split_docs_to_index)} documents from UnstructuredPDFLoader (mode was likely 'paged' or 'elements').")


    if not split_docs_to_index:
        print(f"Error: No documents to index after processing {file_path}")
        return False

    print(f"ChromaDB: Indexing {len(split_docs_to_index)} document chunks into '{collection.name}'...")

    doc_contents = [item['text'] for item in split_docs_to_index]
    metadatas = [item['metadata'] for item in split_docs_to_index]
    ids = [f"{file_basename}_hash_{file_hash}_chunk_{i}" for i in range(len(split_docs_to_index))]

    try:
        collection.add(documents=doc_contents, metadatas=metadatas, ids=ids)
        print(f"ChromaDB: Successfully indexed {len(split_docs_to_index)} chunks from '{file_basename}' into '{collection.name}'.")
        return True
    except Exception as e:
        print(f"Error adding documents to ChromaDB collection '{collection.name}': {e}")
        return False

def search_documents(
    query_text: str,
    collection: Optional[chromadb.api.models.Collection.Collection],
    n_results: int = TOP_K_RESULTS,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> Optional[List[Dict]]:
    """지정된 컬렉션에서 쿼리와 관련된 문서를 검색합니다."""
    if not collection:
        print(f"ChromaDB: Collection is None. Cannot search for '{query_text}'.")
        return None
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata, # 메타데이터 필터링 (선택적)
            include=['documents', 'metadatas', 'distances']
        )
        if results and results['documents'] and results['documents'][0] is not None:
            search_results = []
            for i in range(len(results['documents'][0])):
                search_results.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    "distance": results['distances'][0][i] if results['distances'] and results['distances'][0] else None,
                })
            return search_results
        else:
            print(f"ChromaDB: No documents found for query '{query_text}' in '{collection.name}'.")
            return []
    except Exception as e:
        print(f"Error querying ChromaDB collection '{collection.name}': {e}")
        return None

def add_past_case_document(
    collection: Optional[chromadb.api.models.Collection.Collection],
    case_id: str, # 고유 ID
    document_content: str,
    metadata: Dict[str, Any]
) -> bool:
    """과거 사례 문서를 지정된 컬렉션에 추가/업데이트합니다."""
    if not collection:
        print(f"ChromaDB: Collection is None. Cannot add past case ID '{case_id}'.")
        return False
    try:
        existing = collection.get(ids=[case_id])
        if existing and existing['ids']:
            print(f"ChromaDB: Past case with ID '{case_id}' already exists in '{collection.name}'. Updating.")
            collection.update(ids=[case_id], documents=[document_content], metadatas=[metadata])
        else:
            collection.add(ids=[case_id], documents=[document_content], metadatas=[metadata])
        print(f"ChromaDB: Past case '{metadata.get('case_name', case_id)}' added/updated in '{collection.name}'.")
        return True
    except Exception as e:
        print(f"Error adding/updating past case ID '{case_id}' in ChromaDB: {e}")
        return False

# --- 초기 데이터 로딩 실행 함수 ---
def initialize_vector_databases():
    """애플리케이션 시작 시 호출되어 필요한 벡터 데이터베이스를 초기화합니다."""
    if not checklist_collection or not past_cases_collection:
        print("ChromaDB: One or more collections (checklist, past_cases) failed to initialize. Skipping data loading.")
        return

    print("ChromaDB: Initializing vector databases...")
    # 1. "자율점검표" PDF 로드 및 색인
    if PDF_DATA_PATH and os.path.exists(PDF_DATA_PATH):
        # UnstructuredPDFLoader 사용, RecursiveCharacterTextSplitter는 선택적으로 사용 (여기서는 False로)
        load_pdf_with_unstructured_and_index(PDF_DATA_PATH, checklist_collection, use_recursive_splitter=False)
    else:
        print(f"Warning: PDF_DATA_PATH '{PDF_DATA_PATH}' not configured or file does not exist. Checklist DB will be empty or not updated.")

    # 2. 초기 과거 사례 데이터 로드 (자율점검표 부록4 기반)
    print("ChromaDB: Adding/Updating sample past cases from checklist appendix (if not exist)...")
    # 자율점검표 부록 4의 사례들을 구조화하여 추가
    appendix_cases = [
        {
            "case_id": "appendix_case_1_generative_ai",
            "case_name": "초거대 생성 인공지능의 윤리적 문제 (자율점검표 부록4 사례1)",
            "content_summary": "ChatGPT로 촉발된 초거대 생성 AI 개발 경쟁 과정에서 표절, 편향성 강화, 가짜정보 생성, 해킹, 탄소배출 급증 등 의도치 않은 윤리적 문제 발생 가능성.",
            "ethical_considerations": "기술 의존 및 오용 방지, 시스템의 차별가능성 및 결과 편향 점검, 환경적 영향 관련 자료공개 및 정책적·기술적 대응 필요.",
            "related_core_requirements": ["인권보장", "프라이버시 보호", "다양성 존중", "침해금지", "공공성", "연대성", "안전성"],
            "source_description": "2025 자율점검표 부록4 사례1"
        },
        {
            "case_id": "appendix_case_2_iruda_chatbot",
            "case_name": "스캐터랩 인공지능 챗봇 '이루다' 사건 (자율점검표 부록4 사례2)",
            "content_summary": "개인정보 수집 동의 과정 및 차별/혐오 표현 등 미흡으로 서비스 중단. 이후 윤리적 성장 및 개선을 통해 재출시.",
            "ethical_considerations": "개인정보 보호 및 관리를 위한 조치 필요, 과정뿐만이 아닌, 결과의 비편향성 점검 필요, 개발자의 인적 구성 다양성 확보 필요.",
            "related_core_requirements": ["프라이버시 보호", "데이터 관리", "인권보장", "침해금지", "다양성 존중", "연대성", "공공성"],
            "source_description": "2025 자율점검표 부록4 사례2"
        },
        # ... 자율점검표 부록 4의 모든 사례(총 18개)를 유사한 구조로 추가 ...
        # 예시 (사례 15, 16)
        {
            "case_id": "appendix_case_15_election_fake_news",
            "case_name": "대선 후보 모함 가짜뉴스 확산 (자율점검표 부록4 사례15)",
            "content_summary": "튀르키예 대선 투표 며칠 전 테러집단이 야당 후보를 지지하는 가짜 영상 확산, 선거 결과에 영향.",
            "ethical_considerations": "특정 개인이나 집단 이익을 위해 악용되지 않도록 방지 필요, AI 생성물 표시 및 사후 추적을 위한 최신의 기술적 수단 도입 필요.",
            "related_core_requirements": ["공공성", "책임성", "투명성"],
            "source_description": "2025 자율점검표 부록4 사례15"
        },
        {
            "case_id": "appendix_case_16_deepfake_crime",
            "case_name": "딥페이크 성범죄 사건 (자율점검표 부록4 사례16)",
            "content_summary": "생성 AI로 아동 성 착취 영상물 제작 및 유포로 실형 선고.",
            "ethical_considerations": "악의적 이용자에 대한 사전적 및 사후적 대응 체계 마련 필요, AI 생성물 표시 및 사후 추적을 위한 최신의 기술적 수단 도입 필요.",
            "related_core_requirements": ["인권보장", "프라이버시 보호", "침해금지"],
            "source_description": "2025 자율점검표 부록4 사례16"
        }
    ]

    for case_data in appendix_cases:
        # 검색 및 LLM 컨텍스트로 활용될 주요 내용을 document_content로 구성
        doc_content = f"사례명: {case_data['case_name']}\n요약: {case_data['content_summary']}\n윤리적 고려사항: {case_data['ethical_considerations']}\n관련 핵심요건: {', '.join(case_data['related_core_requirements'])}"
        metadata = {
            "case_name": case_data['case_name'],
            "source_description": case_data['source_description'],
            "content_summary": case_data['content_summary'], # 검색 결과에서 바로 볼 수 있도록 메타데이터에도 추가
            "ethical_considerations": case_data['ethical_considerations'],
            "related_core_requirements": ",".join(case_data['related_core_requirements']) # 리스트를 문자열로
        }
        add_past_case_document(past_cases_collection, case_data['case_id'], doc_content, metadata)

    print("ChromaDB: Vector database initialization process complete.")

# --- 테스트용 실행 블록 ---
if __name__ == '__main__':
    print("Running db_utils.py directly for testing...")
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(dotenv_path):
        print(f"Loading .env from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Ensure API keys are set in environment.")

    # 설정값 다시 로드 (환경 변수 로드 후)
    # 실제로는 config 모듈을 다시 임포트하거나, Config 클래스 인스턴스를 사용하는 것이 좋음
    # 여기서는 간단히 전역 변수를 사용한다고 가정
    OPENAI_API_KEY_TEST = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY_TEST:
        print("Warning: OPENAI_API_KEY not found after .env load attempt in test block.")

    initialize_vector_databases() # 데이터베이스 초기화 (PDF 로드 및 과거 사례 추가)

    print("\n--- Checklist Collection Search Test (UnstructuredPDFLoader) ---")
    if checklist_collection:
        checklist_results = search_documents("AI 서비스의 투명성 확보 방안", checklist_collection, n_results=2)
        if checklist_results:
            for i, res in enumerate(checklist_results):
                print(f"  Result {i+1}: (Distance: {res['distance']:.4f})")
                print(f"    Source: {res['metadata'].get('source')}, Page: {res['metadata'].get('page_number', 'N/A')}")
                print(f"    Content: {res['document'][:200]}...")
        else:
            print("  No results found or error occurred in checklist search.")
    else:
        print("  Checklist collection not available for search test.")


    print("\n--- Past Cases Collection Search Test ---")
    if past_cases_collection:
        past_case_results = search_documents("챗봇 개인정보 유출 사례", past_cases_collection, n_results=1)
        if past_case_results:
            for i, res in enumerate(past_case_results):
                print(f"  Result {i+1}: (Distance: {res['distance']:.4f})")
                print(f"    Case Name: {res['metadata'].get('case_name')}")
                print(f"    Source: {res['metadata'].get('source_description')}")
                print(f"    Content Summary: {res['metadata'].get('content_summary')[:150]}...")
                # print(f"    Full Document: {res['document'][:200]}...") # 필요시 전체 문서 내용 확인
        else:
            print("  No results found or error occurred in past cases search.")
    else:
        print("  Past cases collection not available for search test.")