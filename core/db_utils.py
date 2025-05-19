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
            "ethical_considerations": "기술 의존 및 오용 방지를 위한 조치 필요. 시스템의 차별가능성 및 결과 편향 점검 필요. 환경적 영향 관련 자료공개 및 정책적·기술적 대응 필요.",
            "related_core_requirements": ["인권보장", "프라이버시 보호", "다양성 존중", "침해금지", "공공성", "연대성", "안전성"],
            "source_description": "2025 자율점검표 부록4 사례1"
        },
        {
            "case_id": "appendix_case_2_iruda_chatbot",
            "case_name": "스캐터랩 인공지능 챗봇 '이루다' (자율점검표 부록4 사례2)",
            "content_summary": "스캐터랩은 인공지능 챗봇 '이루다 1.0'을 최초 출시('20.12)하였으나, 개인정보 수집 동의 과정 및 차별 표현 등 미흡한 부분으로 인해 3주만에 서비스 중단. 윤리적 성장, 서비스 개선(개인정보 보호조치 강화, 어뷰징 모델 개발 등)을 통해 '이루다 2.0' 정식 출시('22.10).",
            "ethical_considerations": "개인정보 보호 및 관리를 위한 조치 필요. 과정뿐만이 아닌, 결과의 비편향성 점검 필요. 개발자의 인적 구성 다양성 확보 필요.",
            "related_core_requirements": ["프라이버시 보호", "데이터 관리", "인권보장", "침해금지", "다양성 존중", "연대성", "공공성"],
            "source_description": "2025 자율점검표 부록4 사례2"
        },
        {
            "case_id": "appendix_case_3_uk_exam_algorithm",
            "case_name": "영국 대입 시험 점수산정 시스템 (자율점검표 부록4 사례3)",
            "content_summary": "영국 시험감독청은 코로나19로 취소된 대입 시험 'A레벨'을 대신하여 알고리즘으로 성적을 산출하였으나 편향된 알고리즘 결과에 대한 강한 사회적 반발로 인해 철회함('20.8). 특히 사립학교의 부유층 학생에게 유리한 결과를 도출하여 사회적 불평등 강화를 초래.",
            "ethical_considerations": "결과의 공정성 여부 점검 필요. 중요 결정의 도출 과정에 대한 설명 가능성 확보 필요. 이해관계자의 참여 기회 보장 필요.",
            "related_core_requirements": ["공공성", "다양성 존중", "투명성", "연대성"],
            "source_description": "2025 자율점검표 부록4 사례3"
        },
        {
            "case_id": "appendix_case_4_us_police_facial_recognition",
            "case_name": "미국 경찰 범죄수사용 안면인식 시스템 (자율점검표 부록4 사례4)",
            "content_summary": "미국 경찰국은 안면인식 기술을 활용한 범죄 수사 사건에 흑인 3명을 범죄자로 오인하여 부당 고소·체포하여 인권침해 및 인종차별 논란을 빚음. 특히, 인종, 성별, 민족 등의 요인에 따라 정확도가 보장되지 않은 안면인식 기술에 의존한 체포와 구금.",
            "ethical_considerations": "기술 의존 및 오용 방지를 위한 조치 필요. 알고리즘 편향성 점검 필요.",
            "related_core_requirements": ["인권보장", "침해금지", "다양성 존중", "데이터 관리", "공공성", "연대성"],
            "source_description": "2025 자율점검표 부록4 사례4"
        },
        {
            "case_id": "appendix_case_5_dutch_welfare_fraud_detection",
            "case_name": "네덜란드 복지수당 사기탐지 시스템 (자율점검표 부록4 사례5)",
            "content_summary": "네덜란드 정부는 복지혜택 부정수급과 세금 사기를 단속하기 위해 위험탐지시스템(SyRi)을 개발·활용하였으나 중앙정부 및 지방자치단체의 데이터를 활용한 사생활 침해, 저소득층, 이민자 등 소수·취약집단 차별, 비공개 인공지능 모델·데이터에 대한 투명성 부족 문제를 지적한 법원 판결로 철폐됨('20.2).",
            "ethical_considerations": "개인정보 보호 및 관리를 위한 조치 필요. 시스템 적용대상 범위 및 결과 편향 점검 필요. (훈련)데이터, 인공지능 모델의 투명성 확보 필요.",
            "related_core_requirements": ["프라이버시 보호", "인권보장", "다양성 존중", "침해금지", "공공성", "투명성", "데이터 관리", "연대성"],
            "source_description": "2025 자율점검표 부록4 사례5"
        },
        {
            "case_id": "appendix_case_6_hirevue_interview_software",
            "case_name": "미국 하이어뷰社 채용 면접 영상분석 소프트웨어 (자율점검표 부록4 사례6)",
            "content_summary": "인공지능 기술을 활용해 입사 지원자의 말투, 얼굴 표정 등을 분석하여 직무적합성을 판단할 수 있다고 홍보한 하이어뷰는 허위·과대 광고로 FTC에 제소되었으며('19.11), 하이어뷰社는 대응방안으로 안면인식 기술의 활용을 중단한다고 발표('21.1).",
            "ethical_considerations": "분석 소프트웨어의 효과에 대한 과학적 증거 확보 필요. 시스템의 차별가능성 및 생체정보 수집 여부 검토 필요.",
            "related_core_requirements": ["침해금지", "책임성", "투명성"],
            "source_description": "2025 자율점검표 부록4 사례6"
        },
        {
            "case_id": "appendix_case_7_ai_power_consumption_carbon_emission",
            "case_name": "인공지능 기술의 전력소비와 탄소배출 (자율점검표 부록4 사례7)",
            "content_summary": "인공지능 기술은 방대한 양의 데이터를 기반으로 훈련, 개발 및 운영되며 데이터센터, 클라우드 인프라, 기타 하드웨어에 소요되는 전력과 탄소배출에 따른 부정적 환경 영향에 대한 우려 및 정책적·기술적 대응방안이 이슈화됨.",
            "ethical_considerations": "환경적 영향 관련 자료공개 및 정책적·기술적 대응 필요.",
            "related_core_requirements": ["공공성", "연대성"],
            "source_description": "2025 자율점검표 부록4 사례7"
        },
        {
            "case_id": "appendix_case_8_ms_chatbot_tay_discrimination",
            "case_name": "마이크로소프트 대화용 챗봇 '테이'의 차별 발언 (자율점검표 부록4 사례8)",
            "content_summary": "마이크로소프트는 16세 미국인 소녀를 벤치마킹한 딥러닝 기반의 대화형 챗봇 '테이(Tay)'를 선보였으나, 일부 극단주의자가 주입한 악의적 데이터 학습으로 인종·성차별 및 자극적인 정치 발언 등이 문제가 되어 출시 16시간 만에 운영 중단('16.3).",
            "ethical_considerations": "데이터 및 알고리즘 편향성 점검 필요. 악의적 이용자에 대한 사전적 및 사후적 대응 체계 마련 필요.",
            "related_core_requirements": ["인권보장", "데이터 관리", "침해금지", "다양성 존중", "연대성", "공공성"],
            "source_description": "2025 자율점검표 부록4 사례8"
        },
        {
            "case_id": "appendix_case_9_gatebox_hologram_chatbot_overdependence",
            "case_name": "일본 기업 '빈클루'의 홀로그램 챗봇 '게이트박스' 과의존 (자율점검표 부록4 사례9)",
            "content_summary": "일본의 스타트업 기업 '빈클루(VinClu)'가 개발한 AI 어시스턴스 '게이트박스(Gatebox)'가 일본의 유명 아이돌 캐릭터를 홀로그램 모델로 차용하면서 한 남성이 해당 캐릭터와 결혼 선언('17). 게이트박스(Gatebox)는 인터넷 정보 전달 및 사물인터넷(IoT) 관리와 함께 챗봇서비스를 제공하는 AI 어시스턴스 제품.",
            "ethical_considerations": "인공지능 기반 서비스를 제공하며 인간과 인공지능 간의 상호 소통이라는 사실 공지. 이용자의 과의존, 현실과의 혼동, 과몰입 등을 방지하기 위한 대응책 마련 필요.",
            "related_core_requirements": ["침해금지", "공공성", "책임성", "안전성"],
            "source_description": "2025 자율점검표 부록4 사례9"
        },
        {
            "case_id": "appendix_case_10_nabla_psychiatric_chatbot_suicide_recommendation",
            "case_name": "프랑스 기업 '나블라' 정신과 상담용 챗봇의 자살 권유 (자율점검표 부록4 사례10)",
            "content_summary": "'나블라(Nabla)'에서 정신과 상담 목적으로 개발된 GPT-3 기반 의료용 챗봇이 출시를 앞두고 테스트 중 모의 환자와의 대화에서 자살 권유('20.10).",
            "ethical_considerations": "알고리즘 투명성 및 안전성 점검 필요. 자연어처리모델 학습데이터의 무결성 제고 노력 필요.",
            "related_core_requirements": ["인권보장", "데이터 관리", "침해금지", "공공성", "안전성"],
            "source_description": "2025 자율점검표 부록4 사례10"
        },
        {
            "case_id": "appendix_case_11_amazon_alexa_penny_challenge",
            "case_name": "아마존 Alexa가 10세 아동에게 '페니 챌린지' 추천 (자율점검표 부록4 사례11)",
            "content_summary": "인공지능 소통 플랫폼인 Alexa가 10세 아동과 대화하던 중 '도전할 만한 것이 무엇이 있느냐'는 질문에 벽에 붙어 있는 콘센트에 충전기를 꽂은 뒤 동전(페니)로 건드려 불꽃을 내는 위험한 장난인 '페니 챌린지'를 추천('21.10).",
            "ethical_considerations": "출력한 결과치에 대한 안전성을 지속적으로 점검하고 평가하기 위한 절차 마련 필요. 알고리즘 검색 결과에 대한 심의 기준 부여 등 조치 필요.",
            "related_core_requirements": ["안전성", "공공성", "침해금지"],
            "source_description": "2025 자율점검표 부록4 사례11"
        },
        {
            "case_id": "appendix_case_12_itutor_group_age_discrimination_ai",
            "case_name": "중국 '아이튜터그룹'의 고령 지원자 거부 인공지능 (자율점검표 부록4 사례12)",
            "content_summary": "미국 교사를 고용하여 온라인 과외 서비스를 제공하는 중국의 '아이튜터그룹(iTutorGroup)'이 채용 과정에서 55세 이상의 여성과 60세 이상의 남성 지원자를 거부하도록 설계된 인공지능 채용 도구를 활용하자 미국 '평등고용기회위원회(EEOC)'에 의해 '고용상 연령차별금지법(ADEA)' 위반을 이유로 제소('23.8).",
            "ethical_considerations": "기술 오용 방지를 위한 조치 필요. 알고리즘 투명성 확보를 통한 알고리즘의 의도적 차별에 대한 점검 필요.",
            "related_core_requirements": ["인권보장", "다양성 존중", "침해금지", "투명성"],
            "source_description": "2025 자율점검표 부록4 사례12"
        },
        {
            "case_id": "appendix_case_13_public_institution_ai_recruitment_disclosure_refusal",
            "case_name": "공공기관의 인공지능 채용 도구 활용에 관한 정보공개거부 (자율점검표 부록4 사례13)",
            "content_summary": "국내 시민단체가 인공지능 채용 도구를 활용한 공공기관에 대하여 해당 도구의 편향성 및 개인정보 침해 여부 등을 검토하고자 정보공개청구를 신청하였으나 이를 거부당하자 법원에 정보공개거부처분취소소송을 제기하였고('20), 법원은 공공기관이 인공지능 채용 도구 개발사의 교육 자료, 지원자 개인정보 관리 문서, 업체와의 계약 서류 등을 공개하도록 원고 일부 승소 판결('22).",
            "ethical_considerations": "인공지능시스템 활용에 관한 투명성 확보를 통한 프라이버시, 다양성 존중 등 타 핵심요건 고려 필요. 국가·공공기관 또는 교육기관 등 공공영역의 인공지능시스템 활용 시 더욱 엄격한 기준 적용 필요.",
            "related_core_requirements": ["인권보장", "프라이버시 보호", "다양성 존중", "공공성", "책임성", "투명성"],
            "source_description": "2025 자율점검표 부록4 사례13"
        },
        {
            "case_id": "appendix_case_14_getty_images_stability_ai_copyright",
            "case_name": "Getty Images와 Stability AI 저작권 침해 소송 사건 (자율점검표 부록4 사례14)",
            "content_summary": "이미지 플랫폼 회사 '게티이미지(Getty Images)'는 이미지 생성 AI 서비스 '스테이블 디퓨전(Stable Diffusion)'의 개발사인 '스태빌리티 AI(Stability Al)'를 상대로 저작권 침해 소송을 제기('23.1).",
            "ethical_considerations": "(훈련)데이터, 인공지능 모델의 투명성 확보 필요. 저작권 및 라이선스 문제에 대한 철저한 검토 및 법적 책임에 대한 대비.",
            "related_core_requirements": ["침해금지", "데이터 관리", "투명성"],
            "source_description": "2025 자율점검표 부록4 사례14"
        },
        {
            "case_id": "appendix_case_15_election_fake_news",
            "case_name": "대선 후보를 모함하는 가짜뉴스 확산 (자율점검표 부록4 사례15)",
            "content_summary": "튀르키예 대선 투표 며칠 전 테러집단이 야당 후보를 지지하는 가짜 영상이 확산되었고, 결과적으로 해당 후보는 대선에서 패배하였으며 선거가 끝난 뒤에야 조작된 영상이란 것이 밝혀짐('23.5).",
            "ethical_considerations": "특정 개인이나 집단 이익을 위해 악용되지 않도록 방지 필요. AI 생성물 표시 및 사후 추적을 위한 최신의 기술적 수단 도입 필요.",
            "related_core_requirements": ["공공성", "책임성", "투명성"],
            "source_description": "2025 자율점검표 부록4 사례15"
        },
        {
            "case_id": "appendix_case_16_deepfake_crime",
            "case_name": "딥페이크 성범죄 사건 (자율점검표 부록4 사례16)",
            "content_summary": "생성 AI로 아동 성 착취 영상물 360여개를 제작한 40대 남성에게 징역 2년 6월 실형 선고('23.9). 텔레그램 참여자들로부터 피해자들의 개인정보를 넘겨받아 이를 이용해 아동·청소년 대상 허위 영상물 92개와 성인 대상 허위 영상물 1,275개를 제작·유포한 20대 남성 구속기소 ('24.9).", # 원문 날짜가 24.9로 되어있어 그대로 사용합니다.
            "ethical_considerations": "악의적 이용자에 대한 사전적 및 사후적 대응 체계 마련 필요. AI 생성물 표시 및 사후 추적을 위한 최신의 기술적 수단 도입 필요.",
            "related_core_requirements": ["인권보장", "프라이버시 보호", "침해금지"],
            "source_description": "2025 자율점검표 부록4 사례16"
        },
        {
            "case_id": "appendix_case_17_cfo_deepfake_fraud",
            "case_name": "다국적 금융기업 CFO 사칭 금융사기 (자율점검표 부록4 사례17)",
            "content_summary": "홍콩의 다국적 금융기업 직원이 최고 재무 책임자(CFO)를 사칭한 이메일을 통해 거액의 자금 이체할 것을 요구받은 뒤, 딥페이크 기술을 이용하여 CFO의 모습과 목소리를 완벽하게 모방한 화상회의에서도 동일한 지시를 받자 340억 원을 송금하는 사건 발생('24.2).",
            "ethical_considerations": "기술 오용 방지를 위한 조치 필요. 피해가 발생했을 때 피해 확산을 방지할 수 있는 절차 마련 필요.",
            "related_core_requirements": ["침해금지", "공공성", "책임성", "안전성"],
            "source_description": "2025 자율점검표 부록4 사례17"
        },
        {
            "case_id": "appendix_case_18_digital_resurrection_of_deceased",
            "case_name": "고인(故人)을 AI로 재현하는 디지털 부활 (자율점검표 부록4 사례18)",
            "content_summary": "한 블로거가 유가족의 동의 없이 AI 기술을 이용해 중국 가수 고(故) 차오런량(乔任梁)을 재현하고, 팬들에게 안부 인사를 전하는 영상을 제작해 논란 발생('24.3). 영화 '에이리언: 로물루스' 제작진은 유족의 허락을 받아 배우 고(故) 이언 홈을 AI 기술로 구현해 영화에 등장시켰으나, 일부 관객과 비평가들이 거부감을 드러내며 윤리적 논란 제기('24.8).", # 원문 날짜가 24.8로 되어있어 그대로 사용합니다.
            "ethical_considerations": "기술의 오남용 방지를 위한 조치 필요. 디지털 부활에 대한 사회적 수용도 및 윤리적 쟁점에 대한 고려 필요.",
            "related_core_requirements": ["프라이버시 보호", "침해금지"],
            "source_description": "2025 자율점검표 부록4 사례18"
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
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # core 폴더 기준 상위 폴더의 .env
    
    # 현재 파일(db_utils.py)이 core 폴더에 있고, .env 파일이 프로젝트 루트에 있다고 가정
    # 프로젝트 루트 경로를 얻는 더 견고한 방법
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    dotenv_path = os.path.join(project_root, '.env')

    if os.path.exists(dotenv_path):
        print(f"Loading .env from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Ensure API keys are set in environment.")

    OPENAI_API_KEY_TEST = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY_TEST:
        print("Warning: OPENAI_API_KEY not found after .env load attempt in test block.")
    
    # config 값들이 환경 변수 로드 후 반영되도록 initialize_vector_databases 함수 내부에서
    # client 및 collection이 생성될 때 최신 config 값을 사용하게 됩니다.
    # 만약 config.py의 값들이 모듈 로드 시점에 고정된다면,
    # initialize_vector_databases를 호출하기 전에 config 모듈을 reload 하거나,
    # config 값들을 함수 인자로 전달하는 방식을 고려해야 합니다.
    # 현재 코드는 전역 변수를 사용하므로, 스크립트 실행 시점의 환경 변수를 따릅니다.

    initialize_vector_databases()

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
        past_case_results = search_documents("챗봇 개인정보 유출 사례", past_cases_collection, n_results=2) # 2개 결과 요청
        if past_case_results:
            for i, res in enumerate(past_case_results):
                print(f"  Result {i+1}: (Distance: {res['distance']:.4f})")
                print(f"    Case Name: {res['metadata'].get('case_name')}")
                print(f"    Source: {res['metadata'].get('source_description')}")
                print(f"    Content Summary: {res['metadata'].get('content_summary')[:150]}...")
        else:
            print("  No results found or error occurred in past cases search.")
    else:
        print("  Past cases collection not available for search test.")
