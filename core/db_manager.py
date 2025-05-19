# core/db_manager.py
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader # 또는 다른 PDF 로더
from core.config import CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, PDF_DATA_PATH, \
                        CHROMA_CHECKLIST_COLLECTION_NAME, CHROMA_PAST_CASES_COLLECTION_NAME
import os
from typing import List, Dict, Any

# 임베딩 함수 설정 (jhgan/ko-sroberta-multitask 사용)
# SentenceTransformerEmbeddingFunction은 로컬 모델 사용 시 CPU를 사용하며, GPU 설정은 추가 구성 필요
# huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
#     api_key="YOUR_HF_TOKEN_IF_NEEDED_FOR_PRIVATE_MODELS", # 공개 모델은 필요 없음
#     model_name=EMBEDDING_MODEL_NAME
# )
# LangChain의 HuggingFaceEmbeddings를 사용하는 것이 더 일반적일 수 있음
# 여기서는 chromadb.utils의 것을 예시로 들지만, LangChain 통합을 고려하여 변경 가능

# LangChain의 SentenceTransformerEmbeddings 사용 권장
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ChromaDB 클라이언트 초기화 (PersistentClient 사용)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# 임베딩 모델 초기화
embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_or_create_collection(collection_name: str):
    """지정된 이름의 컬렉션을 가져오거나 생성합니다."""
    try:
        collection = client.get_collection(name=collection_name, embedding_function=None) # 임베딩은 직접 처리
        print(f"Collection '{collection_name}' loaded.")
    except: # chromadb.errors.CollectionNotFoundError 등 구체적 예외 처리 필요
        print(f"Collection '{collection_name}' not found. Creating new one.")
        # embedding_function은 add 시점에 metadata로 지정 가능, 또는 get_collection 시 지정.
        # 여기서는 add 시점에 임베딩을 직접 생성하므로, 컬렉션 생성 시에는 명시적 임베딩 함수 지정 불필요.
        collection = client.create_collection(name=collection_name)
    return collection

def load_pdf_and_split(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """PDF 파일을 로드하고 텍스트를 청크로 분할합니다."""
    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return []

    try:
        # UnstructuredPDFLoader 또는 PyPDFLoader 사용
        # loader = UnstructuredPDFLoader(file_path, mode="elements") # 요소별로 더 잘게 나눔
        loader = PyPDFLoader(file_path) # 페이지별로 나눔
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks_with_metadata = []
    for doc in documents:
        page_content = doc.page_content
        metadata = doc.metadata # 페이지 번호 등
        
        split_texts = text_splitter.split_text(page_content)
        for i, text_chunk in enumerate(split_texts):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index_in_page"] = i
            # 원본 문서의 어느 부분인지 알 수 있도록 메타데이터 추가 가능
            # 첫 몇 글자로 미리보기 추가 
            chunks_with_metadata.append({
                "document": text_chunk,
                "metadata": chunk_metadata,
                "id": f"{os.path.basename(file_path)}_page{metadata.get('page',0)}_chunk{i}" # 고유 ID 생성
            })
    print(f"Loaded and split {file_path}: {len(chunks_with_metadata)} chunks created.")
    return chunks_with_metadata

def add_documents_to_collection(collection_name: str, documents_with_metadata: List[Dict[str, Any]]):
    """문서 청크와 메타데이터를 컬렉션에 추가합니다. (임베딩 포함)"""
    collection = get_or_create_collection(collection_name)
    
    if not documents_with_metadata:
        print(f"No documents to add to collection '{collection_name}'.")
        return

    docs_to_add = [item["document"] for item in documents_with_metadata]
    metadatas_to_add = [item["metadata"] for item in documents_with_metadata]
    ids_to_add = [item["id"] for item in documents_with_metadata]
    
    # 문서 임베딩 (LangChain 임베딩 모델 사용)
    embeddings_to_add = embedding_model.embed_documents(docs_to_add)
    
    try:
        # 동일 ID가 이미 존재하면 업데이트하도록 수정 (ChromaDB는 기본적으로 upsert 지원 안함, add는 중복 ID 에러 발생)
        # 간단하게는 기존 ID 삭제 후 추가 또는, 컬렉션의 add/update/upsert 동작 확인 필요.
        # 여기서는 간단히 add를 시도하고, 이미 데이터가 있다면 추가하지 않도록 할 수 있음.
        # 또는, 컬렉션을 비우고 다시 로드하는 스크립트를 별도로 관리.
        # 데모용으로, 실행 시마다 컬렉션을 초기화하고 다시 로드한다고 가정.
        
        # 먼저 해당 ID들이 있는지 확인하고, 있다면 업데이트, 없다면 추가 (더 복잡한 로직)
        # 지금은 간단히 add 시도
        existing_ids = set(collection.get(ids=ids_to_add)['ids']) if ids_to_add else set()
        
        new_docs_to_add = []
        new_embeddings_to_add = []
        new_metadatas_to_add = []
        new_ids_to_add = []

        for i, doc_id in enumerate(ids_to_add):
            if doc_id not in existing_ids:
                new_docs_to_add.append(docs_to_add[i])
                new_embeddings_to_add.append(embeddings_to_add[i])
                new_metadatas_to_add.append(metadatas_to_add[i])
                new_ids_to_add.append(ids_to_add[i])
        
        if new_ids_to_add:
            collection.add(
                embeddings=new_embeddings_to_add,
                documents=new_docs_to_add,
                metadatas=new_metadatas_to_add,
                ids=new_ids_to_add
            )
            print(f"Added {len(new_ids_to_add)} new documents to collection '{collection_name}'.")
        else:
            print(f"No new documents to add. All provided IDs might already exist in '{collection_name}'.")

    except Exception as e:
        print(f"Error adding documents to collection '{collection_name}': {e}")


def query_collection(collection_name: str, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """컬렉션에서 쿼리 텍스트와 유사한 문서를 검색합니다."""
    collection = get_or_create_collection(collection_name)
    
    # 쿼리 텍스트 임베딩
    query_embedding = embedding_model.embed_query(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding], # 리스트 형태로 전달
        n_results=n_results,
        include=['documents', 'metadatas', 'distances'] # 필요한 정보 포함
    )
    
    # 결과 포맷팅 (LangChain의 Document 객체 형태로 변환하거나, 필요한 dict 형태로 가공)
    formatted_results = []
    if results and results.get('ids')[0]: # 결과가 있을 때
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
    return formatted_results

def initialize_checklist_db():
    """'자율점검표' PDF를 ChromaDB에 로드합니다."""
    print("Initializing Checklist DB...")
    chunks = load_pdf_and_split(PDF_DATA_PATH)
    if chunks:
        # 이전 컬렉션이 있다면 삭제하고 새로 생성 (데이터 중복 방지 및 업데이트 용이)
        try:
            client.delete_collection(name=CHROMA_CHECKLIST_COLLECTION_NAME)
            print(f"Collection '{CHROMA_CHECKLIST_COLLECTION_NAME}' deleted for re-initialization.")
        except: # chromadb.errors.CollectionNotFoundError 등
            pass # 없으면 그냥 진행
        add_documents_to_collection(CHROMA_CHECKLIST_COLLECTION_NAME, chunks)
    else:
        print("Failed to load or split checklist PDF. DB not initialized.")

def initialize_past_cases_db(past_cases_data: List[Dict[str, Any]]):
    """
    제공된 과거 사례 데이터를 ChromaDB에 로드합니다.
    past_cases_data: [{"document": "사례 내용...", "metadata": {"case_name": "...", "source": "..."}, "id": "unique_id"}, ...]
    """
    print("Initializing Past Cases DB...")
    if past_cases_data:
        try:
            client.delete_collection(name=CHROMA_PAST_CASES_COLLECTION_NAME)
            print(f"Collection '{CHROMA_PAST_CASES_COLLECTION_NAME}' deleted for re-initialization.")
        except:
            pass
        add_documents_to_collection(CHROMA_PAST_CASES_COLLECTION_NAME, past_cases_data)
    else:
        print("No past cases data provided. Past Cases DB not initialized with new data.")

# --- 데이터베이스 초기화 실행 (애플리케이션 시작 시 한 번 또는 별도 스크립트로) ---
# if __name__ == "__main__":
#     initialize_checklist_db()
    
#     # 예시 과거 사례 데이터 (실제로는 파일에서 읽거나 다른 방식으로 준비)
#     sample_past_cases = [
#         {
#             "document": "이루다 챗봇 개인정보 유출 및 혐오 발언 논란. 학습 데이터의 개인정보 비식별화 미흡 및 사용자 동의 문제가 지적됨.",
#             "metadata": {"case_name": "이루다 챗봇 논란", "source": "뉴스 기사 종합", "year": 2021, "risk_type": "privacy,bias"},
#             "id": "iruda_case_2021"
#         },
#         {
#             "document": "아마존 AI 채용 시스템 성차별 논란. 과거 남성 위주의 데이터를 학습하여 여성 지원자에게 불리한 평가를 내림.",
#             "metadata": {"case_name": "아마존 AI 채용 차별", "source": "Reuters", "year": 2018, "risk_type": "bias,discrimination"},
#             "id": "amazon_hr_ai_bias_2018"
#         }
#     ]
#     initialize_past_cases_db(sample_past_cases)

#     # 테스트 쿼리
#     checklist_results = query_collection(CHROMA_CHECKLIST_COLLECTION_NAME, "프라이버시 침해 방지", n_results=2)
#     print("\nChecklist Query Results for '프라이버시 침해 방지':")
#     for res in checklist_results:
#         print(f"  ID: {res['id']}, Distance: {res['distance']:.4f}, Page: {res['metadata'].get('page')}")
#         print(f"    Doc: {res['document'][:100]}...")

#     past_case_results = query_collection(CHROMA_PAST_CASES_COLLECTION_NAME, "챗봇 개인정보 문제", n_results=1)
#     print("\nPast Case Query Results for '챗봇 개인정보 문제':")
#     for res in past_case_results:
#         print(f"  ID: {res['id']}, Case: {res['metadata'].get('case_name')}, Distance: {res['distance']:.4f}")
#         print(f"    Doc: {res['document'][:100]}...")