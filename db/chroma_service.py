# ChromaDB 서비스 유틸리티 (윤리 진단용)
# past_cases_collection, search_documents, TOP_K_RESULTS 제공
from typing import Any, List
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os

# ChromaDB 경로 및 모델 지정
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'chromadb')
EMBEDDING_MODEL_NAME = 'jhgan/ko-sroberta-multitask'

# TOP_K_RESULTS: 검색 결과 상위 N개
TOP_K_RESULTS = 3

# ChromaDB 클라이언트 및 임베딩 함수 초기화
client = PersistentClient(path=CHROMA_DB_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

# past_cases_collection 로드
try:
    past_cases_collection = client.get_collection('past_cases_collection', embedding_function=embedding_func)
except Exception as e:
    print(f"ChromaDB: past_cases_collection 로드 실패: {e}")
    past_cases_collection = None

def search_documents(query: str, collection, n_results: int = TOP_K_RESULTS) -> List[Any]:
    """
    주어진 쿼리(query)에 대해 ChromaDB 컬렉션에서 유사한 문서를 검색합니다.
    """
    if collection is None:
        return []
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        # 결과에서 문서 정보 추출
        docs = []
        for i in range(len(results.get('ids', [[]])[0])):
            doc = {
                'id': results['ids'][0][i] if results.get('ids') else None,
                'document': results['documents'][0][i] if results.get('documents') else None,
                'metadata': results['metadatas'][0][i] if results.get('metadatas') else None
            }
            docs.append(doc)
        return docs
    except Exception as e:
        print(f"ChromaDB: 문서 검색 오류: {e}")
        return []