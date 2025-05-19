import json
from typing import Dict, Optional, List
from langchain_openai import ChatOpenAI

try:
    from ..core.states import ProjectState, ServiceAnalysisOutput
    from ..core.config import OPENAI_API_KEY, LLM_MODEL_NAME, TOP_K_RESULTS
    from ..core.db_utils import search_documents, checklist_collection # checklist_collection 직접 사용
    from ..prompts.common_diagnoser_prompts import COMMON_ETHICS_DIAGNOSIS_PROMPT_TEMPLATE
except ImportError:
    from core.states import ProjectState, ServiceAnalysisOutput
    from core.config import OPENAI_API_KEY, LLM_MODEL_NAME, TOP_K_RESULTS
    from core.db_utils import search_documents, checklist_collection
    from prompts.common_diagnoser_prompts import COMMON_ETHICS_DIAGNOSIS_PROMPT_TEMPLATE

class CommonEthicsDiagnoserAgent:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.2, openai_api_key=OPENAI_API_KEY)
        print("CommonEthicsDiagnoserAgent initialized.")

    async def diagnose_common_risks(self, state: ProjectState) -> ProjectState:
        print("--- Common Ethics Diagnoser Agent: 공통 윤리 리스크 진단 시작 ---")
        analyzed_info: Optional[ServiceAnalysisOutput] = state.get("analyzed_service_info")
        service_name = state.get("service_name", "알 수 없는 서비스")
        service_type = state.get("service_type", "기타")

        if not analyzed_info:
            print("  Warning: 서비스 분석 정보가 없습니다. 공통 진단을 건너<0xEB><0><0x8A><0x8D>니다.")
            state["common_ethics_risks"] = {"error": "서비스 분석 정보 누락"}
            return state

        # 자율점검표 10대 핵심 요건 키워드 (RAG 검색용)
        ethics_core_principles = [
            "인권보장", "프라이버시 보호", "다양성 존중", "침해금지", "공공성",
            "연대성", "데이터 관리", "책임성", "안전성", "투명성"
        ]

        retrieved_checklist_items_str = "\n"
        print("  자율점검표에서 관련 항목 검색 중...")
        if checklist_collection:
            # 각 핵심 원칙에 대해 검색하거나, 서비스 설명 전체로 한 번 검색할 수 있음
            # 여기서는 서비스 설명과 각 원칙을 조합하여 검색
            query_for_rag = f"{service_name} ({service_type}) 서비스의 {', '.join(ethics_core_principles)} 관련 윤리적 고려사항"

            # 더 나은 방법: 각 원칙별로 검색하여 관련성 높은 내용을 모음
            all_retrieved_docs = []
            for principle in ethics_core_principles:
                principle_query = f"{service_name} 서비스의 {principle} 원칙 관련 내용"
                # TOP_K_RESULTS를 각 원칙별로 가져오면 너무 많을 수 있으므로 조절 (예: 1~2개)
                results = search_documents(principle_query, checklist_collection, n_results=1) # 각 원칙당 1개씩
                if results:
                    all_retrieved_docs.extend([res['document'] for res in results])

            if all_retrieved_docs:
                # 중복 제거 및 결합
                unique_docs = list(set(all_retrieved_docs))
                retrieved_checklist_items_str += "\n\n--- 자율점검표 참고 항목 ---\n"
                for i, doc_content in enumerate(unique_docs[:TOP_K_RESULTS]): # 전체 검색 결과 중 상위 N개만 사용
                    retrieved_checklist_items_str += f"항목 {i+1}:\n{doc_content}\n\n"
                print(f"  {len(unique_docs)}개의 고유한 참고 항목 검색됨 (최대 {TOP_K_RESULTS}개 사용).")
            else:
                retrieved_checklist_items_str += "자율점검표에서 직접적으로 관련된 항목을 찾지 못했습니다. 일반 원칙에 따라 평가합니다.\n"
        else:
            print("  Warning: Checklist collection이 초기화되지 않았습니다. RAG 검색을 건너<0xEB><0><0x8A><0x8D>니다.")
            retrieved_checklist_items_str += "자율점검표 데이터베이스에 접근할 수 없습니다.\n"


        prompt = COMMON_ETHICS_DIAGNOSIS_PROMPT_TEMPLATE.format(
            service_name=service_name,
            service_type=service_type,
            target_functions=analyzed_info.get("target_functions", "N/A"),
            key_features=analyzed_info.get("key_features", "N/A"),
            data_usage_estimation=analyzed_info.get("data_usage_estimation", "N/A"),
            estimated_users_stakeholders=analyzed_info.get("estimated_users_stakeholders", "N/A"),
            retrieved_checklist_items=retrieved_checklist_items_str
        )

        try:
            print("  LLM 호출하여 공통 윤리 리스크 분석 중...")
            response = await self.llm.ainvoke(prompt)
            response_content = str(response.content)
            print(f"  LLM 응답 수신 (일부): {response_content[:200]}...")

            try:
                json_start_index = response_content.find('{')
                json_end_index = response_content.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_str = response_content[json_start_index:json_end_index]
                    common_risks_dict = json.loads(json_str)
                    state["common_ethics_risks"] = common_risks_dict
                    print("  공통 윤리 리스크 진단 결과가 상태에 저장되었습니다.")
                else:
                    print(f"  LLM 응답에서 유효한 JSON 객체를 찾을 수 없습니다. 응답: {response_content}")
                    state["error_message"] = "공통 진단 LLM 응답에서 JSON 파싱 실패"
                    state["common_ethics_risks"] = None
            except json.JSONDecodeError as e:
                print(f"  LLM 응답 JSON 파싱 오류: {e}")
                state["error_message"] = f"공통 진단 LLM 응답 JSON 파싱 오류: {e}"
                state["common_ethics_risks"] = None
        except Exception as e:
            print(f"  LLM 호출 중 오류 발생: {e}")
            state["error_message"] = f"공통 진단 LLM 호출 오류: {e}"
            state["common_ethics_risks"] = None

        print("--- Common Ethics Diagnoser Agent: 공통 윤리 리스크 진단 완료 ---")
        return state

# --- 테스트용 ---
@staticmethod
async def run_standalone_test():
    import asyncio
    from dotenv import load_dotenv
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path): load_dotenv(dotenv_path=dotenv_path)

    try:
        from ..core.db_utils import initialize_vector_databases
    except ImportError:
        from core.db_utils import initialize_vector_databases

    if not OPENAI_API_KEY:
        print("Standalone Test Error: OPENAI_API_KEY is not set in config.")
        return

    if checklist_collection is None or past_cases_collection is None:
        print("Test: Initializing vector databases for common diagnoser test...")
        initialize_vector_databases()


    initial_state: ProjectState = {
        "service_name": "AI 여행 추천 챗봇 '여행친구'",
        "service_type": "챗봇",
        "service_initial_info": "사용자의 취향과 예산에 맞춰 맞춤형 여행지를 추천하고, 항공권 및 숙소 예약 링크를 제공하는 대화형 AI 서비스입니다.",
        "analyzed_service_info": {
            "target_functions": "맞춤형 여행지 추천, 항공/숙소 정보 제공, 여행 관련 Q&A",
            "key_features": "자연어 이해 기반 대화형 추천, 개인화된 결과, 원스톱 정보 제공 시도",
            "data_usage_estimation": "사용자 대화 내용, 선호 여행 스타일, 과거 여행 기록(수집 시 동의 필요), 예산 정보, 위치 정보(선택적), 외부 예약 사이트 연동 데이터(추정됨)",
            "estimated_users_stakeholders": "자유 여행객, 특정 목적 여행객(신혼여행, 가족여행 등), 여행사, 항공사, 숙박업체, 지역 관광청"
        },
        "main_risk_categories": ["bias_chatbot", "privacy_chatbot"],
        "pending_specific_diagnoses": ["bias_chatbot", "privacy_chatbot"],
        "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
    }

    diagnoser_agent = CommonEthicsDiagnoserAgent()
    updated_state = await diagnoser_agent.diagnose_common_risks(initial_state)

    print("\n--- 최종 상태 (Common Ethics Diagnoser 테스트 후) ---")
    if updated_state.get("error_message"):
        print(f"  에러: {updated_state['error_message']}")

    common_risks = updated_state.get("common_ethics_risks")
    if common_risks:
        print("  공통 윤리 리스크 진단 결과 (JSON):")
        print(json.dumps(common_risks, indent=2, ensure_ascii=False))
    else:
        print("  공통 윤리 리스크가 생성되지 않았습니다.")

if __name__ == '__main__':
    import asyncio
    asyncio.run(CommonEthicsDiagnoserAgent.run_standalone_test())