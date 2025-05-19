from typing import List, Optional, Dict, Any

try:
    from ..core.states import ProjectState, PastCaseAnalysis # 필요한 타입 임포트
    from ..core.db_utils import search_documents, checklist_collection, past_cases_collection # DB 유틸리티 (필요시 사용)
    from ..core.config import LLM_MODEL_NAME, OPENAI_API_KEY # LLM 설정
except ImportError: # 직접 실행 또는 테스트 시
    from core.states import ProjectState, PastCaseAnalysis
    from core.db_utils import search_documents, checklist_collection, past_cases_collection
    from core.config import LLM_MODEL_NAME, OPENAI_API_KEY

from langchain_openai import ChatOpenAI # LLM 사용을 위해

class EthicsRiskDiagnoserRecSysAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.2, openai_api_key=OPENAI_API_KEY)
        print("EthicsRiskDiagnoserRecSysAgent initialized.")

    async def diagnose(self, state: ProjectState) -> ProjectState:
        print("--- Ethics Risk Diagnoser Agent (RecSys Focus): 진단 시작 ---")
        pending_diagnoses: Optional[List[str]] = state.get("pending_specific_diagnoses")
        analyzed_info = state.get("analyzed_service_info") # 서비스 분석 정보 (필요시 활용)
        service_name = state.get("service_name", "알 수 없는 서비스")

        current_diagnosis_category = ""
        if pending_diagnoses and len(pending_diagnoses) > 0:
            current_diagnosis_category = pending_diagnoses.pop(0) # 큐에서 첫 번째 항목을 꺼내고 제거
            print(f"  처리 중인 추천 시스템 특화 진단 카테고리: {current_diagnosis_category}")
            state["pending_specific_diagnoses"] = pending_diagnoses # 변경된 큐를 상태에 반영
        else:
            print("  Warning: 추천 시스템 특화 진단 큐가 비어있습니다.")
            return state # 더 이상 처리할 항목 없음

        # current_diagnosis_category에 따른 실제 심층 진단 로직
        if current_diagnosis_category == "filter_bubble_recommendation":
            print(f"  심층 진단 수행: {current_diagnosis_category} for {service_name}")
            # 1. 관련된 자율점검표 항목 검색 (예: checklist_collection.query(...))
            # 2. 관련된 과거 사례 검색 (예: past_cases_collection.query("추천 시스템 필터버블" 등))
            # 3. LLM을 사용하여 분석 및 결과 생성 (아래는 예시 결과)
            analysis_result_text = f"'{service_name}'의 필터버블(filter_bubble_recommendation) 분석 결과: 사용자가 특정 정보에만 과도하게 노출될 위험이 있으며, 다양한 관점의 정보를 접할 기회가 줄어들 수 있습니다. (상세 내용 추가 필요)"
            severity_level = "medium" # 예시 심각도

            # 결과를 specific_ethics_risks_by_category에 누적
            if state.get("specific_ethics_risks_by_category") is None:
                state["specific_ethics_risks_by_category"] = {}
            state["specific_ethics_risks_by_category"][current_diagnosis_category] = {
                "analysis": analysis_result_text,
                "severity": severity_level
            }

            # 과거 사례 분석 결과도 유사하게 추가
            if state.get("past_case_analysis_results_by_category") is None:
                state["past_case_analysis_results_by_category"] = {}
            state["past_case_analysis_results_by_category"][current_diagnosis_category] = [
                PastCaseAnalysis(
                    case_name="예시 추천 시스템 필터버블 사례",
                    cause_and_structure="알고리즘이 사용자의 과거 행동에만 지나치게 의존하여 유사한 콘텐츠만 반복적으로 추천함.",
                    vulnerabilities="다양한 정보 소스에 대한 접근성이 낮고, 사용자의 관심사를 확장하려는 노력이 부족함.",
                    comparison_with_target_service=f"'{service_name}' 또한 개인화 추천에 중점을 두므로 유사한 필터버블 발생 가능성 검토 필요.",
                    key_insights="필터버블은 사용자의 정보 편식을 심화시키고 사회적 고립을 야기할 수 있음. 알고리즘의 다양성 확보 및 사용자 선택권 강화 필요."
                )
            ]
            print(f"  {current_diagnosis_category} 진단 결과 저장됨.")

        elif current_diagnosis_category == "transparency_recommendation":
            print(f"  심층 진단 수행: {current_diagnosis_category} for {service_name}")
            analysis_result_text = f"'{service_name}'의 추천 기준 투명성(transparency_recommendation) 분석 결과: 추천 결과가 어떤 기준으로 제공되는지 사용자가 명확히 알기 어려워 신뢰도 저하 및 오해를 유발할 수 있습니다. (상세 내용 추가 필요)"
            severity_level = "low" # 예시 심각도

            if state.get("specific_ethics_risks_by_category") is None:
                state["specific_ethics_risks_by_category"] = {}
            state["specific_ethics_risks_by_category"][current_diagnosis_category] = {
                "analysis": analysis_result_text,
                "severity": severity_level
            }

            if state.get("past_case_analysis_results_by_category") is None:
                state["past_case_analysis_results_by_category"] = {}
            state["past_case_analysis_results_by_category"][current_diagnosis_category] = [
                PastCaseAnalysis(
                    case_name="예시 추천 시스템 투명성 부족 사례",
                    cause_and_structure="추천 로직이 복잡하고 비공개되어 사용자가 추천 이유를 이해하기 어려움.",
                    vulnerabilities="사용자가 추천 결과를 통제하거나 피드백을 제공할 수 있는 효과적인 수단이 부족함.",
                    comparison_with_target_service=f"'{service_name}'의 추천 로직 공개 수준 및 사용자 설명 방식 검토 필요.",
                    key_insights="추천 기준의 투명성 제고는 사용자 신뢰 확보 및 서비스 만족도 향상에 기여함. 설명 가능한 AI(XAI) 원칙 적용 고려."
                )
            ]
            print(f"  {current_diagnosis_category} 진단 결과 저장됨.")
        
        else:
            print(f"  Warning: 알 수 없는 추천 시스템 특화 진단 카테고리입니다: {current_diagnosis_category}")


        # 공통 진단 로직 (필요시 추가, 예시)
        # if state.get("common_ethics_risks") is None:
        #     state["common_ethics_risks"] = {}
        # state["common_ethics_risks"]["E02.01_user_autonomy_recsys"] = "추천 시스템은 사용자의 자율적 선택을 존중해야 함 (검토 필요)."

        print("--- Ethics Risk Diagnoser Agent (RecSys Focus): 진단 완료 ---")
        return state

# --- 에이전트 테스트용 (직접 실행 시) ---
async def run_standalone_test():
    # 테스트를 위한 초기 상태 (RiskClassifierAgent 실행 후 상태)
    initial_state_filter_bubble: ProjectState = {
        "service_name": "뉴스피드 추천 서비스 '뉴스콕'",
        "service_type": "추천 알고리즘",
        "analyzed_service_info": {
            "target_functions": "개인 맞춤형 뉴스 기사 추천",
            "key_features": "실시간 관심사 분석, 유사 사용자 그룹핑",
            "data_usage_estimation": "기사 열람 기록, 검색어, 소셜 활동 데이터",
            "estimated_users_stakeholders": "뉴스 구독자, 언론사, 광고주"
        },
        "main_risk_categories": ["filter_bubble_recommendation", "transparency_recommendation"],
        "pending_specific_diagnoses": ["filter_bubble_recommendation"], # 이 테스트에서는 필터버블만 진단
        "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None,
        "common_ethics_risks": None,
        "error_message": None
    }

    initial_state_transparency: ProjectState = {
        "service_name": "영화 추천 서비스 '무비나잇'",
        "service_type": "추천 알고리즘",
        "analyzed_service_info": {
            "target_functions": "사용자 취향 기반 영화 추천",
            "key_features": "별점 예측, 콘텐츠 기반 필터링, 협업 필터링",
            "data_usage_estimation": "시청 기록, 별점, 선호 장르",
            "estimated_users_stakeholders": "영화 시청자, 콘텐츠 제공업체"
        },
        "main_risk_categories": ["filter_bubble_recommendation", "transparency_recommendation"],
        "pending_specific_diagnoses": ["transparency_recommendation"], # 투명성만 진단
        "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None,
        "common_ethics_risks": None,
        "error_message": None
    }

    diagnoser_agent = EthicsRiskDiagnoserRecSysAgent()

    print("\n--- Filter Bubble Diagnosis Test ---")
    updated_state_fb = await diagnoser_agent.diagnose(initial_state_filter_bubble)
    if updated_state_fb.get("error_message"):
        print(f"  Error: {updated_state_fb['error_message']}")
    if updated_state_fb.get("specific_ethics_risks_by_category"):
        print(f"  Specific Risks (Filter Bubble): {updated_state_fb['specific_ethics_risks_by_category'].get('filter_bubble_recommendation')}")
    if updated_state_fb.get("past_case_analysis_results_by_category"):
        print(f"  Past Cases (Filter Bubble): {updated_state_fb['past_case_analysis_results_by_category'].get('filter_bubble_recommendation')}")
    print(f"  Pending diagnoses after: {updated_state_fb.get('pending_specific_diagnoses')}")


    print("\n--- Transparency Diagnosis Test ---")
    updated_state_tr = await diagnoser_agent.diagnose(initial_state_transparency)
    if updated_state_tr.get("error_message"):
        print(f"  Error: {updated_state_tr['error_message']}")
    if updated_state_tr.get("specific_ethics_risks_by_category"):
        print(f"  Specific Risks (Transparency): {updated_state_tr['specific_ethics_risks_by_category'].get('transparency_recommendation')}")
    if updated_state_tr.get("past_case_analysis_results_by_category"):
        print(f"  Past Cases (Transparency): {updated_state_tr['past_case_analysis_results_by_category'].get('transparency_recommendation')}")
    print(f"  Pending diagnoses after: {updated_state_tr.get('pending_specific_diagnoses')}")


if __name__ == '__main__':
    import asyncio
    import os
    from dotenv import load_dotenv
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded .env from {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}. OPENAI_API_KEY must be set via environment.")

    if not OPENAI_API_KEY: 
        print("Standalone Test Error: OPENAI_API_KEY is not set. Please set it in your .env file or environment.")
    else:
        asyncio.run(run_standalone_test())