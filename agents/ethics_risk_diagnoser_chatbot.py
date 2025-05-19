from typing import List, Optional, Dict, Any

try:
    from ..core.states import ProjectState, PastCaseAnalysis # 필요한 타입 임포트
    from ..core.db_utils import search_documents, checklist_collection, past_cases_collection # DB 유틸리티
    from ..core.config import LLM_MODEL_NAME, OPENAI_API_KEY # LLM 설정
except ImportError:
    from core.states import ProjectState, PastCaseAnalysis
    from core.db_utils import search_documents, checklist_collection, past_cases_collection
    from core.config import LLM_MODEL_NAME, OPENAI_API_KEY

from langchain_openai import ChatOpenAI

class EthicsRiskDiagnoserChatbotAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.2, openai_api_key=OPENAI_API_KEY)
        print("EthicsRiskDiagnoserChatbotAgent initialized.")

    async def diagnose(self, state: ProjectState) -> ProjectState:
        print("--- Ethics Risk Diagnoser Agent (Chatbot Focus): 진단 시작 ---")
        pending_diagnoses: Optional[List[str]] = state.get("pending_specific_diagnoses")
        analyzed_info = state.get("analyzed_service_info")

        current_diagnosis_category = ""
        if pending_diagnoses and len(pending_diagnoses) > 0:
            current_diagnosis_category = pending_diagnoses.pop(0) # 큐에서 첫 번째 항목을 꺼내고 제거
            print(f"  처리 중인 챗봇 특화 진단 카테고리: {current_diagnosis_category}")
            state["pending_specific_diagnoses"] = pending_diagnoses # 변경된 큐를 상태에 반영
        else:
            print("  Warning: 챗봇 특화 진단 큐가 비어있습니다.")
            return state # 더 이상 처리할 항목 없음

        # 여기에 current_diagnosis_category (예: "bias_chatbot" 또는 "privacy_chatbot")에 따른
        # 실제 심층 진단 로직 (LLM 호출, DB 검색 등) 구현
        # 예시: "bias_chatbot" 진단
        if current_diagnosis_category == "bias_chatbot":
            print(f"  심층 진단 수행: {current_diagnosis_category} for {state.get('service_name')}")
            # 1. 관련된 자율점검표 항목 검색 (checklist_collection)
            # 2. 관련된 과거 사례 검색 (past_cases_collection, "챗봇 편향" 등 키워드)
            # 3. LLM을 사용하여 분석 및 결과 생성
            bias_analysis_result = f"'{state.get('service_name')}'의 {current_diagnosis_category} 분석 결과: (상세 내용)..."

            # 결과를 specific_ethics_risks_by_category에 누적
            if state.get("specific_ethics_risks_by_category") is None:
                state["specific_ethics_risks_by_category"] = {}
            state["specific_ethics_risks_by_category"][current_diagnosis_category] = {"analysis": bias_analysis_result, "severity": "medium"} # 예시

            # 과거 사례 분석 결과도 유사하게 추가
            if state.get("past_case_analysis_results_by_category") is None:
                state["past_case_analysis_results_by_category"] = {}
            state["past_case_analysis_results_by_category"][current_diagnosis_category] = [
                PastCaseAnalysis(case_name="예시 챗봇 편향 사례", cause_and_structure="...", vulnerabilities="...", comparison_with_target_service="...", key_insights="...")
            ]
            print(f"  {current_diagnosis_category} 진단 결과 저장됨.")

        elif current_diagnosis_category == "privacy_chatbot":
            print(f"  심층 진단 수행: {current_diagnosis_category} for {state.get('service_name')}")
            privacy_analysis_result = f"'{state.get('service_name')}'의 {current_diagnosis_category} 분석 결과: (상세 내용)..."
            if state.get("specific_ethics_risks_by_category") is None:
                state["specific_ethics_risks_by_category"] = {}
            state["specific_ethics_risks_by_category"][current_diagnosis_category] = {"analysis": privacy_analysis_result, "severity": "high"}

            if state.get("past_case_analysis_results_by_category") is None:
                state["past_case_analysis_results_by_category"] = {}
            state["past_case_analysis_results_by_category"][current_diagnosis_category] = [
                PastCaseAnalysis(case_name="예시 챗봇 개인정보 유출 사례", cause_and_structure="...", vulnerabilities="...", comparison_with_target_service="...", key_insights="...")
            ]
            print(f"  {current_diagnosis_category} 진단 결과 저장됨.")

        # 공통 진단 로직 (예시, 모든 특화 진단 노드에서 호출하거나 별도 노드로 분리)
        if state.get("common_ethics_risks") is None:
            state["common_ethics_risks"] = {}
        state["common_ethics_risks"]["E01.01_human_dignity_chatbot"] = "챗봇 서비스는 사용자의 인격적 대우를 고려해야 함 (검토 필요)." # 예시 결과

        print("--- Ethics Risk Diagnoser Agent (Chatbot Focus): 진단 완료 ---")
        return state