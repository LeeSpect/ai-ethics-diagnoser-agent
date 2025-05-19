# agents/improvement_suggester.py
import json
from typing import Dict, Optional, Any
from langchain_openai import ChatOpenAI

try:
    from ..core.states import ProjectState, ImprovementMeasures, ServiceAnalysisOutput
    from ..core.config import OPENAI_API_KEY, LLM_MODEL_NAME
    from ..prompts.suggester_prompts import IMPROVEMENT_SUGGESTION_PROMPT_TEMPLATE
    # ReportTypeDeciderAgent의 _format_summary와 유사한 헬퍼 함수가 필요할 수 있음
    # 여기서는 간단히 ReportTypeDeciderAgent의 것을 참조하거나 직접 구현
    from .report_type_decider import ReportTypeDeciderAgent # _format_summary 사용 목적
except ImportError:
    from core.states import ProjectState, ImprovementMeasures, ServiceAnalysisOutput
    from core.config import OPENAI_API_KEY, LLM_MODEL_NAME
    from prompts.suggester_prompts import IMPROVEMENT_SUGGESTION_PROMPT_TEMPLATE
    from agents.report_type_decider import ReportTypeDeciderAgent


class ImprovementSuggesterAgent:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        # 개선안 제안은 창의성과 구체성이 필요하므로 temperature를 약간 높일 수 있음 (예: 0.5)
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.5, openai_api_key=OPENAI_API_KEY)
        self._summary_formatter = ReportTypeDeciderAgent() # _format_summary 메소드 사용 목적
        print("ImprovementSuggesterAgent initialized.")

    async def suggest_improvements(self, state: ProjectState) -> ProjectState:
        print("--- Improvement Suggester Agent: 개선 권고안 생성 시작 ---")

        # 이 에이전트는 issue_found == True일 때만 호출된다고 가정
        if not state.get("issue_found"):
            print("  'issue_found'가 False이므로 개선안 생성을 건너<0xEB><0><0x8A><0x8D>니다.")
            state["improvement_recommendations"] = None
            return state

        service_name = state.get("service_name", "N/A")
        service_type = state.get("service_type", "N/A")
        analyzed_info: Optional[ServiceAnalysisOutput] = state.get("analyzed_service_info")

        common_risks = state.get("common_ethics_risks")
        specific_risks = state.get("specific_ethics_risks_by_category")
        past_cases = state.get("past_case_analysis_results_by_category")

        if not analyzed_info:
            print("  Error: 서비스 분석 정보가 없어 개선안을 생성할 수 없습니다.")
            state["error_message"] = "개선안 생성: 서비스 분석 정보 누락"
            state["improvement_recommendations"] = None
            return state

        # 진단 결과 요약 문자열 생성 (ReportTypeDeciderAgent의 _format_summary 활용)
        common_summary = self._summary_formatter._format_summary(common_risks, "공통 윤리 리스크 진단 요약")
        specific_summary = self._summary_formatter._format_summary(specific_risks, "주요 리스크 카테고리별 심층 진단 요약")
        past_cases_summary = self._summary_formatter._format_summary(past_cases, "과거 AI 윤리 문제 사례 분석 및 시사점 요약")

        prompt = IMPROVEMENT_SUGGESTION_PROMPT_TEMPLATE.format(
            service_name=service_name,
            service_type=service_type,
            target_functions=analyzed_info.get("target_functions", "N/A"),
            key_features=analyzed_info.get("key_features", "N/A"),
            data_usage_estimation=analyzed_info.get("data_usage_estimation", "N/A"),
            common_risks_summary=common_summary,
            specific_risks_summary=specific_summary,
            past_cases_summary=past_cases_summary
        )

        try:
            print("  LLM 호출하여 개선 권고안 생성 중...")
            response = await self.llm.ainvoke(prompt)
            response_content = str(response.content)
            print(f"  LLM 응답 (개선안): {response_content[:300]}...")

            try:
                json_start_index = response_content.find('{')
                json_end_index = response_content.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_str = response_content[json_start_index:json_end_index]
                    recommendations_dict = json.loads(json_str)

                    # TypedDict로 변환
                    recommendations: ImprovementMeasures = {
                        "short_term": recommendations_dict.get("short_term", []),
                        "mid_long_term": recommendations_dict.get("mid_long_term", []),
                        "governance_suggestions": recommendations_dict.get("governance_suggestions", [])
                    }
                    state["improvement_recommendations"] = recommendations
                    print("  개선 권고안이 상태에 저장되었습니다.")
                else:
                    print(f"  LLM 응답(개선안)에서 유효한 JSON 객체를 찾을 수 없습니다. 응답: {response_content}")
                    state["error_message"] = "개선안 LLM 응답 JSON 파싱 실패 (형식)"
                    state["improvement_recommendations"] = None

            except json.JSONDecodeError as e:
                print(f"  LLM 응답(개선안) JSON 파싱 오류: {e}")
                state["error_message"] = f"개선안 LLM 응답 JSON 파싱 오류: {e}"
                state["improvement_recommendations"] = None

        except Exception as e:
            print(f"  LLM 호출(개선안) 중 오류 발생: {e}")
            state["error_message"] = f"개선안 LLM 호출 오류: {e}"
            state["improvement_recommendations"] = None

        print("--- Improvement Suggester Agent: 개선 권고안 생성 완료 ---")
        return state

# --- 테스트용 ---
async def run_standalone_test():
    import asyncio
    from dotenv import load_dotenv
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path): load_dotenv(dotenv_path=dotenv_path)

    try:
        from ..core.config import OPENAI_API_KEY as OAI_KEY_TEST # 테스트용으로 별도 변수
        from ..core.states import PastCaseAnalysis
    except ImportError:
        from core.config import OPENAI_API_KEY as OAI_KEY_TEST
        from core.states import PastCaseAnalysis

    if not OAI_KEY_TEST:
         print("Standalone Test Error: OPENAI_API_KEY is not set in config for test.")
         return

    # 예시 상태 (ReportTypeDeciderAgent가 'issue_found = True'로 판단한 후)
    state_requiring_improvement: ProjectState = {
        "service_name": "AI 친구 챗봇 '버디'",
        "service_type": "챗봇",
        "service_initial_info": "...",
        "analyzed_service_info": {"target_functions": "일상 대화, 감정 지원", "key_features": "인간과 유사한 대화", "data_usage_estimation": "모든 대화 내용 저장 및 분석", "estimated_users_stakeholders": "외로운 현대인, 심리 상담소"},
        "common_ethics_risks": {
            "E02_privacy": "모든 대화 내용 저장 시 개인정보보호법 위반 소지 및 과도한 정보 수집 우려 (E02.01, E02.02).",
            "E03_diversity": "특정 사용자 그룹(예: 노년층)에 대한 이해 부족으로 인한 편향적 응답 가능성 (E03.02)."
        },
        "specific_ethics_risks_by_category": {
            "privacy_chatbot": {"analysis": "대화 데이터 익명화 처리가 불분명하며, 제3자 공유 가능성에 대한 고지 부족.", "severity": "high"},
            "bias_chatbot": {"analysis": "특정 정치적/사회적 이슈에 대해 한쪽으로 치우친 답변을 생성하는 경향 발견.", "severity": "medium"}
        },
        "past_case_analysis_results_by_category": {
            "privacy_chatbot": [PastCaseAnalysis(case_name="이루다 개인정보 논란", key_insights="사전동의 및 비식별화 철저 필요", cause_and_structure="...", vulnerabilities="...", comparison_with_target_service="...")],
            "bias_chatbot": [PastCaseAnalysis(case_name="MS Tay 혐오발언 사건", key_insights="악의적 학습 데이터 필터링 및 지속적 모니터링 중요", cause_and_structure="...", vulnerabilities="...", comparison_with_target_service="...")]
        },
        "issue_found": True, # 개선안 필요!
        "report_type": "issue_improvement_report", # 이미 결정됨
        # 나머지 필드
        "main_risk_categories":None, "pending_specific_diagnoses":None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
    }

    suggester_agent = ImprovementSuggesterAgent()
    updated_state = await suggester_agent.suggest_improvements(state_requiring_improvement)

    print("\n--- 최종 상태 (Improvement Suggester 테스트 후) ---")
    if updated_state.get("error_message"):
        print(f"  에러: {updated_state['error_message']}")

    recommendations = updated_state.get("improvement_recommendations")
    if recommendations:
        print("  생성된 개선 권고안:")
        print(f"    단기적: {recommendations.get('short_term')}")
        print(f"    중장기적: {recommendations.get('mid_long_term')}")
        print(f"    거버넌스: {recommendations.get('governance_suggestions')}")
    else:
        print("  개선 권고안이 생성되지 않았습니다 (또는 issue_found가 False였을 수 있음).")

if __name__ == '__main__':
    import asyncio
    asyncio.run(run_standalone_test())