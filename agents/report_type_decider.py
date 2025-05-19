# agents/report_type_decider.py
import json
from typing import Dict, Optional, Any
from langchain_openai import ChatOpenAI

try:
    from ..core.states import ProjectState, PastCaseAnalysis
    from ..core.config import OPENAI_API_KEY, LLM_MODEL_NAME
    from ..prompts.decider_prompts import ISSUE_SEVERITY_ASSESSMENT_PROMPT_TEMPLATE
except ImportError:
    from core.states import ProjectState, PastCaseAnalysis
    from core.config import OPENAI_API_KEY, LLM_MODEL_NAME
    from prompts.decider_prompts import ISSUE_SEVERITY_ASSESSMENT_PROMPT_TEMPLATE

class ReportTypeDeciderAgent:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        # 판단의 일관성을 위해 temperature를 낮게 설정하거나,
        # 더 정교한 판단 로직을 위해 별도의 few-shot 예시를 제공하는 것도 고려 가능
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.1, openai_api_key=OPENAI_API_KEY)
        print("ReportTypeDeciderAgent initialized.")

    def _format_summary(self, data: Optional[Dict[str, Any]], title: str) -> str:
        """진단 결과 요약 문자열 생성 헬퍼 함수"""
        if not data:
            return f"{title}: 제공된 정보 없음\n"

        summary_str = f"{title}:\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and "analysis" in value: # 특화 진단 결과 형식 가정
                    summary_str += f"  - {key}: {value.get('analysis', 'N/A')} (심각도 추정: {value.get('severity', 'N/A')})\n"
                elif isinstance(value, list): # 과거 사례 분석 결과 형식 가정
                    summary_str += f"  - {key}:\n"
                    for item in value:
                        if isinstance(item, dict) and "case_name" in item:
                            summary_str += f"    - 사례명: {item.get('case_name')}: {item.get('key_insights', 'N/A')}\n"
                        else:
                            summary_str += f"    - {str(item)[:100]}...\n" # 일반 리스트 항목
                else:
                    summary_str += f"  - {key}: {str(value)[:150]}...\n" # 일반 딕셔너리 항목
        else:
            summary_str += f"  {str(data)[:300]}...\n"
        return summary_str + "\n"

    async def decide_report_type(self, state: ProjectState) -> ProjectState:
        print("--- Report Type Decider Agent: 보고서 유형 결정 시작 ---")
        service_name = state.get("service_name", "N/A")
        service_type = state.get("service_type", "N/A")

        common_risks = state.get("common_ethics_risks")
        specific_risks = state.get("specific_ethics_risks_by_category")
        past_cases = state.get("past_case_analysis_results_by_category")

        common_summary = self._format_summary(common_risks, "공통 윤리 리스크 진단 요약")
        specific_summary = self._format_summary(specific_risks, "주요 리스크 카테고리별 심층 진단 요약")
        past_cases_summary = self._format_summary(past_cases, "과거 AI 윤리 문제 사례 분석 및 시사점 요약")

        # 모든 진단 정보가 없는 경우, 기본적으로 '문제 없음' 및 '개선안 불필요'로 처리할 수 있음
        if not common_risks and not specific_risks:
            print("  Warning: 유효한 진단 결과가 없어 '문제 없음'으로 간주합니다.")
            state["issue_found"] = False
            state["report_type"] = "no_issue_report"
            print(f"  판단 결과: issue_found = False, report_type = no_issue_report (진단 결과 부족)")
            print("--- Report Type Decider Agent: 보고서 유형 결정 완료 ---")
            return state

        prompt = ISSUE_SEVERITY_ASSESSMENT_PROMPT_TEMPLATE.format(
            service_name=service_name,
            service_type=service_type,
            common_risks_summary=common_summary,
            specific_risks_summary=specific_summary,
            past_cases_summary=past_cases_summary
        )

        try:
            print("  LLM 호출하여 '심각한 문제' 여부 판단 중...")
            response = await self.llm.ainvoke(prompt)
            response_content = str(response.content)
            print(f"  LLM 응답 (심각도 판단): {response_content}")

            try:
                json_start_index = response_content.find('{')
                json_end_index = response_content.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_str = response_content[json_start_index:json_end_index]
                    decision_dict = json.loads(json_str)

                    issue_found_by_llm = decision_dict.get("issue_found", False) # 기본값 False
                    reasoning = decision_dict.get("reasoning", "LLM으로부터 판단 근거를 받지 못했습니다.")

                    state["issue_found"] = bool(issue_found_by_llm) # 명시적으로 bool 타입 변환

                    if state["issue_found"]:
                        state["report_type"] = "issue_improvement_report"
                    else:
                        state["report_type"] = "no_issue_report"

                    print(f"  LLM 판단 결과: issue_found = {state['issue_found']}, 근거: {reasoning}")
                    print(f"  결정된 보고서 유형: {state['report_type']}")

                else:
                    print(f"  LLM 응답(심각도 판단)에서 유효한 JSON 객체를 찾을 수 없습니다. 응답: {response_content}")
                    state["error_message"] = "심각도 판단 LLM 응답 JSON 파싱 실패 (형식)"
                    # 안전하게 '문제 있음'으로 간주하여 개선안 검토 유도 또는 기본값 설정
                    state["issue_found"] = True 
                    state["report_type"] = "issue_improvement_report"
                    print("  안전 조치: issue_found = True, report_type = issue_improvement_report (파싱 실패)")


            except json.JSONDecodeError as e:
                print(f"  LLM 응답(심각도 판단) JSON 파싱 오류: {e}")
                state["error_message"] = f"심각도 판단 LLM 응답 JSON 파싱 오류: {e}"
                state["issue_found"] = True # 파싱 실패 시 안전하게 '문제 있음'으로
                state["report_type"] = "issue_improvement_report"
                print("  안전 조치: issue_found = True, report_type = issue_improvement_report (파싱 오류)")

        except Exception as e:
            print(f"  LLM 호출(심각도 판단) 중 오류 발생: {e}")
            state["error_message"] = f"심각도 판단 LLM 호출 오류: {e}"
            state["issue_found"] = True # 호출 실패 시 안전하게 '문제 있음'으로
            state["report_type"] = "issue_improvement_report"
            print("  안전 조치: issue_found = True, report_type = issue_improvement_report (호출 오류)")

        print("--- Report Type Decider Agent: 보고서 유형 결정 완료 ---")
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
    except ImportError:
        from core.config import OPENAI_API_KEY as OAI_KEY_TEST

    if not OAI_KEY_TEST:
         print("Standalone Test Error: OPENAI_API_KEY is not set in config for test.")
         return

    # 예시 상태 (진단이 완료된 후)
    state_with_issues: ProjectState = {
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
        # 나머지 필드
        "main_risk_categories":None, "pending_specific_diagnoses":None, "issue_found": None, "report_type": None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
    }

    state_no_major_issues: ProjectState = {
        "service_name": "기업용 문서 요약 AI '퀵썸'",
        "service_type": "기타", # 예시
        "service_initial_info": "...",
        "analyzed_service_info": {"target_functions": "긴 문서를 핵심 내용만 요약", "key_features": "높은 정확도, 빠른 처리 속도", "data_usage_estimation": "사용자 업로드 문서 (임시 처리 후 즉시 파기 명시)", "estimated_users_stakeholders": "기업 직원, 연구원"},
        "common_ethics_risks": {
            "E07_data_management": "문서 임시 처리 및 즉시 파기 정책이 명확하나, 실제 이행 여부 감사 필요 (E07.03).",
            "E10_transparency": "요약 알고리즘의 상세 로직은 비공개이나, 결과의 편향 가능성에 대한 일반적 고지는 충분함 (E10.01)."
        },
        "specific_ethics_risks_by_category": { # 특화 진단에서 큰 문제가 없었다고 가정
            "general_risk": {"analysis": "전반적으로 윤리적 운영 수준이 양호하나, 데이터 처리 로그 기록 강화 권장.", "severity": "low"}
        },
        "past_case_analysis_results_by_category": {}, # 큰 관련 사례 없었다고 가정
        "main_risk_categories":None, "pending_specific_diagnoses":None, "issue_found": None, "report_type": None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
    }

    decider_agent = ReportTypeDeciderAgent()

    print("\n--- Test with Potential Issues ---")
    updated_state_issues = await decider_agent.decide_report_type(state_with_issues)
    print(f"  Decision for '버디': issue_found = {updated_state_issues.get('issue_found')}, report_type = {updated_state_issues.get('report_type')}")

    print("\n--- Test with No Major Issues ---")
    updated_state_no_issues = await decider_agent.decide_report_type(state_no_major_issues)
    print(f"  Decision for '퀵썸': issue_found = {updated_state_no_issues.get('issue_found')}, report_type = {updated_state_no_issues.get('report_type')}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(run_standalone_test())