import json
import datetime
from typing import Dict, Optional, Any, List
from langchain_openai import ChatOpenAI

try:
    from ..core.states import ProjectState, ServiceAnalysisOutput, PastCaseAnalysis, ImprovementMeasures
    from ..core.config import OPENAI_API_KEY, LLM_MODEL_NAME
    from ..prompts.report_generator_prompts import ISSUE_IMPROVEMENT_REPORT_TEMPLATE, NO_ISSUE_REPORT_TEMPLATE
except ImportError:
    from core.states import ProjectState, ServiceAnalysisOutput, PastCaseAnalysis, ImprovementMeasures
    from core.config import OPENAI_API_KEY, LLM_MODEL_NAME
    from prompts.report_generator_prompts import ISSUE_IMPROVEMENT_REPORT_TEMPLATE, NO_ISSUE_REPORT_TEMPLATE

class ReportGeneratorAgent:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        # 보고서 생성은 지침을 잘 따라야 하므로 temperature를 너무 높이지 않음 (예: 0.3)
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.3, openai_api_key=OPENAI_API_KEY)
        print("ReportGeneratorAgent initialized.")

    def _format_dict_to_string(self, data: Optional[Dict[Any, Any]], indent=2) -> str:
        """딕셔너리를 포맷된 문자열로 변환 (LLM 컨텍스트용)"""
        if not data:
            return "제공된 정보 없음"
        try:
            # 간단한 JSON 문자열 변환 후, LLM이 이해하기 쉽게 줄바꿈 추가
            # 또는 각 키-값을 명확히 나열하는 방식
            formatted_str = ""
            for key, value in data.items():
                if isinstance(value, dict):
                    formatted_str += f"- {key}:\n"
                    for sub_key, sub_value in value.items():
                        # PastCaseAnalysis 같은 복잡한 객체나 리스트도 처리
                        if isinstance(sub_value, list) and sub_value and isinstance(sub_value[0], dict) and "case_name" in sub_value[0]: # PastCaseAnalysis 리스트
                            formatted_str += f"  - {sub_key}:\n"
                            for case_item in sub_value:
                                formatted_str += f"    - 사례명: {case_item.get('case_name')}\n      인사이트: {case_item.get('key_insights')}\n" # 필요한 정보만
                        elif isinstance(sub_value, dict) and "analysis" in sub_value : # specific_ethics_risks
                            formatted_str += f"    - {sub_key}: {sub_value.get('analysis')} (심각도: {sub_value.get('severity')})\n"
                        else:
                            formatted_str += f"    - {sub_key}: {str(sub_value)[:150]}...\n"
                elif isinstance(value, list):
                    formatted_str += f"- {key}: {', '.join(map(str,value))[:200]}...\n"
                else:
                    formatted_str += f"- {key}: {str(value)[:200]}...\n"
            return formatted_str if formatted_str else "내용 없음"
        except Exception:
            return str(data) # 실패 시 원본 데이터 문자열화


    async def generate_report(self, state: ProjectState) -> ProjectState:
        print("--- Report Generator Agent: 최종 보고서 생성 시작 ---")
        report_type = state.get("report_type")
        if not report_type:
            print("  Error: 보고서 유형이 결정되지 않았습니다. 보고서 생성을 건너뜁니다.")
            state["error_message"] = "보고서 생성: 보고서 유형 누락"
            state["final_report_markdown"] = None
            return state

        service_name = state.get("service_name", "N/A")
        service_type_str = state.get("service_type", "N/A") # service_type은 Literal이므로 str로 사용
        current_date_str = datetime.date.today().isoformat()
        
        analysis_agency = "SK AX - SKALA"
        
        analyzed_info: Optional[ServiceAnalysisOutput] = state.get("analyzed_service_info")
        common_risks = state.get("common_ethics_risks")
        specific_risks = state.get("specific_ethics_risks_by_category")
        past_cases_by_cat = state.get("past_case_analysis_results_by_category")
        recommendations: Optional[ImprovementMeasures] = state.get("improvement_recommendations")

        # LLM에게 제공할 컨텍스트 정보 포맷팅
        # 서비스 분석 결과
        sa_target_functions = analyzed_info.get("target_functions", "N/A") if analyzed_info else "N/A"
        sa_key_features = analyzed_info.get("key_features", "N/A") if analyzed_info else "N/A"
        sa_data_usage = analyzed_info.get("data_usage_estimation", "N/A") if analyzed_info else "N/A"
        sa_stakeholders = analyzed_info.get("estimated_users_stakeholders", "N/A") if analyzed_info else "N/A"

        # 리스크 평가 상세 (보고서 프롬프트 플레이스홀더용)
        risk_assessment_details = self._format_dict_to_string(common_risks) + "\n" + \
                                  self._format_dict_to_string(specific_risks)
        # 주요 취약점은 LLM이 보고서 생성 시 종합적으로 판단하여 작성하도록 유도하거나,
        # 이전 단계에서 별도 필드로 요약된 내용을 사용
        identified_vulnerabilities = "LLM이 종합적으로 기술할 부분 또는 이전 에이전트 결과" # 예시

        # 과거 사례 분석 상세 (보고서 프롬프트 플레이스홀더용)
        past_case_names_list = []
        past_cases_analysis_str = ""
        if past_cases_by_cat:
            for category, cases in past_cases_by_cat.items():
                past_cases_analysis_str += f"\n  카테고리 '{category}' 관련 과거 사례:\n"
                for case in cases:
                    past_case_names_list.append(case.get("case_name", "N/A"))
                    past_cases_analysis_str += f"    - 사례명: {case.get('case_name', 'N/A')}\n"
                    past_cases_analysis_str += f"      원인 및 구조: {case.get('cause_and_structure', 'N/A')}\n"
                    past_cases_analysis_str += f"      취약점: {case.get('vulnerabilities', 'N/A')}\n"
                    past_cases_analysis_str += f"      시사점: {case.get('key_insights', 'N/A')}\n"
        past_case_names_placeholder_str = ", ".join(list(set(past_case_names_list))) if past_case_names_list else "해당 없음"

        # 비교분석 및 주요 인사이트는 LLM이 종합적으로 생성하도록 플레이스홀더 유지
        comparison_insights_placeholder_str = "LLM이 종합적으로 기술할 부분"
        key_takeaways_placeholder_str = "LLM이 종합적으로 기술할 부분"

        # 전체 컨텍스트 (LLM에게 한 번에 전달)
        # 각 에이전트의 원본 결과들을 문자열로 조합
        context_for_llm = f"""
        [서비스 분석 정보]
        대상 기능: {sa_target_functions}
        주요 특징: {sa_key_features}
        데이터 수집/활용 (추정): {sa_data_usage}
        예상 사용자 및 이해관계자: {sa_stakeholders}

        [공통 윤리 리스크 진단 결과]
        {self._format_dict_to_string(common_risks)}

        [주요 리스크 카테고리별 심층 진단 결과]
        {self._format_dict_to_string(specific_risks)}

        [과거 AI 윤리 문제 사례 분석 결과]
        {past_cases_analysis_str}
        """
        if recommendations: # 개선안이 있는 경우에만 컨텍스트에 추가
            context_for_llm += f"""
            [제안된 개선 권고안]
            단기적: {recommendations.get('short_term')}
            중장기적: {recommendations.get('mid_long_term')}
            거버넌스: {recommendations.get('governance_suggestions')}
            """

        # 보고서 유형에 따라 프롬프트 선택 및 포맷팅
        if report_type == "issue_improvement_report":
            prompt_template = ISSUE_IMPROVEMENT_REPORT_TEMPLATE
            
            # 필수 키 확인
            required_keys = ['service_name', 'current_date', 'analysis_agency', 'service_type', 
                            'target_functions', 'key_features', 'data_usage_estimation', 
                            'estimated_users_stakeholders', 'risk_assessment_details_placeholder',
                            'identified_vulnerabilities_placeholder', 'past_case_names_placeholder',
                            'past_cases_analysis_details_placeholder', 'comparison_insights_placeholder',
                            'key_takeaways_placeholder', 'short_term_recommendations_placeholder',
                            'mid_long_term_recommendations_placeholder', 'governance_suggestions_placeholder',
                            'conclusion_placeholder', 'context']
            
            for key in required_keys:
                if key not in locals() and key not in state:
                    print(f"  Error: 필수 키 '{key}'가 누락되었습니다.")
                    state["error_message"] = f"보고서 생성: 필수 키 '{key}' 누락"
                    state["final_report_markdown"] = None
                    return state
            
            # 플레이스홀더 채우기
            prompt = prompt_template.format(
                service_name=service_name,
                current_date=current_date_str,
                analysis_agency=analysis_agency,
                summary_placeholder="[SUMMARY 요약은 LLM이 아래 내용을 바탕으로 직접 작성]",
                service_type=service_type_str,
                target_functions=sa_target_functions,
                key_features=sa_key_features,
                data_usage_estimation=sa_data_usage,
                estimated_users_stakeholders=sa_stakeholders,
                risk_assessment_details_placeholder=risk_assessment_details, # common + specific 요약
                identified_vulnerabilities_placeholder=identified_vulnerabilities,
                past_case_names_placeholder=past_case_names_placeholder_str,
                past_cases_analysis_details_placeholder=past_cases_analysis_str, # 이미 포맷된 문자열
                comparison_insights_placeholder=comparison_insights_placeholder_str,
                key_takeaways_placeholder=key_takeaways_placeholder_str,
                short_term_recommendations_placeholder=self._format_list_to_string(recommendations.get("short_term", []) if recommendations else []),
                mid_long_term_recommendations_placeholder=self._format_list_to_string(recommendations.get("mid_long_term", []) if recommendations else []),
                governance_suggestions_placeholder=self._format_list_to_string(recommendations.get("governance_suggestions", []) if recommendations else []),
                conclusion_placeholder="[결론은 LLM이 전체 내용을 바탕으로 직접 작성]",
                context=context_for_llm # LLM이 참고할 전체 정보
            )
        elif report_type == "no_issue_report":
            prompt_template = NO_ISSUE_REPORT_TEMPLATE
            prompt = prompt_template.format(
                service_name=service_name,
                current_date=current_date_str,
                analysis_agency=analysis_agency,
                summary_placeholder_no_issue="[SUMMARY 요약은 LLM이 아래 내용을 바탕으로 직접 작성]",
                service_type=service_type_str,
                target_functions=sa_target_functions,
                key_features=sa_key_features,
                data_usage_estimation=sa_data_usage,
                estimated_users_stakeholders=sa_stakeholders,
                risk_assessment_details_no_issue_placeholder=risk_assessment_details, # common + specific 요약 (문제 없다는 톤으로)
                past_case_names_placeholder=past_case_names_placeholder_str,
                past_cases_analysis_details_no_issue_placeholder=past_cases_analysis_str, # 참고용으로
                implications_for_current_service_placeholder="LLM이 종합적으로 기술할 부분",
                conclusion_no_issue_placeholder="[결론은 LLM이 전체 내용을 바탕으로 직접 작성]",
                context=context_for_llm
            )
        else:
            print(f"  Error: 알 수 없는 보고서 유형 '{report_type}'.")
            state["error_message"] = f"보고서 생성: 알 수 없는 보고서 유형 '{report_type}'"
            state["final_report_markdown"] = None
            return state

        try:
            print(f"  LLM 호출하여 '{report_type}' 보고서 생성 중...")
            # print(f"DEBUG: Prompt for report generation:\n{prompt[:1000]}...\n") # 프롬프트 디버깅용
            response = await self.llm.ainvoke(prompt)
            final_report_content = str(response.content)
            state["final_report_markdown"] = final_report_content
            print(f"  최종 보고서 생성 완료 (일부): {final_report_content[:300]}...")
        except Exception as e:
            print(f"  LLM 호출(보고서 생성) 중 오류 발생: {e}")
            state["error_message"] = f"보고서 생성 LLM 호출 오류: {e}"
            state["final_report_markdown"] = None

        print("--- Report Generator Agent: 최종 보고서 생성 완료 ---")
        return state

    def _format_list_to_string(self, data_list: Optional[List[str]]) -> str:
        if not data_list:
            return "- 해당 사항 없음"
        return "\n".join([f"- {item}" for item in data_list])


# --- 테스트용 ---
async def run_standalone_test():
    import asyncio
    from dotenv import load_dotenv
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path): load_dotenv(dotenv_path=dotenv_path)

    try:
        from ..core.config import OPENAI_API_KEY as OAI_KEY_TEST
    except ImportError:
        from core.config import OPENAI_API_KEY as OAI_KEY_TEST

    if not OAI_KEY_TEST:
        print("Standalone Test Error: OPENAI_API_KEY is not set in config for test.")
        return

    # 테스트 상태 1: 개선안 포함 보고서
    state_with_improvements: ProjectState = {
        "service_name": "AI 친구 챗봇 '버디'", "service_type": "챗봇", "service_initial_info": "...",
        "analyzed_service_info": {"target_functions": "일상 대화, 감정 지원", "key_features": "인간과 유사한 대화", "data_usage_estimation": "모든 대화 내용 저장 및 분석", "estimated_users_stakeholders": "외로운 현대인, 심리 상담소"},
        "common_ethics_risks": {"E02_privacy": "과도한 정보 수집 우려.", "E03_diversity": "편향 응답 가능성."},
        "specific_ethics_risks_by_category": {
            "privacy_chatbot": {"analysis": "익명화 불분명.", "severity": "high"},
            "bias_chatbot": {"analysis": "특정 주제 편향 발견.", "severity": "medium"}
        },
        "past_case_analysis_results_by_category": {
            "privacy_chatbot": [PastCaseAnalysis(case_name="이루다", key_insights="동의/비식별화 중요", cause_and_structure="...",vulnerabilities="...",comparison_with_target_service="...")],
        },
        "issue_found": True, 
        "report_type": "issue_improvement_report",
        "improvement_recommendations": {
            "short_term": ["데이터 수집 최소화 원칙 적용", "편향성 완화 위한 데이터셋 다양화"],
            "mid_long_term": ["차등 개인정보보호(DP) 기술 도입 연구", "AI 윤리 정기 교육 의무화"],
            "governance_suggestions": ["AI 윤리 담당자 지정 및 책임 명확화", "사용자 피드백 기반 지속적 개선 프로세스 구축"]
        },
        "main_risk_categories":None, "pending_specific_diagnoses":None, "final_report_markdown": None, "error_message": None
    }

    # 테스트 상태 2: 문제 없음 보고서
    state_no_issues: ProjectState = {
        "service_name": "기업용 문서 요약 AI '퀵썸'", "service_type": "기타", "service_initial_info": "...",
        "analyzed_service_info": {"target_functions": "문서 요약", "key_features": "높은 정확도", "data_usage_estimation": "업로드 문서 임시 처리 후 즉시 파기", "estimated_users_stakeholders": "기업 직원"},
        "common_ethics_risks": {"E07_data_management": "파기 정책 명확, 이행 감사 필요."},
        "specific_ethics_risks_by_category": {"general_risk": {"analysis": "전반적 양호, 로그 기록 강화 권장.", "severity": "low"}},
        "past_case_analysis_results_by_category": {},
        "issue_found": False,
        "report_type": "no_issue_report",
        "improvement_recommendations": None, # 문제 없으므로 개선안 없음
        "main_risk_categories":None, "pending_specific_diagnoses":None, "final_report_markdown": None, "error_message": None
    }

    report_agent = ReportGeneratorAgent()

    print("\n--- Test 1: Report with Improvements ---")
    updated_state_1 = await report_agent.generate_report(state_with_improvements)
    if updated_state_1.get("error_message"):
        print(f"  Error: {updated_state_1['error_message']}")
    else:
        print("  Report (with improvements) generated successfully (first 500 chars):")
        print(updated_state_1.get("final_report_markdown", "")[:500] + "...")

    print("\n--- Test 2: Report with No Issues ---")
    updated_state_2 = await report_agent.generate_report(state_no_issues)
    if updated_state_2.get("error_message"):
        print(f"  Error: {updated_state_2['error_message']}")
    else:
        print("  Report (no issues) generated successfully (first 500 chars):")
        print(updated_state_2.get("final_report_markdown", "")[:500] + "...")


if __name__ == '__main__':
    import asyncio
    asyncio.run(run_standalone_test())