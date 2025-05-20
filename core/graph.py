from langgraph.graph import StateGraph, END
from typing import Literal

from core.states import ProjectState
from agents.service_analyzer import ServiceAnalyzerAgent
from agents.risk_classifier import RiskClassifierAgent
from agents.common_ethics_diagnoser import CommonEthicsDiagnoserAgent
from agents.specific_risk_diagnosis_router import SpecificRiskDiagnosisRouterAgent
from agents.ethics_risk_diagnoser_chatbot import EthicsRiskDiagnoserChatbotAgent
from agents.ethics_risk_diagnoser_recsys import EthicsRiskDiagnoserRecSysAgent
from agents.ethics_risk_diagnoser_imagegen import EthicsRiskDiagnoserImageGenAgent
from agents.ethics_risk_diagnoser_general import EthicsRiskDiagnoserGeneralAgent
from agents.report_type_decider import ReportTypeDeciderAgent
from agents.improvement_suggester import ImprovementSuggesterAgent
from agents.report_generator import ReportGeneratorAgent


# --- 에이전트 인스턴스 생성 ---
service_analyzer = ServiceAnalyzerAgent(use_web_search=True) # 웹 검색 사용 여부 설정
risk_classifier = RiskClassifierAgent()
common_diagnoser = CommonEthicsDiagnoserAgent()
specific_risk_router = SpecificRiskDiagnosisRouterAgent() # 이 에이전트의 메소드는 조건부 엣지 함수로 사용됨
diagnoser_chatbot = EthicsRiskDiagnoserChatbotAgent(use_web_search=True)
diagnoser_recsys = EthicsRiskDiagnoserRecSysAgent(use_web_search=True) # 각 특화 진단기도 웹 검색 사용 가능
diagnoser_imagegen = EthicsRiskDiagnoserImageGenAgent(use_web_search=True)
diagnoser_general = EthicsRiskDiagnoserGeneralAgent()
report_type_decider = ReportTypeDeciderAgent()
improvement_suggester = ImprovementSuggesterAgent()
report_generator = ReportGeneratorAgent()

# --- LangGraph 워크플로우 정의 ---
workflow = StateGraph(ProjectState)

# 1. 노드 정의 (각 에이전트의 핵심 비동기 메소드를 노드로 등록)
workflow.add_node("service_analyzer_node", service_analyzer.analyze_service)
workflow.add_node("risk_classifier_node", risk_classifier.classify_risks)
workflow.add_node("common_ethics_diagnoser_node", common_diagnoser.diagnose_common_risks)

# 특화 진단 노드들
workflow.add_node("diagnoser_chatbot_node", diagnoser_chatbot.diagnose)
workflow.add_node("diagnoser_recsys_node", diagnoser_recsys.diagnose)
workflow.add_node("diagnoser_imagegen_node", diagnoser_imagegen.diagnose)
workflow.add_node("diagnoser_general_node", diagnoser_general.diagnose)

workflow.add_node("report_type_decider_node", report_type_decider.decide_report_type)
workflow.add_node("improvement_suggester_node", improvement_suggester.suggest_improvements)
workflow.add_node("report_generator_node", report_generator.generate_report)

# 2. 시작점 설정
workflow.set_entry_point("service_analyzer_node")

# 3. 엣지 연결
workflow.add_edge("service_analyzer_node", "risk_classifier_node")
workflow.add_edge("risk_classifier_node", "common_ethics_diagnoser_node")

# CommonEthicsDiagnoser 이후 SpecificRiskDiagnosisRouter로 라우팅 결정
# 라우터 자체는 노드가 아니라, 조건부 엣지의 함수로 사용됨
workflow.add_conditional_edges(
    "common_ethics_diagnoser_node", # 이 노드 실행 후 라우팅 결정
    specific_risk_router.get_next_diagnosis_node, # 상태를 받아 다음 노드 이름을 반환하는 함수
    {
        "diagnoser_chatbot_node": "diagnoser_chatbot_node",
        "diagnoser_recsys_node": "diagnoser_recsys_node",
        "diagnoser_imagegen_node": "diagnoser_imagegen_node",
        "diagnoser_general_node": "diagnoser_general_node",
        "report_type_decider_node": "report_type_decider_node" # 모든 특화 진단 완료 시
    }
)

# 각 특화 진단 노드 실행 후 다시 라우터로 돌아가서 다음 특화 진단 처리 또는 완료 결정
workflow.add_edge("diagnoser_chatbot_node", "common_ethics_diagnoser_node") # 주의: Common을 다시 거치지 않고 라우터로 가야 함
workflow.add_edge("diagnoser_recsys_node", "common_ethics_diagnoser_node")   # 수정 필요
workflow.add_edge("diagnoser_imagegen_node", "common_ethics_diagnoser_node") # 수정 필요
workflow.add_edge("diagnoser_general_node", "common_ethics_diagnoser_node")  # 수정 필요

# --- 특화 진단 후 라우터로 복귀 ---
# 즉, common_ethics_diagnoser_node가 일종의 라우팅 허브 역할을 합니다.

# ReportTypeDecider 이후 조건부 분기 (개선안 제안 여부)
def should_suggest_improvements(state: ProjectState) -> Literal["improvement_suggester_node", "report_generator_node"]:
    print("--- Deciding whether to suggest improvements ---")
    if state.get("issue_found") is True and state.get("report_type") == "issue_improvement_report":
        print("  Issue found, routing to Improvement Suggester.")
        return "improvement_suggester_node"
    else:
        print("  No issue found or report type is no_issue, routing to Report Generator.")
        return "report_generator_node"

workflow.add_conditional_edges(
    "report_type_decider_node",
    should_suggest_improvements,
    {
        "improvement_suggester_node": "improvement_suggester_node",
        "report_generator_node": "report_generator_node"
    }
)

# ImprovementSuggester (호출된 경우) 이후 ReportGenerator로
workflow.add_edge("improvement_suggester_node", "report_generator_node")

# ReportGenerator 이후 종료
workflow.add_edge("report_generator_node", END)

# 4. 그래프 컴파일
app = workflow.compile()
print("LangGraph workflow compiled successfully.")

# --- 전체 워크플로우 테스트를 위한 함수 ---
async def run_full_workflow_test(initial_state: ProjectState):
    print(f"\n--- 전체 워크플로우 테스트 시작 (서비스: {initial_state.get('service_name')}) ---")
    async for event in app.astream(initial_state):
        # 이벤트 스트리밍 (각 노드 실행 전후 등)
        # print(f"\nEvent: {event.event}")
        # print(f"  Node: {event.name}")
        # if event.data and event.data.get(event.name):
        #      print(f"  Output: {event.data.get(event.name)}")
        # 여기서는 최종 결과만 확인
        pass

    # 최종 상태 가져오기 (invoke를 사용하면 마지막 상태만 반환)
    final_state = await app.ainvoke(initial_state)

    print(f"\n--- 전체 워크플로우 테스트 완료 (서비스: {initial_state.get('service_name')}) ---")
    if final_state.get("error_message"):
        print(f"최종 에러: {final_state.get('error_message')}")

    print(f"보고서 유형: {final_state.get('report_type')}")
    if final_state.get("issue_found"):
        print("심각한 문제 발견됨. 개선안이 제안되었습니다.")
        # print(f"개선안: {final_state.get('improvement_recommendations')}")
    else:
        print("심각한 문제 발견되지 않음.")

    print("\n최종 생성된 보고서 (일부):")
    report_content = final_state.get("final_report_markdown")
    if report_content is None:
        print("보고서 생성 실패 (내용 없음)...")
    else:
        print(str(report_content)[:1000] + "...")
    return final_state


# --- 테스트 코드 ---
if __name__ == '__main__':
    import asyncio
    from dotenv import load_dotenv
    import os
    import datetime

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded .env from {dotenv_path} for graph test")
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

    # db_utils 초기화
    # 실제 애플리케이션에서는 main 시작 시 한 번 호출
    try:
        from .db_utils import initialize_vector_databases
    except ImportError:
        from core.db_utils import initialize_vector_databases

    print("Initializing databases for graph test...")
    initialize_vector_databases() # DB 준비


    # 테스트할 초기 상태 정의
    test_state_chatbot_issues: ProjectState = {
        "service_name": "AI 상담 챗봇 '친구야'",
        "service_type": "챗봇",
        "service_initial_info": "사용자의 고민을 들어주고 위로와 조언을 해주는 챗봇입니다. 익명으로 사용할 수 있으며, 24시간 응답합니다. 사용자의 감정을 분석하여 맞춤형 대화를 제공하려고 합니다.",
        "analyzed_service_info": None, "main_risk_categories": None, "pending_specific_diagnoses": None,
        "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None,
        "current_date": datetime.date.today().isoformat()
    }

    test_state_recsys_no_issues: ProjectState = {
        "service_name": "학술 논문 추천 시스템 '페이퍼 파인더'",
        "service_type": "추천 알고리즘",
        "service_initial_info": "사용자의 연구 분야 및 관심 키워드를 기반으로 최신 학술 논문을 추천해주는 시스템입니다. 투명한 추천 근거를 제시하려고 노력합니다.",
        "analyzed_service_info": None, "main_risk_categories": None, "pending_specific_diagnoses": None,
        "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None,
        "current_date": datetime.date.today().isoformat()
    }

    async def main_test_runner():
        await run_full_workflow_test(test_state_chatbot_issues)
        # await run_full_workflow_test(test_state_recsys_no_issues) # 다른 케이스도 테스트

    asyncio.run(main_test_runner())

    # 다이어그램 생성 (선택적, playwright 또는 pygraphviz 필요)
    # try:
    #     image_bytes = app.get_graph().draw_mermaid_png()
    #     with open("project_workflow_diagram.png", "wb") as f:
    #         f.write(image_bytes)
    #     print("\nWorkflow diagram saved to project_workflow_diagram.png")
    # except Exception as e:
    #     print(f"\nError generating workflow diagram: {e}")
    #     print("Ensure playwright is installed (pip install playwright && playwright install chromium)")
    #     print("Alternatively, LangGraph's ASCII diagram can be printed:")
    #     app.get_graph().print_ascii()