from typing import Literal, List, Optional

try:
    from ..core.states import ProjectState
except ImportError:
    from core.states import ProjectState


class SpecificRiskDiagnosisRouterAgent:
    def __init__(self):
        print("SpecificRiskDiagnosisRouterAgent initialized.")

    async def get_next_diagnosis_node(self, state: ProjectState) -> Literal[
        "diagnoser_chatbot_node",
        "diagnoser_recsys_node",
        "diagnoser_imagegen_node",
        "diagnoser_general_node",
        "report_type_decider_node" # 모든 특화 진단 완료 시 다음 노드
    ]:
        print("--- Specific Risk Diagnosis Router Agent: 다음 특화 진단 노드 결정 ---")
        pending_diagnoses: Optional[List[str]] = state.get("pending_specific_diagnoses")

        if pending_diagnoses and len(pending_diagnoses) > 0:
            next_diagnosis_category = pending_diagnoses[0] # 큐의 첫 번째 항목 확인 (제거는 각 진단 노드에서)
            print(f"  다음 처리할 특화 진단 카테고리: {next_diagnosis_category}")
            if next_diagnosis_category == "bias_chatbot" or next_diagnosis_category == "privacy_chatbot":
                # "chatbot" 키워드가 포함되면 diagnoser_chatbot_node로 보냄
                # 실제로는 main_risk_categories에 "chatbot_diagnosis" 같은 통합 카테고리명을 넣는 것이 더 명확할 수 있음
                # 현재는 main_risk_categories에 "bias_chatbot", "privacy_chatbot" 등이 직접 들어감.
                # 이들을 묶어서 하나의 챗봇 진단 노드로 보내려면, 그 노드에서 두 리스크를 모두 다뤄야 함.
                # 또는, 이 라우터에서 "bias_chatbot"과 "privacy_chatbot"을 "diagnoser_chatbot_node"로 매핑.
                if "chatbot" in next_diagnosis_category:
                    print("  라우팅: diagnoser_chatbot_node")
                    return "diagnoser_chatbot_node"
            elif "recommendation" in next_diagnosis_category:
                print("  라우팅: diagnoser_recsys_node")
                return "diagnoser_recsys_node"
            elif "generation" in next_diagnosis_category:
                print("  라우팅: diagnoser_imagegen_node")
                return "diagnoser_imagegen_node"
            elif next_diagnosis_category == "general_risk":
                print("  라우팅: diagnoser_general_node")
                return "diagnoser_general_node"
            else:
                # 정의되지 않은 카테고리 처리
                print(f"  Warning: 알 수 없는 특화 진단 카테고리 '{next_diagnosis_category}'. diagnoser_general_node로 라우팅합니다.")
                return "diagnoser_general_node"
        else:
            print("  모든 특화 진단 완료. Report Type Decider로 라우팅합니다.")
            return "report_type_decider_node"

# --- 테스트용 ---
@staticmethod
async def run_standalone_test():
    import asyncio

    # 테스트용 상태
    state_chatbot_pending_bias: ProjectState = {
        "service_name": "Test Chatbot", "service_type": "챗봇", "service_initial_info": "...",
        "pending_specific_diagnoses": ["bias_chatbot", "privacy_chatbot"],
        # ... 나머지 필드는 테스트에 불필요하므로 None 또는 기본값으로 설정
        "analyzed_service_info": None, "main_risk_categories": ["bias_chatbot", "privacy_chatbot"],
        "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
    }
    state_recsys_pending_filter: ProjectState = {
        "service_name": "Test RecSys", "service_type": "추천 알고리즘", "service_initial_info": "...",
        "pending_specific_diagnoses": ["filter_bubble_recommendation", "transparency_recommendation"],
        "main_risk_categories": ["filter_bubble_recommendation", "transparency_recommendation"],
        "analyzed_service_info": None, "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
    }
    state_img_gen_pending_copyright: ProjectState = {
        "service_name": "Test ImgGen", "service_type": "이미지 생성 AI", "service_initial_info": "...",
        "pending_specific_diagnoses": ["copyright_generation", "deepfake_generation"],
        "main_risk_categories": ["copyright_generation", "deepfake_generation"],
        "analyzed_service_info": None, "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
    }
    state_general_pending: ProjectState = {
        "service_name": "Test Other", "service_type": "기타", "service_initial_info": "...",
        "pending_specific_diagnoses": ["general_risk"],
        "main_risk_categories": ["general_risk"],
        "analyzed_service_info": None, "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
    }
    state_all_done: ProjectState = {
        "service_name": "Test Done", "service_type": "챗봇", "service_initial_info": "...",
        "pending_specific_diagnoses": [], # 큐가 비어 있음
        "main_risk_categories": ["bias_chatbot"], # 이미 처리했다고 가정
        "analyzed_service_info": None, "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
        "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
    }

    router_agent = SpecificRiskDiagnosisRouterAgent()

    print(f"\nTest 1 (Chatbot - bias): Expected diagnoser_chatbot_node, Got: {await router_agent.get_next_diagnosis_node(state_chatbot_pending_bias)}")
    # 큐에서 bias_chatbot을 제거했다고 가정하고 다음 테스트
    state_chatbot_pending_bias["pending_specific_diagnoses"] = ["privacy_chatbot"]
    print(f"Test 2 (Chatbot - privacy): Expected diagnoser_chatbot_node, Got: {await router_agent.get_next_diagnosis_node(state_chatbot_pending_bias)}")


    print(f"\nTest 3 (RecSys - filter_bubble): Expected diagnoser_recsys_node, Got: {await router_agent.get_next_diagnosis_node(state_recsys_pending_filter)}")
    state_recsys_pending_filter["pending_specific_diagnoses"] = ["transparency_recommendation"]
    print(f"Test 4 (RecSys - transparency): Expected diagnoser_recsys_node, Got: {await router_agent.get_next_diagnosis_node(state_recsys_pending_filter)}")

    print(f"\nTest 5 (ImgGen - copyright): Expected diagnoser_imagegen_node, Got: {await router_agent.get_next_diagnosis_node(state_img_gen_pending_copyright)}")

    print(f"\nTest 6 (General): Expected diagnoser_general_node, Got: {await router_agent.get_next_diagnosis_node(state_general_pending)}")
    print(f"\nTest 7 (All Done): Expected report_type_decider_node, Got: {await router_agent.get_next_diagnosis_node(state_all_done)}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(SpecificRiskDiagnosisRouterAgent.run_standalone_test())