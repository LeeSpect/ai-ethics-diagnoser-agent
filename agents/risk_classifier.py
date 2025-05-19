from typing import List, Optional, Dict, Any

try:
    from ..core.states import ProjectState # poetry run ... 또는 패키지 내부에서 실행 시
except ImportError: # python -m agents.risk_classifier 등으로 직접 실행 또는 테스트 시
    from core.states import ProjectState


class RiskClassifierAgent:
    def __init__(self):
        print("RiskClassifierAgent initialized.")

    async def classify_risks(self, state: ProjectState) -> ProjectState:
        print("--- Risk Classifier Agent: 주요 리스크 카테고리 분류 시작 ---")
        service_type = state.get("service_type")
        analyzed_info = state.get("analyzed_service_info") # 참고용으로 받을 수 있으나, 현재 로직은 service_type만 사용

        main_risk_categories: List[str] = []

        if not service_type:
            print("  Warning: Service type not found in state. Defaulting to general risk.")
            main_risk_categories = ["general_risk"]
        elif service_type == "챗봇":
            # 주요 우려 리스크
            main_risk_categories = ["bias_chatbot", "privacy_chatbot"]
            print("  분류된 주요 리스크 (챗봇): 편향된 발언, 개인정보 유출")
        elif service_type == "추천 알고리즘":
            main_risk_categories = ["filter_bubble_recommendation", "transparency_recommendation"]
            print("  분류된 주요 리스크 (추천 알고리즘): 필터버블, 추천 기준 불투명성")
        elif service_type == "이미지 생성 AI":
            main_risk_categories = ["copyright_generation", "deepfake_generation"]
            print("  분류된 주요 리스크 (이미지 생성 AI): 저작권 침해, 딥페이크 악용")
        else: # "기타" 유형 또는 정의되지 않은 유형
            main_risk_categories = ["general_risk"]
            print(f"  분류된 주요 리스크 ({service_type}): 일반 리스크 진단")

        # main_risk_categories가 비어있다면 (위 로직상 발생하지 않지만 방어적으로)
        if not main_risk_categories:
            main_risk_categories = ["general_risk"]
            print("  Warning: No specific risk categories identified. Defaulting to general_risk.")


        state["main_risk_categories"] = main_risk_categories
        # pending_specific_diagnoses는 main_risk_categories의 복사본으로 시작
        # 특화 진단 라우터가 이 큐를 보고 하나씩 처리
        state["pending_specific_diagnoses"] = list(main_risk_categories) # 새로운 리스트로 복사

        print(f"  상태 저장: main_risk_categories = {state['main_risk_categories']}")
        print(f"  상태 저장: pending_specific_diagnoses = {state['pending_specific_diagnoses']}")
        print("--- Risk Classifier Agent: 주요 리스크 카테고리 분류 완료 ---")
        return state

    # --- 에이전트 테스트용 (직접 실행 시) ---
    @staticmethod
    async def run_standalone_test():
        import asyncio

        # 테스트용 초기 상태 (ServiceAnalyzerAgent 실행 후 상태 가정)
        initial_state_chatbot: ProjectState = {
            "service_name": "마음상담 챗봇 '마음친구'",
            "service_type": "챗봇",
            "service_initial_info": "...",
            "analyzed_service_info": { # ServiceAnalyzerAgent가 생성한 결과 (예시)
                "target_functions": "고민 상담, 위로 제공",
                "key_features": "익명성, 공감 대화",
                "data_usage_estimation": "대화 내용 저장 및 분석 (익명화 처리)",
                "estimated_users_stakeholders": "일상적 스트레스를 겪는 일반인"
            },
            "main_risk_categories": None, "pending_specific_diagnoses": None,
            "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
            "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
            "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
        }

        initial_state_recsys: ProjectState = {
            "service_name": "맞춤형 뉴스 추천봇",
            "service_type": "추천 알고리즘",
            "service_initial_info": "...",
            "analyzed_service_info": {"target_functions": "뉴스 기사 추천", "key_features": "개인 관심사 기반", "data_usage_estimation": "열람 기록, 선호도", "estimated_users_stakeholders": "뉴스 소비자, 언론사"},
            "main_risk_categories": None, "pending_specific_diagnoses": None,
            "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
            "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
            "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
        }

        initial_state_other: ProjectState = {
            "service_name": "AI 날씨 예측봇",
            "service_type": "기타", # 또는 정의되지 않은 유형
            "service_initial_info": "...",
            "analyzed_service_info": {"target_functions": "날씨 정보 제공", "key_features": "정확도 높은 예측", "data_usage_estimation": "기상 데이터, 위치 정보", "estimated_users_stakeholders": "일반 사용자, 농업 종사자"},
            "main_risk_categories": None, "pending_specific_diagnoses": None,
            "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
            "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
            "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
        }


        classifier_agent = RiskClassifierAgent()

        print("\n--- Chatbot Risk Classification Test ---")
        updated_state_chatbot = await classifier_agent.classify_risks(initial_state_chatbot)
        print(f"  Chatbot - Main Risks: {updated_state_chatbot.get('main_risk_categories')}")
        print(f"  Chatbot - Pending Diagnoses: {updated_state_chatbot.get('pending_specific_diagnoses')}")

        print("\n--- Recommendation System Risk Classification Test ---")
        updated_state_recsys = await classifier_agent.classify_risks(initial_state_recsys)
        print(f"  RecSys - Main Risks: {updated_state_recsys.get('main_risk_categories')}")
        print(f"  RecSys - Pending Diagnoses: {updated_state_recsys.get('pending_specific_diagnoses')}")

        print("\n--- Other Type Risk Classification Test ---")
        updated_state_other = await classifier_agent.classify_risks(initial_state_other)
        print(f"  Other - Main Risks: {updated_state_other.get('main_risk_categories')}")
        print(f"  Other - Pending Diagnoses: {updated_state_other.get('pending_specific_diagnoses')}")

# --- 에이전트 테스트용 (직접 실행 시) ---
# @staticmethod # 이 줄은 클래스 내부로 이동했습니다.
# async def run_standalone_test(): # 이 함수는 클래스 내부로 이동했습니다.
# // ... existing code ...

if __name__ == '__main__':
    import asyncio
    # run_standalone_test를 staticmethod로 만들었으므로 클래스명으로 직접 호출
    asyncio.run(RiskClassifierAgent.run_standalone_test())