from typing import List, Optional, Dict, Any

try:
    from ..core.states import ProjectState, PastCaseAnalysis # PastCaseAnalysis는 일반 진단에서 덜 사용될 수 있지만, 구조적 일관성을 위해 포함
    from ..core.db_utils import search_documents, checklist_collection # 일반 점검표 항목 검색에 사용될 수 있음
    from ..core.config import LLM_MODEL_NAME, OPENAI_API_KEY
except ImportError: # 직접 실행 또는 테스트 시
    from core.states import ProjectState, PastCaseAnalysis
    from core.db_utils import search_documents, checklist_collection
    from core.config import LLM_MODEL_NAME, OPENAI_API_KEY

from langchain_openai import ChatOpenAI

class EthicsRiskDiagnoserGeneralAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.2, openai_api_key=OPENAI_API_KEY)
        print("EthicsRiskDiagnoserGeneralAgent initialized.")

    async def diagnose(self, state: ProjectState) -> ProjectState:
        print("--- Ethics Risk Diagnoser Agent (General): 진단 시작 ---")
        pending_diagnoses: Optional[List[str]] = state.get("pending_specific_diagnoses")
        analyzed_info = state.get("analyzed_service_info") # 서비스 분석 정보
        service_name = state.get("service_name", "알 수 없는 서비스")
        service_initial_info = state.get("service_initial_info", "제공된 설명 없음")

        current_diagnosis_category = ""
        # "general_risk"가 큐의 첫 번째 항목일 때만 처리
        if pending_diagnoses and len(pending_diagnoses) > 0 and pending_diagnoses[0] == "general_risk":
            current_diagnosis_category = pending_diagnoses.pop(0)
            print(f"  처리 중인 일반 진단 카테고리: {current_diagnosis_category}")
            state["pending_specific_diagnoses"] = pending_diagnoses
        else:
            if not pending_diagnoses or len(pending_diagnoses) == 0:
                print("  Warning: 일반 진단 큐가 비어있습니다. 'general_risk'를 처리할 수 없습니다.")
            else:
                print(f"  Warning: 일반 진단 노드에 잘못된 큐가 전달되었습니다. 다음 항목: {pending_diagnoses[0]}. 'general_risk'만 처리합니다.")
            return state

        # "general_risk"에 대한 일반적인 윤리 점검 로직
        # 예: 자율점검표의 핵심 공통 항목들에 대한 평가 또는 LLM을 사용한 전반적인 검토
        print(f"  일반 윤리 위험 진단 수행: {current_diagnosis_category} for {service_name}")

        # LLM을 사용하여 서비스 설명 기반으로 일반적인 윤리적 고려사항 도출 (예시)
        prompt = f"""
        다음 AI 서비스에 대한 일반적인 윤리적 고려 사항을 검토하고, 주요 점검 항목과 간단한 코멘트를 제공해주세요.
        서비스명: {service_name}
        서비스 설명: {service_initial_info}
        분석된 서비스 정보: {analyzed_info}

        주요 점검 항목 (예시):
        - 데이터 프라이버시 보호는 적절한가?
        - 서비스의 투명성과 설명 가능성은 확보되었는가?
        - 잠재적인 오용 가능성에 대한 대비가 있는가?
        - 사회적 편익과 위험 간의 균형은 고려되었는가?

        결과는 아래 형식의 JSON 객체로만 응답해주세요. (다른 설명 없이 JSON만 출력)
        {{
            "E00.01_data_privacy_general": "데이터 수집, 활용, 폐기 전 과정에서 개인정보보호 원칙 준수 여부 (상세 검토 필요).",
            "E00.02_transparency_accountability_general": "서비스 작동 방식, 결정 근거에 대한 사용자 이해도 제고 및 책임 소재 명확화 필요 (상세 검토 필요).",
            "E00.03_misuse_potential_general": "악의적 사용 또는 의도치 않은 부정적 결과 발생 가능성 최소화 방안 검토 (상세 검토 필요).",
            "E00.04_societal_impact_general": "서비스가 사회 및 환경에 미치는 긍정적/부정적 영향 분석 및 균형점 모색 (상세 검토 필요)."
        }}
        """
        try:
            response = await self.llm.ainvoke(prompt)
            response_content = str(response.content)
            
            # LLM 응답에서 JSON 부분만 추출 (간단한 방식)
            json_start_index = response_content.find('{')
            json_end_index = response_content.rfind('}') + 1
            if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                json_str = response_content[json_start_index:json_end_index]
                import json # json 모듈 임포트
                general_risks_from_llm = json.loads(json_str)
                
                if state.get("common_ethics_risks") is None:
                    state["common_ethics_risks"] = {}
                
                for key, value in general_risks_from_llm.items():
                    state["common_ethics_risks"][key] = value
                print(f"  LLM 기반 일반 윤리 위험 분석 결과 저장됨: {list(general_risks_from_llm.keys())}")

            else: # LLM이 JSON을 제대로 반환하지 못한 경우 기본값 사용
                print("  Warning: LLM으로부터 일반 윤리 위험 분석 결과를 JSON 형식으로 받지 못했습니다. 기본값을 사용합니다.")
                if state.get("common_ethics_risks") is None: state["common_ethics_risks"] = {}
                state["common_ethics_risks"]["E00.00_general_overview_fallback"] = "서비스 전반의 일반 윤리 원칙 준수 여부 검토 (LLM 분석 실패, 상세 수동 검토 필요)."

        except Exception as e:
            print(f"  LLM 호출 중 오류 발생 (General Diagnosis): {e}")
            if state.get("common_ethics_risks") is None: state["common_ethics_risks"] = {}
            state["common_ethics_risks"]["E00.00_general_overview_error"] = f"서비스 전반의 일반 윤리 원칙 준수 여부 검토 (LLM 오류: {e}, 상세 수동 검토 필요)."


        # 필요시, 자율점검표 DB에서 "일반" 또는 "공통" 카테고리의 항목들을 검색하여 추가 분석
        # common_checklist_items = search_documents(checklist_collection, query_texts=["일반 원칙", "공통 고려사항"], n_results=5)
        # if common_checklist_items:
        #     # ... (검색된 항목들을 바탕으로 추가적인 상태 업데이트) ...
        #     print(f"  자율점검표 기반 일반 항목 {len(common_checklist_items)}개 추가 검토됨.")

        print("--- Ethics Risk Diagnoser Agent (General): 진단 완료 ---")
        return state

# --- 에이전트 테스트용 (직접 실행 시) ---
async def run_standalone_test():
    initial_state_general: ProjectState = {
        "service_name": "AI 기반 스마트 시티 관제 시스템",
        "service_type": "기타", # 또는 특정되지 않은 복합 서비스
        "service_initial_info": "도시 내 CCTV, 센서 데이터를 통합 분석하여 교통 흐름 최적화, 긴급 상황 감지 및 대응을 지원하는 시스템입니다.",
        "analyzed_service_info": {
            "target_functions": "교통 최적화, 위험 감지, 공공 안전 지원",
            "key_features": "실시간 데이터 분석, 예측 모델, 자동화된 알림",
            "data_usage_estimation": "CCTV 영상, 위치 정보, 센서 데이터 (대규모, 민감 정보 포함 가능성)",
            "estimated_users_stakeholders": "시청 공무원, 경찰, 소방, 시민"
        },
        "main_risk_categories": ["general_risk"], # RiskClassifier가 이렇게 분류했다고 가정
        "pending_specific_diagnoses": ["general_risk"], # General Diagnoser가 이 큐를 받음
        "common_ethics_risks": None,
        "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None,
        "error_message": None
    }

    diagnoser_agent = EthicsRiskDiagnoserGeneralAgent()

    print("\n--- General Risk Diagnosis Test ---")
    updated_state_gen = await diagnoser_agent.diagnose(initial_state_general)
    
    if updated_state_gen.get("error_message"):
        print(f"  Error: {updated_state_gen['error_message']}")
    
    if updated_state_gen.get("common_ethics_risks"):
        print("  Common Ethics Risks (General Diagnosis):")
        for risk_code, risk_desc in updated_state_gen["common_ethics_risks"].items():
            print(f"    - {risk_code}: {risk_desc}")
            
    print(f"  Pending diagnoses after: {updated_state_gen.get('pending_specific_diagnoses')}")


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