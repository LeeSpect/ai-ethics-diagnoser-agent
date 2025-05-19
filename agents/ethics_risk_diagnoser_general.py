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

# 일반 윤리 진단용 프롬프트 템플릿
GENERAL_DIAGNOSIS_PROMPT_TEMPLATE = """
AI 서비스 '{service_name}'의 일반 윤리 고려사항 검토:

서비스 분석 정보:
- 대상 기능: {target_functions}
- 주요 특징: {key_features}
- 데이터 수집/활용 (추정): {data_usage_estimation}

자율점검표 공통 진단 결과 요약:
{common_ethics_risks_summary}

위 정보를 바탕으로 '{service_name}' 서비스 운영 시 전반적으로 고려해야 할 추가적인 윤리적 사항이나,
특정 카테고리에 속하지 않지만 중요하다고 판단되는 일반적인 윤리 리스크에 대해 기술해 주십시오.

결과를 다음 JSON 형식으로 반환해 주십시오:
{{
  "general_ethical_considerations": "일반적인 윤리적 고려사항 상세 기술...",
  "potential_general_risks": "특정 카테고리에 속하지 않는 잠재적 일반 리스크 목록..."
}}
"""

class EthicsRiskDiagnoserGeneralAgent:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.3, openai_api_key=OPENAI_API_KEY)
        print("EthicsRiskDiagnoserGeneralAgent initialized.")

    async def diagnose(self, state: ProjectState) -> ProjectState:
        print("--- Ethics Risk Diagnoser Agent (General Focus): 일반 진단 시작 ---")
        pending_diagnoses: List[str] = state.get("pending_specific_diagnoses", [])
        analyzed_info: Optional[Dict[str, Any]] = state.get("analyzed_service_info")
        service_name = state.get("service_name", "N/A")
        
        # common_ethics_risks는 일반적으로 Dict 형태일 것이므로 json.dumps로 문자열화
        common_ethics_risks = state.get("common_ethics_risks", {})
        common_risks_summary_str = json.dumps(common_ethics_risks, ensure_ascii=False, indent=2)

        current_general_risk_category = ""

        if not pending_diagnoses:
            print("  Info: 일반 진단 큐가 비어있습니다. (pending_specific_diagnoses is empty)")
            return state

        # 이 에이전트는 "general_risk"만 처리
        if pending_diagnoses[0] == "general_risk":
            current_general_risk_category = pending_diagnoses.pop(0)
            state["pending_specific_diagnoses"] = pending_diagnoses # 수정된 큐를 다시 상태에 할당
            print(f"  처리할 일반 진단 카테고리: {current_general_risk_category}")
        else:
            # 큐의 첫 항목이 "general_risk"가 아니면, 이 에이전트는 처리하지 않고 반환
            # 큐에서 항목을 제거하지 않음 (다른 에이전트가 처리할 수 있도록)
            print(f"  Info: 일반 진단 큐의 다음 항목 '{pending_diagnoses[0]}'은(는) 'general_risk'가 아니므로 건너<0xEB><0><0x8A><0x8D>니다.")
            return state

        if not analyzed_info:
            print("  Error: 서비스 분석 정보가 없어 일반 진단을 진행할 수 없습니다.")
            state["error_message"] = f"{current_general_risk_category} 일반 진단: 서비스 분석 정보 누락"
            return state

        prompt = GENERAL_DIAGNOSIS_PROMPT_TEMPLATE.format(
            service_name=service_name,
            target_functions=analyzed_info.get("target_functions", "N/A"),
            key_features=analyzed_info.get("key_features", "N/A"),
            data_usage_estimation=analyzed_info.get("data_usage_estimation", "N/A"),
            common_ethics_risks_summary=common_risks_summary_str
        )

        llm_response_json_data: Optional[Dict[str, Any]] = None
        response_content = ""
        try:
            print(f"  LLM 호출하여 '{current_general_risk_category}' 일반 분석 중...")
            llm_ainvoke_response = await self.llm.ainvoke(prompt)
            response_content = str(llm_ainvoke_response.content)
            print(f"  LLM 응답 수신 (일부): {response_content[:300]}...")

            json_start_index = response_content.find('{')
            json_end_index = response_content.rfind('}') + 1
            if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                json_str = response_content[json_start_index:json_end_index]
                llm_response_json_data = json.loads(json_str)
            else:
                print(f"  Error: LLM 응답에서 유효한 JSON 객체를 찾을 수 없습니다. 응답: {response_content}")
                state["error_message"] = f"{current_general_risk_category} 일반 진단 LLM 응답 JSON 파싱 실패"
                return state
        except json.JSONDecodeError as e:
            print(f"  Error: LLM 응답 JSON 파싱 오류 ({current_general_risk_category}): {e}. 응답: {response_content}")
            state["error_message"] = f"{current_general_risk_category} 일반 진단 LLM 응답 JSON 파싱 오류: {e}"
            return state
        except Exception as e:
            print(f"  Error: LLM 호출 중 오류 발생 ({current_general_risk_category}): {e}")
            state["error_message"] = f"{current_general_risk_category} 일반 진단 LLM 호출 오류: {e}"
            return state

        if llm_response_json_data:
            if state.get("specific_ethics_risks_by_category") is None:
                state["specific_ethics_risks_by_category"] = {}
            # "general_risk"에 대한 결과를 저장
            state["specific_ethics_risks_by_category"][current_general_risk_category] = llm_response_json_data
            print(f"  '{current_general_risk_category}' 일반 진단 결과 저장됨.")
        
        print(f"--- Ethics Risk Diagnoser Agent (General Focus): '{current_general_risk_category}' 일반 진단 완료 ---")
        return state

# --- 에이전트 테스트용 (직접 실행 시) ---
async def run_standalone_test():
    # 테스트를 위한 초기 상태 정의
    initial_state_general: ProjectState = {
        "service_name": "AI 챗봇 도우미 '헬프봇'",
        "service_type": "텍스트 기반 AI",
        "analyzed_service_info": {
            "target_functions": "사용자 질문에 대한 답변 생성, 정보 제공",
            "key_features": "자연어 이해, 다국어 지원",
            "data_usage_estimation": "대규모 텍스트 데이터셋 학습, 사용자 대화 기록",
            "estimated_users_stakeholders": "고객 지원팀, 일반 사용자"
        },
        "common_ethics_risks": { # 자율점검표 공통 진단 결과 예시
            "bias_fairness": "데이터 편향으로 인한 특정 그룹에 불리한 답변 생성 가능성",
            "privacy": "사용자 대화 내용 수집 및 분석에 따른 개인정보 침해 우려"
        },
        "pending_specific_diagnoses": ["general_risk", "other_risk_category"], # "general_risk"가 맨 앞에 있도록 설정
        "specific_ethics_risks_by_category": None,
        "error_message": None
    }

    diagnoser_agent = EthicsRiskDiagnoserGeneralAgent()

    print("\n--- General Risk Diagnosis Test ---")
    updated_state_gen = await diagnoser_agent.diagnose(initial_state_general)
    
    if updated_state_gen.get("error_message"):
        print(f"  Error: {updated_state_gen['error_message']}")
    
    if updated_state_gen.get("specific_ethics_risks_by_category"):
        general_risk_result = updated_state_gen['specific_ethics_risks_by_category'].get('general_risk')
        if general_risk_result:
            print(f"  General Ethical Considerations: {general_risk_result.get('general_ethical_considerations')}")
            print(f"  Potential General Risks: {general_risk_result.get('potential_general_risks')}")
        else:
            print("  General risk 진단 결과가 없습니다.")
            
    print(f"  Pending diagnoses after: {updated_state_gen.get('pending_specific_diagnoses')}") # other_risk_category가 남아있어야 함

if __name__ == '__main__':
    import asyncio
    import os
    from dotenv import load_dotenv
    
    # .env 파일 로드 (프로젝트 루트 기준)
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