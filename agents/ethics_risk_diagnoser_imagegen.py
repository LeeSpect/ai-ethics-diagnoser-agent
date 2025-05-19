from typing import List, Optional, Dict, Any

try:
    from ..core.states import ProjectState, PastCaseAnalysis
    from ..core.db_utils import search_documents, checklist_collection, past_cases_collection # 필요시 사용
    from ..core.config import LLM_MODEL_NAME, OPENAI_API_KEY
except ImportError:
    from core.states import ProjectState, PastCaseAnalysis
    from core.db_utils import search_documents, checklist_collection, past_cases_collection
    from core.config import LLM_MODEL_NAME, OPENAI_API_KEY

from langchain_openai import ChatOpenAI

class EthicsRiskDiagnoserImageGenAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.2, openai_api_key=OPENAI_API_KEY)
        print("EthicsRiskDiagnoserImageGenAgent initialized.")

    async def diagnose(self, state: ProjectState) -> ProjectState:
        print("--- Ethics Risk Diagnoser Agent (ImageGen Focus): 진단 시작 ---")
        pending_diagnoses: Optional[List[str]] = state.get("pending_specific_diagnoses")
        analyzed_info = state.get("analyzed_service_info") # 서비스 분석 정보
        service_name = state.get("service_name", "알 수 없는 서비스")

        current_diagnosis_category = ""
        if pending_diagnoses and len(pending_diagnoses) > 0:
            current_diagnosis_category = pending_diagnoses.pop(0)
            print(f"  처리 중인 이미지 생성 특화 진단 카테고리: {current_diagnosis_category}")
            state["pending_specific_diagnoses"] = pending_diagnoses
        else:
            print("  Warning: 이미지 생성 특화 진단 큐가 비어있습니다.")
            return state

        if current_diagnosis_category == "copyright_generation":
            print(f"  심층 진단 수행: {current_diagnosis_category} for {service_name}")
            analysis_result_text = f"'{service_name}'의 저작권 침해(copyright_generation) 분석 결과: 생성된 이미지가 기존 저작물과 유사하여 저작권을 침해할 가능성이 있습니다. 학습 데이터의 저작권 문제, 생성물의 독창성 부족 등이 원인이 될 수 있습니다. (상세 내용 추가 필요)"
            severity_level = "high"

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
                    case_name="예시 이미지 생성 AI 저작권 침해 사례",
                    cause_and_structure="인터넷에서 무단 수집한 저작권 보호 이미지를 학습 데이터로 사용하여, 생성된 이미지가 원본과 매우 유사하게 나타남.",
                    vulnerabilities="학습 데이터셋에 대한 저작권 검토 미흡, 생성 이미지의 독창성 검증 시스템 부재.",
                    comparison_with_target_service=f"'{service_name}' 또한 대규모 이미지 데이터로 학습했을 가능성이 있으므로, 학습 데이터의 출처 및 라이선스, 생성물의 유사도 검증이 중요함.",
                    key_insights="AI 생성물의 저작권 문제는 법적 분쟁 및 서비스 신뢰도 하락으로 이어질 수 있음. 명확한 라이선스의 데이터 사용, 생성물 필터링, 사용자 가이드라인 제공이 필요."
                )
            ]
            print(f"  {current_diagnosis_category} 진단 결과 저장됨.")

        elif current_diagnosis_category == "deepfake_generation":
            print(f"  심층 진단 수행: {current_diagnosis_category} for {service_name}")
            analysis_result_text = f"'{service_name}'의 딥페이크 악용(deepfake_generation) 분석 결과: 생성된 이미지가 특정 인물의 얼굴이나 모습을 악의적으로 합성하여 명예훼손, 가짜 뉴스 유포 등에 악용될 수 있습니다. (상세 내용 추가 필요)"
            severity_level = "critical"

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
                    case_name="예시 딥페이크 기술 악용 사례",
                    cause_and_structure="사용자가 업로드한 인물 사진을 기반으로 정교한 딥페이크 영상/이미지를 생성하여 온라인에 유포함.",
                    vulnerabilities="생성 기술의 높은 현실감, 악의적 사용자에 대한 탐지 및 제재 수단 미흡, 생성물에 대한 워터마크 또는 출처 표시 부재.",
                    comparison_with_target_service=f"'{service_name}'이 인물 이미지 생성을 지원하거나, 사용자가 업로드한 이미지를 변형하는 기능을 제공한다면 딥페이크 악용 방지 기술 및 정책이 필수적임.",
                    key_insights="딥페이크는 개인의 인격권 침해는 물론 사회적 혼란을 야기할 수 있는 심각한 위협임. 기술적 안전장치(워터마킹, 탐지 기술)와 함께 강력한 사용 정책, 법적 책임 고지가 필요."
                )
            ]
            print(f"  {current_diagnosis_category} 진단 결과 저장됨.")
        
        else:
            print(f"  Warning: 알 수 없는 이미지 생성 특화 진단 카테고리입니다: {current_diagnosis_category}")

        print("--- Ethics Risk Diagnoser Agent (ImageGen Focus): 진단 완료 ---")
        return state

# --- 에이전트 테스트용 (직접 실행) ---
async def run_standalone_test():
    initial_state_copyright: ProjectState = {
        "service_name": "AI 아트 생성기 '아티스트 AI'",
        "service_type": "이미지 생성 AI",
        "analyzed_service_info": {
            "target_functions": "텍스트 프롬프트 기반 예술 이미지 생성",
            "key_features": "다양한 화풍 지원, 고해상도 출력",
            "data_usage_estimation": "대규모 이미지-텍스트 쌍 데이터셋 학습",
            "estimated_users_stakeholders": "디자이너, 콘텐츠 크리에이터, 일반 사용자"
        },
        "main_risk_categories": ["copyright_generation", "deepfake_generation"],
        "pending_specific_diagnoses": ["copyright_generation"],
        "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None,
        "error_message": None
    }

    initial_state_deepfake: ProjectState = {
        "service_name": "AI 프로필 사진 편집기 '페이스튠 AI'",
        "service_type": "이미지 생성 AI",
        "analyzed_service_info": {
            "target_functions": "사용자 사진 기반 스타일 변환, 얼굴 특징 수정",
            "key_features": "자연스러운 보정, 다양한 필터",
            "data_usage_estimation": "사용자 업로드 사진, 얼굴 인식 데이터",
            "estimated_users_stakeholders": "SNS 사용자, 일반인"
        },
        "main_risk_categories": ["copyright_generation", "deepfake_generation"],
        "pending_specific_diagnoses": ["deepfake_generation"],
        "specific_ethics_risks_by_category": None,
        "past_case_analysis_results_by_category": None,
        "error_message": None
    }

    diagnoser_agent = EthicsRiskDiagnoserImageGenAgent()

    print("\n--- Copyright Generation Diagnosis Test ---")
    updated_state_cr = await diagnoser_agent.diagnose(initial_state_copyright)
    if updated_state_cr.get("error_message"):
        print(f"  Error: {updated_state_cr['error_message']}")
    if updated_state_cr.get("specific_ethics_risks_by_category"):
        print(f"  Specific Risks (Copyright): {updated_state_cr['specific_ethics_risks_by_category'].get('copyright_generation')}")
    if updated_state_cr.get("past_case_analysis_results_by_category"):
        print(f"  Past Cases (Copyright): {updated_state_cr['past_case_analysis_results_by_category'].get('copyright_generation')}")
    print(f"  Pending diagnoses after: {updated_state_cr.get('pending_specific_diagnoses')}")

    print("\n--- Deepfake Generation Diagnosis Test ---")
    updated_state_df = await diagnoser_agent.diagnose(initial_state_deepfake)
    if updated_state_df.get("error_message"):
        print(f"  Error: {updated_state_df['error_message']}")
    if updated_state_df.get("specific_ethics_risks_by_category"):
        print(f"  Specific Risks (Deepfake): {updated_state_df['specific_ethics_risks_by_category'].get('deepfake_generation')}")
    if updated_state_df.get("past_case_analysis_results_by_category"):
        print(f"  Past Cases (Deepfake): {updated_state_df['past_case_analysis_results_by_category'].get('deepfake_generation')}")
    print(f"  Pending diagnoses after: {updated_state_df.get('pending_specific_diagnoses')}")


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