import json
import asyncio # 비동기 웹 검색 호출
from typing import Dict, Optional, List, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

try:
    from ..core.states import ProjectState, ServiceAnalysisOutput
    from ..core.config import OPENAI_API_KEY, TAVILY_API_KEY, LLM_MODEL_NAME
    from ..prompts.analyzer_prompts import SERVICE_ANALYSIS_PROMPT_TEMPLATE, WEB_SEARCH_RESULTS_SECTION_TEMPLATE
except ImportError: # 직접 실행 또는 테스트 시
    from core.states import ProjectState, ServiceAnalysisOutput
    from core.config import OPENAI_API_KEY, TAVILY_API_KEY, LLM_MODEL_NAME
    from prompts.analyzer_prompts import SERVICE_ANALYSIS_PROMPT_TEMPLATE, WEB_SEARCH_RESULTS_SECTION_TEMPLATE


class ServiceAnalyzerAgent:
    def __init__(self, use_web_search: bool = True): # 웹 검색 기본 사용
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.3, openai_api_key=OPENAI_API_KEY)
        self.use_web_search = use_web_search
        self.web_search_tool: Optional[TavilySearchResults] = None

        if self.use_web_search:
            if not TAVILY_API_KEY:
                print("Warning: TAVILY_API_KEY가 설정되지 않아 웹 검색을 비활성화합니다.")
                self.use_web_search = False
            else:
                try:
                    self.web_search_tool = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY)
                    print("ServiceAnalyzerAgent: Tavily Web Search tool initialized.")
                except Exception as e:
                    print(f"Warning: TavilySearchResults 초기화 중 오류 발생: {e}. 웹 검색을 비활성화합니다.")
                    self.use_web_search = False

        print(f"ServiceAnalyzerAgent initialized. LLM: {LLM_MODEL_NAME}, Web Search: {self.use_web_search}")

    async def _perform_web_search(self, query: str) -> str:
        """Tavily 웹 검색을 비동기적으로 수행하고 결과를 문자열로 반환합니다."""
        if not self.web_search_tool:
            return "웹 검색 도구가 초기화되지 않았습니다."

        print(f"  Tavily 웹 검색 수행: '{query}'")
        try:
            loop = asyncio.get_running_loop()

            search_results_list_of_dicts: List[Dict[str, Any]] = await loop.run_in_executor(
                None, self.web_search_tool.invoke, query
            )

            if not search_results_list_of_dicts:
                return "웹 검색 결과가 없습니다."

            # 결과를 하나의 문자열로 포맷팅
            formatted_results = "\n\n웹 검색 결과:\n"
            for i, res_dict in enumerate(search_results_list_of_dicts):
                title = res_dict.get("title", "제목 없음")
                url = res_dict.get("url", "URL 없음")
                content = res_dict.get("content", "내용 없음")
                formatted_results += f"{i+1}. 제목: {title}\n   URL: {url}\n   내용: {content[:300]}...\n" # 내용 일부만
            return formatted_results
        except Exception as e:
            print(f"  Tavily 웹 검색 중 오류 발생: {e}")
            return "\n웹 검색 중 오류가 발생했습니다.\n"


    async def analyze_service(self, state: ProjectState) -> ProjectState:
        print("--- Service Analyzer Agent: 서비스 분석 시작 ---")
        service_name = state.get("service_name", "알 수 없는 서비스")
        service_type = state.get("service_type", "기타")
        service_initial_info = state.get("service_initial_info", "제공된 설명 없음")

        web_search_results_section = ""
        if self.use_web_search and self.web_search_tool:
            search_query = f"{service_name} ({service_type}) 기능, 특징, 데이터 정책"
            web_results_str = await self._perform_web_search(search_query)
            if web_results_str and "오류" not in web_results_str and "없습니다" not in web_results_str :
                web_search_results_section = WEB_SEARCH_RESULTS_SECTION_TEMPLATE.format(web_results=web_results_str)
            else:
                web_search_results_section = "\n(웹에서 추가 정보를 가져오지 못했거나 검색 결과가 없습니다.)\n"


        prompt = SERVICE_ANALYSIS_PROMPT_TEMPLATE.format(
            service_name=service_name,
            service_type=service_type,
            service_initial_info=service_initial_info,
            web_search_results_section=web_search_results_section
        )

        try:
            print("  LLM 호출하여 서비스 분석 중...")
            response = await self.llm.ainvoke(prompt)
            response_content = str(response.content) # content가 항상 str이 아닐 수 있으므로 명시적 변환
            print(f"  LLM 응답 수신 (일부): {response_content[:300]}...")

            try:
                json_start_index = response_content.find('{')
                json_end_index = response_content.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_str = response_content[json_start_index:json_end_index]
                    analyzed_data_dict = json.loads(json_str)

                    analyzed_output: ServiceAnalysisOutput = {
                        "target_functions": analyzed_data_dict.get("target_functions", "분석 정보 없음"),
                        "key_features": analyzed_data_dict.get("key_features", "분석 정보 없음"),
                        "data_usage_estimation": analyzed_data_dict.get("data_usage_estimation", "분석 정보 없음"),
                        "estimated_users_stakeholders": analyzed_data_dict.get("estimated_users_stakeholders", "분석 정보 없음"),
                    }
                    state["analyzed_service_info"] = analyzed_output
                    print("  서비스 분석 정보가 상태에 저장되었습니다.")
                else:
                    print(f"  LLM 응답에서 유효한 JSON 객체를 찾을 수 없습니다. 응답: {response_content}")
                    state["error_message"] = "서비스 분석 LLM 응답에서 JSON 파싱 실패: 유효한 JSON 형식이 아님"
                    state["analyzed_service_info"] = None

            except json.JSONDecodeError as e:
                print(f"  LLM 응답 JSON 파싱 오류: {e}")
                print(f"  원본 LLM 응답: {response_content}")
                state["error_message"] = f"서비스 분석 LLM 응답 JSON 파싱 오류: {e}"
                state["analyzed_service_info"] = None

        except Exception as e:
            print(f"  LLM 호출 중 오류 발생: {e}")
            state["error_message"] = f"서비스 분석 LLM 호출 오류: {e}"
            state["analyzed_service_info"] = None

        print("--- Service Analyzer Agent: 서비스 분석 완료 ---")
        return state

    @staticmethod
    async def run_standalone_test():
        from dotenv import load_dotenv
        import os # Ensure os is imported here if not already in this scope
        import asyncio
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        dotenv_path = os.path.join(project_root, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            print(f"Loaded .env from {dotenv_path}")
        else:
            print(f"Warning: .env file not found at {dotenv_path}")

        # This import might be problematic if core.config relies on relative imports
        # and the script is run directly. Consider adjusting if issues persist.
        try:
            from ..core.config import OPENAI_API_KEY, TAVILY_API_KEY
        except ImportError: # Fallback for direct execution
            from core.config import OPENAI_API_KEY, TAVILY_API_KEY


        if not OPENAI_API_KEY:
            print("Standalone Test Error: OPENAI_API_KEY is not set.")
            return
        if not TAVILY_API_KEY:
            print("Standalone Test Warning: TAVILY_API_KEY is not set. Web search will be skipped or fail if enabled.")


        initial_state_test: ProjectState = {
            "service_name": "AI 사진 편집기 '포토매직'",
            "service_type": "이미지 생성 AI",
            "service_initial_info": "사용자가 업로드한 사진을 AI를 사용하여 자동으로 보정하고, 다양한 예술적 필터를 적용하거나, 사진 속 배경을 변경하는 기능을 제공하는 서비스입니다. 모바일 앱 형태로 제공됩니다.",
            "analyzed_service_info": None, "main_risk_categories": None, "pending_specific_diagnoses": None,
            "common_ethics_risks": None, "specific_ethics_risks_by_category": None,
            "past_case_analysis_results_by_category": None, "issue_found": None, "report_type": None,
            "improvement_recommendations": None, "final_report_markdown": None, "error_message": None
        }

        # Need to instantiate the agent to call analyze_service
        # The class name ServiceAnalyzerAgent is in scope here.
        analyzer_agent = ServiceAnalyzerAgent(use_web_search=True)
        updated_state = await analyzer_agent.analyze_service(initial_state_test)

        print("\n--- 최종 상태 (Service Analyzer Agent 테스트 후) ---")
        if updated_state.get("error_message"):
            print(f"에러 발생: {updated_state['error_message']}")

        analyzed_info = updated_state.get("analyzed_service_info")
        if analyzed_info:
            print("분석된 서비스 정보:")
            print(f"  대상 기능: {analyzed_info['target_functions']}")
            print(f"  주요 특징: {analyzed_info['key_features']}")
            print(f"  데이터 수집/활용 (추정): {analyzed_info['data_usage_estimation']}")
            print(f"  예상 사용자/이해관계자: {analyzed_info['estimated_users_stakeholders']}")
        else:
            print("서비스 분석 정보가 생성되지 않았습니다.")

# --- 에이전트 테스트용 (직접 실행) ---

if __name__ == '__main__':
    import asyncio
    # The call remains the same, but now it correctly calls the static method
    asyncio.run(ServiceAnalyzerAgent.run_standalone_test())