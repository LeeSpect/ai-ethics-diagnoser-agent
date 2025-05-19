from typing import List, Optional, Dict, Any

try:
    from ..core.states import ProjectState, PastCaseAnalysis # 필요한 타입 임포트
    from ..core.db_utils import search_documents, checklist_collection, past_cases_collection # DB 유틸리티
    from ..core.config import LLM_MODEL_NAME, OPENAI_API_KEY # LLM 설정
except ImportError:
    from core.states import ProjectState, PastCaseAnalysis
    from core.db_utils import search_documents, checklist_collection, past_cases_collection
    from core.config import LLM_MODEL_NAME, OPENAI_API_KEY

from langchain_openai import ChatOpenAI

# from prompts.specific_diagnoser_prompts import CHATBOT_BIAS_DIAGNOSIS_PROMPT, CHATBOT_PRIVACY_DIAGNOSIS_PROMPT
# Tavily 검색 도구 (선택적)
# from langchain_community.tools.tavily_search import TavilySearchResults # ServiceAnalyzer와 유사하게 초기화 가능

# --- 프롬프트 예시 (실제로는 prompts 폴더에 정의) ---
CHATBOT_SPECIFIC_DIAGNOSIS_PROMPT_TEMPLATE = """ 
AI 챗봇 서비스 '{service_name}'의 특정 윤리 리스크 심층 진단 요청:

진단 대상 리스크 카테고리: {risk_category}

서비스 분석 정보:
- 대상 기능: {target_functions}
- 주요 특징: {key_features}
- 데이터 수집/활용 (추정): {data_usage_estimation}

참고 과거 사례 (DB 검색 결과):
{retrieved_past_cases}

추가 웹 검색 정보 (Tavily):
{web_search_results_section}

위 정보를 종합하여, '{service_name}' 서비스가 '{risk_category}' 측면에서 가질 수 있는 구체적인 윤리적 문제점, 기술적/정책적 취약점, 그리고 과거 유사 사례로부터 얻을 수 있는 교훈 및 현재 서비스에 대한 시사점을 분석해 주십시오.

분석 결과를 다음 JSON 형식으로 반환해 주십시오:
{{
"specific_issue_description": "해당 리스크 카테고리에 대한 서비스의 구체적인 문제점 또는 우려 사항 상세 기술...",
"identified_vulnerabilities": "기술적, 정책적, 운영적 취약점 목록...",
"lessons_from_past_cases": "참고한 과거 사례들로부터 얻은 주요 교훈...",
"implications_for_target_service": "현재 서비스에 대한 구체적인 시사점 및 잠재적 영향..."
}}
"""

class EthicsRiskDiagnoserChatbotAgent:
    def __init__(self, use_web_search: bool = True): # 웹 검색 기본 사용
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.3, openai_api_key=OPENAI_API_KEY)
        self.use_web_search = use_web_search
        self.web_search_tool: Optional[TavilySearchResults] = None
        if self.use_web_search:
            if not TAVILY_API_KEY:
                print("Warning: EthicsRiskDiagnoserChatbotAgent - TAVILY_API_KEY가 없어 웹 검색 비활성화.")
                self.use_web_search = False
            elif TavilySearchResults is None:
                print("Warning: EthicsRiskDiagnoserChatbotAgent - TavilySearchResults를 임포트할 수 없어 웹 검색 비활성화.")
                self.use_web_search = False
            else:
                try:
                    self.web_search_tool = TavilySearchResults(max_results=2, tavily_api_key=TAVILY_API_KEY) # 결과 수 조절
                    print("EthicsRiskDiagnoserChatbotAgent: Tavily Web Search tool initialized.")
                except Exception as e:
                    print(f"Warning: EthicsRiskDiagnoserChatbotAgent - TavilySearchResults 초기화 오류: {e}. 웹 검색 비활성화.")
                    self.use_web_search = False
        print("EthicsRiskDiagnoserChatbotAgent initialized.")

    async def _perform_specific_web_search(self, query: str) -> str:
        if not self.use_web_search or not self.web_search_tool:
            return "(웹 검색 비활성화 또는 도구 초기화 실패)"
        print(f"  Chatbot Diagnoser - Tavily 웹 검색 수행: '{query}'")
        try:
            import asyncio # _perform_specific_web_search 내에서만 사용하므로 여기에 import
            loop = asyncio.get_running_loop()
            # TavilySearchResults.invoke는 동기 함수이므로 run_in_executor 사용
            search_results_list: List[Dict[str, Any]] = await loop.run_in_executor(
                None, self.web_search_tool.invoke, {"query": query} # invoke 인자를 딕셔너리로 전달
            )
            if not search_results_list: return "웹 검색 결과 없음."

            formatted_results = "\n"
            # search_results_list는 content, title, url 키를 가진 딕셔너리 리스트일 것으로 예상
            for i, res_dict in enumerate(search_results_list):
                formatted_results += f"{i+1}. 제목: {res_dict.get('title', 'N/A')}\n   URL: {res_dict.get('url', 'N/A')}\n   내용: {res_dict.get('content', 'N/A')[:200]}...\n"
            return formatted_results
        except Exception as e:
            print(f"  Chatbot Diagnoser - Tavily 웹 검색 오류: {e}")
            return "\n웹 검색 중 오류 발생.\n"

    async def diagnose(self, state: ProjectState) -> ProjectState:
        print("--- Ethics Risk Diagnoser Agent (Chatbot Focus): 특화 진단 시작 ---")
        pending_diagnoses: List[str] = state.get("pending_specific_diagnoses", [])
        analyzed_info: Optional[ServiceAnalysisOutput] = state.get("analyzed_service_info")
        service_name = state.get("service_name", "N/A")

        if not analyzed_info:
            print("  Error: 서비스 분석 정보가 없어 챗봇 특화 진단을 진행할 수 없습니다.")
            state["error_message"] = "챗봇 특화 진단: 서비스 분석 정보 누락"
            return state

        if not pending_diagnoses:
            print("  Info: 챗봇 특화 진단 큐가 비어있습니다.")
            return state

        current_chatbot_risk_category = ""
        category_to_process = pending_diagnoses[0]
        if "chatbot" not in category_to_process:
            print(f"  Warning: 챗봇 진단기에 '{category_to_process}' 카테고리가 전달되었습니다. 이 에이전트는 챗봇 관련 리스크만 처리합니다. 건너<0xEB><0><0x8A><0x8D>니다.")
            return state
        
        current_chatbot_risk_category = category_to_process
        print(f"  처리할 챗봇 특화 진단 카테고리: {current_chatbot_risk_category}")

        retrieved_past_cases_str = "과거 사례 검색 결과 없음."
        if past_cases_collection:
            try:
                db_query = f"{service_name} 서비스의 {current_chatbot_risk_category} 관련 AI 윤리 문제 사례"
                print(f"  ChromaDB 과거 사례 검색 중... Query: '{db_query}'")
                raw_past_cases = search_documents(db_query, past_cases_collection, n_results=TOP_K_RESULTS)
                if raw_past_cases and isinstance(raw_past_cases, list) and all(isinstance(doc, dict) for doc in raw_past_cases):
                    summaries = [doc.get('document', '') for doc in raw_past_cases if doc.get('document')]
                    if summaries:
                        retrieved_past_cases_str = "\n".join([f"- {summary}" for summary in summaries])
                    else:
                        retrieved_past_cases_str = "관련 과거 사례 요약을 찾지 못했습니다."
                else:
                    retrieved_past_cases_str = "관련 과거 사례를 데이터베이스에서 찾지 못했거나 형식이 맞지 않습니다."
                print(f"  과거사례 검색 결과(프롬프트용 일부): {retrieved_past_cases_str[:200]}...")
            except Exception as e:
                print(f"  Error during ChromaDB past_cases_collection search: {e}")
                retrieved_past_cases_str = "과거 사례 검색 중 오류 발생."
        else:
            retrieved_past_cases_str = "과거 사례 데이터베이스 (past_cases_collection)에 접근할 수 없습니다."

        web_search_results_section = "\n(웹 검색 비활성화 또는 결과 없음)\n"
        if self.use_web_search and self.web_search_tool:
            web_query = f"AI 서비스 '{service_name}'의 '{current_chatbot_risk_category}' 관련 윤리적 문제 최신 정보 및 사례"
            raw_web_results = await self._perform_specific_web_search(web_query) # _perform_specific_web_search 호출
            if raw_web_results and "오류" not in raw_web_results and "없음" not in raw_web_results:
                web_search_results_section = f"\n--- 추가 웹 검색 정보 ---\n{raw_web_results}\n"

        prompt = CHATBOT_SPECIFIC_DIAGNOSIS_PROMPT_TEMPLATE.format(
            service_name=service_name,
            risk_category=current_chatbot_risk_category,
            target_functions=analyzed_info.get("target_functions", "N/A"),
            key_features=analyzed_info.get("key_features", "N/A"),
            data_usage_estimation=analyzed_info.get("data_usage_estimation", "N/A"),
            retrieved_past_cases=retrieved_past_cases_str,
            web_search_results_section=web_search_results_section
        )

        llm_response_json_data: Optional[Dict[str, Any]] = None
        response_content = "" # LLM 응답 저장용
        try:
            print(f"  LLM 호출하여 '{current_chatbot_risk_category}' 심층 분석 중...")
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
                state["error_message"] = f"{current_chatbot_risk_category} 진단 LLM 응답 JSON 파싱 실패: 유효한 JSON 형식 아님"
                state["pending_specific_diagnoses"] = pending_diagnoses[1:]
                return state
        except json.JSONDecodeError as e:
            print(f"  Error: LLM 응답 JSON 파싱 오류 ({current_chatbot_risk_category}): {e}. 응답: {response_content}")
            state["error_message"] = f"{current_chatbot_risk_category} 진단 LLM 응답 JSON 파싱 오류: {e}"
            state["pending_specific_diagnoses"] = pending_diagnoses[1:]
            return state
        except Exception as e:
            print(f"  Error: LLM 호출 중 오류 발생 ({current_chatbot_risk_category}): {e}")
            state["error_message"] = f"{current_chatbot_risk_category} 진단 LLM 호출 오류: {e}"
            state["pending_specific_diagnoses"] = pending_diagnoses[1:]
            return state

        if llm_response_json_data:
            if state.get("specific_ethics_risks_by_category") is None:
                state["specific_ethics_risks_by_category"] = {}
            state["specific_ethics_risks_by_category"][current_chatbot_risk_category] = llm_response_json_data
            print(f"  '{current_chatbot_risk_category}'에 대한 심층 분석 결과 저장됨.")

            if state.get("past_case_analysis_results_by_category") is None:
                state["past_case_analysis_results_by_category"] = {}
            
            synthetic_past_case = PastCaseAnalysis(
                case_name=f"{current_chatbot_risk_category} 관련 과거 사례로부터의 교훈 및 시사점 (LLM 분석 기반)",
                cause_and_structure=llm_response_json_data.get("lessons_from_past_cases", "과거 사례로부터의 교훈 정보 없음."),
                vulnerabilities=llm_response_json_data.get("identified_vulnerabilities", "식별된 취약점 정보 없음."),
                comparison_with_target_service=llm_response_json_data.get("implications_for_target_service", "현재 서비스에 대한 시사점 정보 없음."),
                key_insights=(
                    f"주요 교훈: {llm_response_json_data.get('lessons_from_past_cases', 'N/A')}\n"
                    f"현재 서비스 시사점: {llm_response_json_data.get('implications_for_target_service', 'N/A')}"
                )
            )
            state["past_case_analysis_results_by_category"][current_chatbot_risk_category] = [synthetic_past_case]
            print(f"  '{current_chatbot_risk_category}'에 대한 과거 사례 분석 결과 (LLM 기반 종합) 저장됨.")
        
        state["pending_specific_diagnoses"] = pending_diagnoses[1:]
        
        print(f"--- Ethics Risk Diagnoser Agent (Chatbot Focus): '{current_chatbot_risk_category}' 특화 진단 완료 ---")
        return state