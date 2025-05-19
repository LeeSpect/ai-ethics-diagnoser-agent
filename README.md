# AI 윤리성 리스크 진단 및 개선 권고 멀티 에이전트 시스템

**최종 업데이트: 2025년 5월 19일**

본 프로젝트는 특정 AI 서비스(챗봇, 추천 알고리즘, 이미지 생성 AI)를 대상으로 윤리적 리스크를 심층적으로 분석하고, 과거 AI 윤리 문제 사례와의 비교를 통해 실질적인 개선 권고안을 도출하는 멀티 에이전트 시스템입니다. LangGraph와 FastAPI를 기반으로 구현되었으며, "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)"을 핵심 기준으로 활용합니다.

## 1. 프로젝트 개요 (Overview)

-   **프로젝트 목표 (Objective)**:
    -   다양한 AI 서비스 유형(챗봇, 추천 알고리즘, 이미지 생성 AI)에 대한 윤리적 리스크(편향성, 프라이버시 침해, 투명성 부족, 저작권 침해, 딥페이크 악용 등)를 체계적으로 진단합니다.
    -   "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)"을 주요 평가 기준으로 사용하여 객관적이고 일관된 분석을 수행합니다.
    -   과거 국내외 AI 윤리 문제 사례를 심층 분석하고, 이를 현재 진단 대상 서비스와 비교하여 실질적인 인사이트를 도출합니다.
    -   분석 결과 및 인사이트를 바탕으로 구체적이고 실행 가능한 개선 권고안(단기, 중장기, 거버넌스 포함)을 제시합니다.
    -   최종적으로, AI 서비스의 안전성과 신뢰성을 제고하고 윤리적인 AI 개발 및 운영 환경 조성에 기여합니다.
-   **주요 분석 방법 (Methods)**:
    -   **조건부 및 순차적 다중 진단 멀티 에이전트 아키텍처**: LangGraph를 사용하여 서비스 분석 결과와 식별된 주요 리스크 카테고리 리스트에 따라, 필요한 특화 진단 에이전트들을 순차적으로 실행하고 그 결과를 누적하는 동적 에이전트 실행 경로를 결정합니다.
    -   **RAG (Retrieval Augmented Generation)**: ChromaDB에 저장된 "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)" PDF 및 과거 AI 윤리 사례 데이터를 활용하여, LLM이 보다 정확하고 근거 있는 분석 및 제안을 하도록 지원합니다.
    -   **동적 정보 수집**: Tavily Search API를 통해 최신 정보 및 추가적인 과거 AI 윤리 사례를 실시간으로 웹에서 수집하여 분석에 활용합니다.
    -   **LLM 기반 지능형 처리**: GPT-4o 모델을 활용하여 서비스 분석, 리스크 평가, 과거 사례 심층 분석, 개선안 제안, 그리고 최종 보고서 생성 등 핵심 지능형 작업을 수행합니다.
-   **활용 도구 (Tools)**: Python, LangGraph, LangChain, FastAPI, ChromaDB, Tavily Search API, OpenAI API (GPT-4o), Sentence Transformers (jhgan/ko-sroberta-multitask).

## 2. 주요 기능 (Features)

-   **다중 AI 서비스 유형 진단**: 챗봇, 추천 알고리즘, 이미지 생성 AI 서비스 각각의 특성을 고려한 맞춤형 윤리 리스크 진단 프로세스를 제공합니다.
-   **유연한 동적 에이전트 오케스트레이션**:
    -   서비스 분석 결과와 식별된 주요 리스크 유형에 따라 최적의 진단 에이전트가 조건부로 활성화됩니다.
    -   하나의 서비스가 여러 주요 리스크 카테고리를 가질 경우, 관련된 특화 심층 진단들을 순차적으로 모두 수행하고 그 결과를 종합합니다.
-   **심층적인 과거 사례 분석 및 비교**:
    -   ChromaDB에 구축된 과거 AI 윤리 문제 사례 데이터베이스와 Tavily 웹 검색 결과를 종합적으로 활용합니다.
    -   과거 사례의 발생 원인, 문제 구조, 기술적/정책적 취약점 등을 심층 분석합니다.
    -   분석된 과거 사례와 현재 진단 대상 서비스를 비교하여, 구체적인 시사점과 개선 방향에 대한 인사이트를 도출합니다.
-   **상황별 맞춤형 보고서 자동 생성**:
    -   진단 결과, 식별된 윤리적 문제의 심각성 및 개선안 도출 여부에 따라 두 가지 유형의 보고서(문제 없을 시/개선안 포함 시)를 자동으로 생성합니다.
    -   보고서는 명확한 구조와 함께 상세 분석 내용, 과거 사례 비교, 구체적인 개선 권고안을 포함합니다.
-   **API 기반 서비스 제공**: FastAPI를 통해 외부 시스템이나 사용자가 AI 윤리 진단 기능을 쉽게 요청하고, 구조화된 진단 결과를 보고서 형태로 받을 수 있는 RESTful API 인터페이스를 제공합니다.

## 3. 기술 스택 (Tech Stack)

| Category                | Details                                                        |
| :---------------------- | :------------------------------------------------------------- |
| 프로그래밍 언어         | Python 3.11                                                    |
| 핵심 프레임워크         | LangGraph (에이전트 오케스트레이션), LangChain (LLM 통합 및 도구) |
| API 서버                | FastAPI, Uvicorn                                               |
| LLM (Large Language Model)| GPT-4o (OpenAI API 활용)                                       |
| 임베딩 모델 (Embedding) | `jhgan/ko-sroberta-multitask` (Sentence Transformers)          |
| 벡터 데이터베이스 (VectorDB)| ChromaDB (RAG 및 과거 사례 데이터 저장/검색)                 |
| 웹 검색 도구            | Tavily Search API                                              |
| 데이터 직렬화/검증      | Pydantic (FastAPI 스키마 및 State 정의에 활용)                 |

## 4. 에이전트 구성 (Agents)

본 시스템은 다음과 같은 주요 에이전트들로 구성됩니다:

-   **`service_analyzer_agent` (서비스 분석 에이전트)**:
    -   **역할**: 사용자가 입력한 AI 서비스 정보(명칭, 유형, 설명 등)를 바탕으로 서비스의 구체적인 기능, 주요 특징, 예상되는 데이터 수집 및 활용 방식, 주요 사용자 및 이해관계자 그룹 등을 심층적으로 분석하고 구조화된 정보로 출력합니다.
    -   **활용 도구**: LLM (GPT-4o), (선택적으로) Tavily Search API.
-   **`risk_classifier_agent` (리스크 분류 에이전트)**:
    -   **역할**: `service_analyzer_agent`의 분석 결과와 사전 정의된 기준(사용자가 지정한 서비스 유형별 주요 우려 리스크)에 따라, 후속 진단 작업에서 집중적으로 다룰 **하나 이상의 주요 윤리 리스크 카테고리 리스트**(`main_risk_categories`)를 분류하여 `ProjectState`에 저장합니다. (예: `['bias_chatbot', 'privacy_chatbot']`)
    -   **활용 도구**: Python 로직 또는 간단한 LLM 판단.
-   **`specific_risk_diagnosis_router_agent` (특화 리스크 진단 라우터 에이전트)**:
    -   **역할**: `ProjectState`의 `pending_specific_diagnoses` (처리해야 할 특화 진단 카테고리 리스트)를 확인합니다.
        -   만약 처리할 진단 카테고리가 남아있으면, 리스트에서 첫 번째 카테고리를 가져와 해당 특화 진단 노드(예: `diagnoser_chatbot_node`)로 워크플로우를 라우팅합니다.
        -   처리할 진단 카테고리가 더 이상 없으면, `improvement_suggester_agent`로 워크플로우를 라우팅합니다.
    -   **활용 도구**: Python 로직.
-   **`ethics_risk_diagnoser_nodes` (윤리 리스크 진단 노드 - 유형별)**:
    -   실제로는 `diagnoser_chatbot_node`, `diagnoser_recommendation_node`, `diagnoser_image_gen_node` 등으로 나뉨. 각 노드는 할당된 특정 리스크 카테고리에 대해 심층 분석을 수행합니다.
    -   **공통 진단 (각 노드 시작 시 또는 별도 공통 노드에서 선행 수행)**: ChromaDB에 저장된 "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)" PDF 내용을 RAG 방식으로 참조하여, 진단 대상 서비스의 전반적인 윤리 리스크를 평가하고 관련 자율점검표 항목과의 부합 여부를 분석. 결과는 `common_ethics_risks`에 누적 저장.
    -   **심층/과거사례 분석 (특화된 부분)**: 현재 처리 중인 특정 리스크 카테고리(예: `bias_chatbot`)에 대해, ChromaDB에 저장된 관련 과거 AI 윤리 문제 사례 및 Tavily Search API로 수집한 정보를 심층 분석. 과거 사례의 원인, 구조, 취약점을 분석하고 현재 서비스와의 비교를 통해 인사이트 도출. 결과는 `specific_ethics_risks_by_category`와 `past_case_analysis_results_by_category`에 해당 카테고리명과 함께 누적 저장.
    -   작업 완료 후 `pending_specific_diagnoses` 리스트에서 현재 처리한 카테고리를 제거하고, 다시 `specific_risk_diagnosis_router_agent`로 흐름을 돌려보냅니다.
    -   **활용 도구**: LLM (GPT-4o), ChromaDB (RAG), Tavily Search API.
-   **`improvement_suggester_agent` (개선안 제안 에이전트)**:
    -   **역할**: 모든 진단 결과(`common_ethics_risks`, `specific_ethics_risks_by_category`, `past_case_analysis_results_by_category`)를 종합적으로 고려하여, 서비스의 윤리성 강화를 위한 개선 권고안을 제안합니다.
    -   **출력**: 개선안 내용과 함께, `issue_found` 상태값을 판단하여 반환합니다.
    -   **활용 도구**: LLM (GPT-4o).
-   **`report_type_decider_agent` (보고서 유형 결정 에이전트)**:
    -   **역할**: `issue_found` 상태값을 기반으로 생성할 보고서의 유형("문제 없음" 또는 "개선안 포함")을 결정합니다.
    -   **활용 도구**: Python 로직.
-   **`report_generator_agent` (리포트 작성 에이전트)**:
    -   **역할**: 모든 수집된 정보와 결정된 `report_type`을 바탕으로, 최종 보고서를 생성합니다.
    -   **활용 도구**: LLM (GPT-4o), Python 문자열 포맷팅.

## 5. 상태 정의 (LangGraph State)

-   `service_name`: (str) 진단 대상 서비스 명칭.
-   `service_type`: (Literal["챗봇", "추천 알고리즘", "이미지 생성 AI"]) AI 서비스 유형.
-   `service_initial_info`: (str or dict) 사용자가 입력한 초기 서비스 정보.
-   `analyzed_service_info`: (Optional[ServiceAnalysisOutput]) 서비스 분석 결과.
-   `main_risk_categories`: (Optional[List[str]]) `risk_classifier_agent`가 식별한 주요 윤리 리스크 카테고리 리스트. (예: `['bias_chatbot', 'privacy_chatbot']`)
-   `pending_specific_diagnoses`: (Optional[List[str]]) 아직 처리되지 않은 특화 진단 대상 리스크 카테고리 리스트. `main_risk_categories`로 초기화되며, 하나씩 처리될 때마다 제거됨.
-   `common_ethics_risks`: (Optional[Dict[str, str]]) 공통 윤리 리스크 진단 결과 (누적).
-   `specific_ethics_risks_by_category`: (Optional[Dict[str, Dict[str, str]]]) 각 특화 리스크 카테고리별 심층 진단 결과 (누적, 예: `{"bias_chatbot": {...}, "privacy_chatbot": {...}}`).
-   `past_case_analysis_results_by_category`: (Optional[Dict[str, List[PastCaseAnalysis]]]) 각 특화 리스크 카테고리별 과거 사례 분석 결과 (누적).
-   `improvement_recommendations`: (Optional[ImprovementMeasures]) 개선 권고안.
-   `issue_found`: (Optional[bool]) 개선안 필요 여부 (True/False).
-   `report_type`: (Optional[Literal["no_issue_report", "issue_improvement_report"]]) 생성할 보고서 유형.
-   `final_report_markdown`: (Optional[str]) 최종 생성된 보고서 (마크다운 형식).
-   `error_message`: (Optional[str]) 처리 중 발생한 에러 메시지.

## 6. 아키텍처 다이어그램 (Architecture Diagram)

![alt text](image.png)

**개념적 데이터 흐름:**
```
[FastAPI 요청: 서비스 정보]
|
v
[Service Analyzer Agent] --> [Risk Classifier Agent (main_risk_categories 생성, pending_specific_diagnoses 초기화)]
|
v
[Specific Risk Diagnosis Router Agent (pending_specific_diagnoses 확인)] --+
|                                                                     |
+--(조건: Chatbot 진단 필요)-----> [Diagnoser Chatbot Node] -----------+
|                                     (결과 누적, pending에서 제거)      | (다음 진단으로 루프)
+--(조건: RecSys 진단 필요)-------> [Diagnoser RecSys Node] ------------+
|                                     (결과 누적, pending에서 제거)      |
+--(조건: ImgGen 진단 필요)------> [Diagnoser ImgGen Node] ------------+
|                                     (결과 누적, pending에서 제거)      |
|                                                                     |
+--(조건: 모든 특화 진단 완료)--> [Improvement Suggester Agent] ----> [Report Type Decider Agent] --> [Report Generator Agent] --> [FastAPI 응답: 보고서]
```

## 7. 디렉토리 구조 (Directory Structure)

```
├── agents/
│   ├── init.py
│   ├── service_analyzer.py
│   ├── risk_classifier.py
│   ├── specific_risk_diagnosis_router.py # 추가됨
│   ├── ethics_risk_diagnoser_chatbot.py  # 또는 diagnoser 모듈 내 함수로 분리
│   ├── ethics_risk_diagnoser_recsys.py
│   ├── ethics_risk_diagnoser_imagegen.py
│   ├── improvement_suggester.py
│   ├── report_type_decider.py
│   └── report_generator.py
├── app/
│   ├── init.py
│   ├── main.py
│   └── schemas.py
├── core/
│   ├── init.py
│   ├── graph.py
│   ├── states.py
│   └── tools.py
├── data/
│   └── 2025_인공지능_윤리기준_실천을_위한_자율점검표(안).pdf
├── prompts/
│   ├── report_generator_prompts.py
│   └── (기타 에이전트별 프롬프트)
├── resources/
│   └── chromadb/
├── outputs/
├── tests/
│   ├── agents/
│   └── core/
├── .env
├── config.py
├── requirements.txt
├── main_cli.py
└── README.md
```

## 8. 설치 및 실행 (Setup and Execution)

**(추후 상세 내용 추가 예정)**

1.  **저장소 복제**: `git clone {repository_url}`
2.  **가상 환경 생성 및 활성화**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3.  **의존성 설치**: `pip install -r requirements.txt`
4.  **환경 변수 설정**: `.env` 파일을 생성하고 필요한 API 키(OpenAI, Tavily 등)를 입력합니다.
    ```env
    OPENAI_API_KEY="your_openai_api_key"
    TAVILY_API_KEY="your_tavily_api_key"
    # 기타 설정
    ```
5.  **ChromaDB 데이터 로드**: (초기 데이터 로드 스크립트 실행 - `scripts/load_chroma_data.py` 와 같은 형태로 구현 필요)
    - "2025 인공지능 윤리기준 실천을 위한 자율점검표(안).pdf" 임베딩 및 저장
    - (선택적) 초기 과거 AI 윤리 사례 데이터 임베딩 및 저장
6.  **FastAPI 서버 실행**: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
7.  **API 테스트**: HTTP 클라이언트 (예: Postman, curl) 또는 웹 브라우저를 사용하여 API 엔드포인트 테스트.


## 9. 기여자 (Contributors)

-   이주형: 프로젝트 기획, 프롬프트 엔지니어링, 에이전트 설계