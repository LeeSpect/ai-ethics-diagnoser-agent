# prompts/analyzer_prompts.py

SERVICE_ANALYSIS_PROMPT_TEMPLATE = """
AI 서비스 분석 요청:

입력된 AI 서비스 정보:
- 서비스명: {service_name}
- 서비스 유형: {service_type}
- 사용자 제공 설명: {service_initial_info}
{web_search_results_section}

위 정보를 종합하여 다음 항목에 대해 상세히 분석해 주십시오. 각 항목은 명확하고 구체적인 정보를 포함해야 합니다.
만약 정보가 부족하여 추정해야 하는 경우, '추정됨' 또는 '예상됨'이라고 명시해 주십시오.

1.  **대상 기능 (Target Functions)**:
    - 이 서비스의 핵심 기능은 무엇이며, 사용자에게 어떤 가치를 제공합니까? (주요 기능 2-3가지 이상 상세 기술)

2.  **주요 특징 (Key Features)**:
    - 이 서비스만의 독특하거나 차별화되는 주요 특징은 무엇입니까? (기술적, 사용자 경험적, 사업적 측면 등)

3.  **데이터 수집 및 활용 방식 (추정) (Data Collection and Usage - Estimated)**:
    - 서비스 운영을 위해 어떤 종류의 데이터를 수집할 것으로 예상됩니까? (예: 사용자 입력 텍스트, 이미지, 개인 식별 정보, 사용 기록, 웹 쿠키, 외부 연동 데이터 등)
    - 수집된 데이터는 AI 모델 학습 및 서비스 제공에 어떻게 활용될 것으로 예상됩니까? (구체적인 활용 시나리오 명시)
    - 개인정보보호와 관련된 민감한 데이터 처리 가능성이 있다면 명시해 주십시오.

4.  **예상 사용자 및 주요 이해관계자 (Estimated Users and Key Stakeholders)**:
    - 이 서비스의 주요 대상 사용자는 누구입니까? (구체적인 사용자 그룹 명시)
    - 서비스 개발자, 운영자, 투자자 외에 이 서비스로 인해 직간접적으로 영향을 받을 수 있는 다른 주요 이해관계자는 누가 있습니까? (예: 콘텐츠 제작자, 특정 산업 종사자, 사회적 소수자 그룹 등)

분석 결과를 다음 JSON 형식으로 반환해 주십시오:
{{
    "target_functions": "대상 기능에 대한 상세 분석 내용...",
    "key_features": "주요 특징에 대한 상세 분석 내용...",
    "data_usage_estimation": "데이터 수집 및 활용 방식 (추정)에 대한 상세 분석 내용...",
    "estimated_users_stakeholders": "예상 사용자 및 주요 이해관계자에 대한 상세 분석 내용..."
}}
"""

WEB_SEARCH_RESULTS_SECTION_TEMPLATE = """
추가 웹 검색 정보:
{web_results}
"""