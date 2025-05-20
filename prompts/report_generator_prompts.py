# 프롬프트 1: 문제가 있어서 개선안이 도출되었을 때 (issue_improvement_report)
ISSUE_IMPROVEMENT_REPORT_TEMPLATE = """
보고서 생성 요청: AI 윤리성 리스크 진단 보고서 (개선안 포함)

**1. 보고서 기본 정보:**
- 보고서 제목: AI 윤리성 리스크 진단 보고서: {service_name}
- 작성일: {current_date}
- 진단 기관: {analysis_agency}

**2. SUMMARY (보고서 전체 내용을 5줄 이내로 요약):**
{summary_placeholder}

**3. 진단 개요:**
- 진단 대상 AI 서비스: {service_name} (서비스 유형: {service_type})
- 진단 목적: {service_name}의 AI 윤리적 리스크를 "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)"에 근거하여 분석하고, 과거 유사 사례 분석을 통해 개선 권고안을 도출하여 안전하고 신뢰할 수 있는 AI 서비스 구현에 기여함을 목적으로 합니다.
- 평가 기준: "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)" (이하 '자율점검표')

**4. AI 서비스 분석 결과:**
- 대상 기능: {target_functions}
- 주요 특징: {key_features}
- 데이터 수집/활용 (추정): {data_usage_estimation}
- 예상 사용자 및 이해관계자: {estimated_users_stakeholders}

**5. 주요 윤리적 리스크 진단 결과:**
{risk_assessment_details_placeholder}
  (예시:
  - 편향성: 학습 데이터에서의 {bias_source}로 인해 {specific_bias_risk}이 발견되었습니다. 이는 자율점검표 항목 E03.02(데이터 편향 가능성 판단 및 최소화 노력) 위배 가능성이 있습니다.
  - 프라이버시: {privacy_concern_details}와 같은 프라이버시 침해 우려가 식별되었습니다. 자율점검표 항목 E02.01(개인정보보호 자율점검표 점검 수행) 및 E02.02(사생활 침해 우려 검토)에 대한 추가 점검이 필요합니다.)
- 식별된 주요 취약점:
  {identified_vulnerabilities_placeholder}

**6. 과거 AI 윤리 문제 사례 심층 분석 및 인사이트:**
- 분석 대상 과거 사례: {past_case_names_placeholder} (예: 사례1명, 사례2명)
{past_cases_analysis_details_placeholder}
  (예시:
  - 사례 1 ({past_case_name_1}):
    - 원인: {past_case_1_cause}
    - 구조: {past_case_1_structure}
    - 취약점: {past_case_1_vulnerability}
  - 사례 2 ({past_case_name_2}):
    - 원인: {past_case_2_cause}
    - 구조: {past_case_2_structure}
    - 취약점: {past_case_2_vulnerability})
- 진단 대상 서비스({service_name})와의 비교 분석 및 시사점:
  {comparison_insights_placeholder}
- 도출된 주요 인사이트:
  {key_takeaways_placeholder}

**7. 개선 권고안:**
- 단기적 개선 방안:
{short_term_recommendations_placeholder}
- 중장기적 개선 방향:
{mid_long_term_recommendations_placeholder}
- 윤리적 AI 거버넌스 제안:
{governance_suggestions_placeholder}

**8. 결론:**
{conclusion_placeholder}

**[보고서 작성 지침]**
- 제공된 `참고context`의 "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)"의 관련 항목(예: E01.01)을 문맥에 맞게 명시적으로 인용하여 주십시오.
- 과거 사례 분석은 구체적인 원인, 문제 구조, 기술적/정책적 취약점을 명확히 기술하고, 진단 대상 서비스에 적용 가능한 교훈과 인사이트를 논리적으로 도출해 주십시오.
- 개선 권고안은 실행 가능성을 고려하여 구체적으로 제시하고, 필요한 경우 관련 자율점검표 항목을 참조로 언급해 주십시오.
- 전체적으로 전문적이고 객관적인 어조를 유지하며, 명확하고 간결하게 작성해 주십시오.

**참고context**
{context}
"""

# 프롬프트 2: 문제가 없을 때 (no_issue_report)
NO_ISSUE_REPORT_TEMPLATE = """
보고서 생성 요청: AI 윤리성 리스크 진단 보고서 (특이사항 없음)

**1. 보고서 기본 정보:**
- 보고서 제목: AI 윤리성 리스크 진단 보고서: {service_name}
- 작성일: {current_date}
- 진단 기관: {analysis_agency}

**2. SUMMARY (보고서 전체 내용을 5줄 이내로 요약):**
{summary_placeholder_no_issue}

**3. 진단 개요:**
- 진단 대상 AI 서비스: {service_name} (서비스 유형: {service_type})
- 진단 목적: {service_name}의 AI 윤리적 리스크를 "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)"에 근거하여 분석하고, 서비스의 윤리적 운영 상태를 점검하여 안전하고 신뢰할 수 있는 AI 서비스 구현에 기여함을 목적으로 합니다.
- 평가 기준: "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)" (이하 '자율점검표')

**4. AI 서비스 분석 결과:**
- 대상 기능: {target_functions}
- 주요 특징: {key_features}
- 데이터 수집/활용 (추정): {data_usage_estimation}
- 예상 사용자 및 이해관계자: {estimated_users_stakeholders}

**5. 주요 윤리적 리스크 진단 결과:**
{risk_assessment_details_no_issue_placeholder}
- 종합 평가: 전반적으로 "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)"의 주요 항목들을 충족하거나, 식별된 리스크가 낮은 수준으로 관리되고 있는 것으로 평가됩니다. 즉각적인 개선이 필요한 심각한 윤리적 문제는 발견되지 않았습니다.

**6. 과거 AI 윤리 문제 사례 검토 및 시사점 (참고용):**
- 검토 대상 과거 사례: {past_case_names_placeholder}
{past_cases_analysis_details_no_issue_placeholder}
- 현재 서비스 운영에 대한 참고 시사점:
  {implications_for_current_service_placeholder}

**7. 결론:**
{conclusion_no_issue_placeholder}
  (예: {service_name} 서비스에 대한 AI 윤리성 리스크 진단 결과, 현재 시점에서 "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)"의 기준에 비추어 볼 때, 특별히 개선이 요구되는 중대한 윤리적 문제는 발견되지 않았습니다. 향후에도 지속적인 윤리적 관점을 견지하며 서비스를 운영해 나갈 것을 권장합니다.)

**[보고서 작성 지침]**
- 제공된 `참고context`의 "2025 인공지능 윤리기준 실천을 위한 자율점검표(안)"의 관련 항목(예: E01.01)을 문맥에 맞게 명시적으로 인용하여 서비스가 해당 기준을 어떻게 충족하고 있는지 설명해 주십시오.
- 과거 사례 검토는 현재 서비스에 직접적인 문제가 없음을 전제로, 참고 및 예방적 차원에서 간략히 기술하고 시사점을 도출해 주십시오.
- 전체적으로 전문적이고 객관적인 어조를 유지하며, 명확하고 간결하게 작성해 주십시오.

**참고context**
{context}
"""