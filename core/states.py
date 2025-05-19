# core/states.py
from typing import TypedDict, List, Dict, Optional, Literal

class ServiceAnalysisOutput(TypedDict):
    """서비스 분석 에이전트의 출력 구조"""
    target_functions: str
    key_features: str
    data_usage_estimation: str
    estimated_users_stakeholders: str

class PastCaseAnalysis(TypedDict):
    """과거 사례 분석 결과 구조"""
    case_name: str
    cause_and_structure: str
    vulnerabilities: str
    comparison_with_target_service: str
    key_insights: str

class ImprovementMeasures(TypedDict):
    """개선 권고안 구조"""
    short_term: List[str]
    mid_long_term: List[str]
    governance_suggestions: List[str]

class ProjectState(TypedDict):
    """LangGraph 워크플로우 전체에서 사용될 상태 정의"""
    # 초기 입력
    service_name: str
    service_type: Literal["챗봇", "추천 알고리즘", "이미지 생성 AI", "기타"] # "기타" 유형 추가 (General Diagnoser용)
    service_initial_info: str # 사용자가 입력한 서비스에 대한 설명 또는 URL 등

    # Service Analyzer Agent 결과
    analyzed_service_info: Optional[ServiceAnalysisOutput]

    # Risk Classifier Agent 결과
    main_risk_categories: Optional[List[str]] # 예: ['bias_chatbot', 'privacy_chatbot']
    pending_specific_diagnoses: Optional[List[str]] # 처리해야 할 특화 진단 카테고리 큐

    # Ethics Risk Diagnoser Nodes 결과 (누적)
    common_ethics_risks: Optional[Dict[str, str]] # 공통 점검표 항목별 진단 결과
    specific_ethics_risks_by_category: Optional[Dict[str, Dict[str, str]]] # 카테고리별 심층 진단 결과
    past_case_analysis_results_by_category: Optional[Dict[str, List[PastCaseAnalysis]]] # 카테고리별 과거 사례 분석

    # Report Type Decider Agent 결과
    issue_found: Optional[bool] # 개선안이 필요한 심각한 이슈 발견 여부
    report_type: Optional[Literal["no_issue_report", "issue_improvement_report"]] # 생성할 보고서 유형

    # Improvement Suggester Agent 결과 (issue_found가 True일 때만 채워짐)
    improvement_recommendations: Optional[ImprovementMeasures]

    # Report Generator Agent 결과
    final_report_markdown: Optional[str] # 최종 생성된 보고서

    # 에러 처리
    error_message: Optional[str]