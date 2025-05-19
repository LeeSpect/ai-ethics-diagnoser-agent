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
    service_type: Literal["챗봇", "추천 알고리즘", "이미지 생성 AI", "기타"]
    service_initial_info: str

    # Service Analyzer Agent 결과
    analyzed_service_info: Optional[ServiceAnalysisOutput]

    # Risk Classifier Agent 결과
    main_risk_categories: Optional[List[str]]
    pending_specific_diagnoses: Optional[List[str]]

    # Ethics Risk Diagnoser Nodes 결과 (누적)
    common_ethics_risks: Optional[Dict[str, str]]
    specific_ethics_risks_by_category: Optional[Dict[str, Dict[str, str]]]
    past_case_analysis_results_by_category: Optional[Dict[str, List[PastCaseAnalysis]]]

    # Report Type Decider Agent 결과
    issue_found: Optional[bool]
    report_type: Optional[Literal["no_issue_report", "issue_improvement_report"]]

    # Improvement Suggester Agent 결과 (issue_found가 True일 때만 채워짐)
    improvement_recommendations: Optional[ImprovementMeasures]

    # Report Generator Agent 결과
    final_report_markdown: Optional[str]

    # 에러 처리
    error_message: Optional[str]