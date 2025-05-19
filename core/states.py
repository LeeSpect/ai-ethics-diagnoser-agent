from typing import TypedDict, List, Dict, Optional, Literal

class ServiceAnalysisOutput(TypedDict, total=False): # total=False로 하여 일부 키가 없을 수도 있음
    target_functions: str
    key_features: str
    data_usage_estimation: str
    estimated_users_stakeholders: str

class PastCaseAnalysis(TypedDict, total=False):
    case_name: str
    cause_and_structure: str
    vulnerabilities: str
    comparison_with_target_service: str
    key_insights: str

class ImprovementMeasures(TypedDict, total=False):
    short_term: List[str]
    mid_long_term: List[str]
    governance_suggestions: List[str]

class ProjectState(TypedDict):
    # 초기 입력
    service_name: str
    service_type: Literal["챗봇", "추천 알고리즘", "이미지 생성 AI", "기타"] # "기타" 유형 (General 진단용)
    service_initial_info: str

    # 분석 및 진단 결과
    analyzed_service_info: Optional[ServiceAnalysisOutput]
    main_risk_categories: Optional[List[str]] # Risk Classifier가 식별한 주요 리스크 (예: "bias_chatbot")
    pending_specific_diagnoses: Optional[List[str]] # 처리해야 할 특화 진단 카테고리 큐
    
    common_ethics_risks: Optional[Dict[str, str]] # 공통 진단 결과 (자율점검표 항목: 설명)
    # 특화 진단 결과 (카테고리별로 저장)
    specific_ethics_risks_by_category: Optional[Dict[str, Dict[str, str]]] 
    past_case_analysis_results_by_category: Optional[Dict[str, List[PastCaseAnalysis]]]

    # 개선안 및 보고서 관련
    issue_found: Optional[bool] # Report Type Decider가 설정
    report_type: Optional[Literal["no_issue_report", "issue_improvement_report"]] # Report Type Decider가 설정
    improvement_recommendations: Optional[ImprovementMeasures] # Improvement Suggester가 설정 (issue_found가 True일 때)
    
    final_report_markdown: Optional[str] # 최종 보고서
    error_message: Optional[str] # 에러 발생 시 메시지