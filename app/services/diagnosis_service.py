from typing import Dict, Any
from fastapi import HTTPException
import datetime
from core.graph import app as langgraph_app
from core.states import ProjectState


class DiagnosisService:
    @staticmethod
    async def run_diagnosis(service_name: str, service_info: str) -> Dict[str, Any]:
        """
        AI 서비스 윤리성 진단을 실행하고 결과를 반환합니다.
        
        Args:
            service_name: 서비스 이름
            service_info: 서비스 초기 정보
            
        Returns:
            진단 결과 리포트 딕셔너리
        """
        try:
            # ProjectState 초기화
            initial_state: ProjectState = {
                "service_name": service_name,
                "service_type": "기타",
                "service_initial_info": service_info,
                "analyzed_service_info": None,
                "main_risk_categories": None,
                "pending_specific_diagnoses": None,
                "common_ethics_risks": None,
                "specific_ethics_risks_by_category": None,
                "past_case_analysis_results_by_category": None,
                "issue_found": None,
                "report_type": None,
                "improvement_recommendations": None,
                "final_report_markdown": None,
                "error_message": None,
                "current_date": datetime.date.today().isoformat()
            }
            
            # LangGraph 워크플로우 실행
            final_state = await langgraph_app.ainvoke(initial_state)
            
            if final_state.get("error_message"):
                raise HTTPException(status_code=500, detail=final_state["error_message"])
                
            return {
                "report_type": final_state.get("report_type"),
                "report_content": final_state.get("final_report_markdown"),
                "improvement_recommendations": final_state.get("improvement_recommendations")
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"진단 과정 중 오류 발생: {str(e)}"
            )