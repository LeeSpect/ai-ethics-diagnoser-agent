import axios from 'axios';

// FastAPI 서버 주소 (환경에 맞게 수정)
const API_BASE_URL = 'http://localhost:8001'; // 포트 8001로 변경

export interface DiagnosisInput {
  service_name: string;
  service_initial_info: string;
}

export interface DiagnosisResponse {
  // FastAPI 응답 구조에 맞춰 정의 (예시)
  // final_report_markdown: string;
  // 또는 전체 상태 객체를 받을 수도 있습니다.
  [key: string]: any; // 우선은 유연하게 받도록 설정
}

export const callDiagnosisApi = async (input: DiagnosisInput): Promise<DiagnosisResponse> => {
  try {
    const response = await axios.post<DiagnosisResponse>(`${API_BASE_URL}/diagnosis`, input);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      // 서버에서 보낸 에러 메시지가 있다면 사용
      throw new Error(error.response.data.detail || '진단 API 호출 중 오류가 발생했습니다.');
    } else {
      // 그 외 네트워크 오류 등
      throw new Error('진단 API 호출 중 알 수 없는 오류가 발생했습니다.');
    }
  }
};