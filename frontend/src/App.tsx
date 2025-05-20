import { useState } from 'react';
import { AppBar, Toolbar, Typography, Container, Grid, TextField, Button, Card, CardContent, CircularProgress, Alert, Box, Paper } from '@mui/material';
import { createTheme, ThemeProvider, styled } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { callDiagnosisApi, type DiagnosisInput } from './api/diagnosis'; // 경로가 맞는지 확인
import ReactMarkdown from 'react-markdown';
import LogViewer from './components/LogViewer'; // LogViewer 컴포넌트 임포트

// 테마 정의 (개선된 버전)
const theme = createTheme({
  palette: {
    mode: 'light', // 'light' 또는 'dark'
    primary: {
      main: '#1976d2', // 진한 파란색 (신뢰감)
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#9c27b0', // 보라색 (창의성, 혁신)
      light: '#ba68c8',
      dark: '#7b1fa2',
    },
    background: {
      default: '#f4f6f8', // 부드러운 회색 배경
      paper: '#ffffff',
    },
    text: {
      primary: '#333333',
      secondary: '#555555',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 700,
      color: '#1976d2', // Primary color
      marginBottom: '0.5em',
    },
    h5: {
      fontWeight: 600,
      color: '#1565c0', // Darker primary
      marginTop: '1em',
      marginBottom: '0.5em',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 4px -1px rgba(0,0,0,0.2), 0 4px 5px 0 rgba(0,0,0,0.14), 0 1px 10px 0 rgba(0,0,0,0.12)', // 좀 더 부드러운 그림자
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8, // 부드러운 버튼 모서리
          textTransform: 'none', // 버튼 텍스트 대문자 변환 없음
          fontWeight: 600,
        },
        containedPrimary: {
          transition: 'transform 0.2s ease-in-out',
          '&:hover': {
            transform: 'scale(1.03)', // 호버 시 약간 확대
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12, // 카드 모서리 둥글게
          boxShadow: '0 8px 16px rgba(0,0,0,0.1)', // 카드 그림자 개선
          transition: 'transform 0.3s ease, box-shadow 0.3s ease',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: '0 12px 24px rgba(0,0,0,0.15)',
          }
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8, // 입력 필드 모서리
          },
        },
      },
    },
  },
});

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginTop: theme.spacing(3),
  marginBottom: theme.spacing(3),
  backgroundColor: theme.palette.background.paper,
}));


interface DiagnosisFormProps {
  onSubmit: (data: { serviceName: string; serviceInfo: string }) => void;
  isLoading: boolean;
}

function DiagnosisForm({ onSubmit, isLoading }: DiagnosisFormProps) {
  const [serviceName, setServiceName] = useState('');
  const [serviceInfo, setServiceInfo] = useState('');

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    onSubmit({ serviceName, serviceInfo });
  };

  return (
    <StyledPaper elevation={3}>
      <Typography variant="h5" gutterBottom component="h2">
        AI 서비스 정보 입력
      </Typography>
      <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1 }}>
        <TextField
          margin="normal"
          required
          fullWidth
          id="serviceName"
          label="서비스 이름"
          name="serviceName"
          autoComplete="serviceName"
          autoFocus
          value={serviceName}
          onChange={(e) => setServiceName(e.target.value)}
          disabled={isLoading}
        />
        <TextField
          margin="normal"
          required
          fullWidth
          name="serviceInfo"
          label="서비스 상세 설명"
          id="serviceInfo"
          multiline
          rows={6}
          value={serviceInfo}
          onChange={(e) => setServiceInfo(e.target.value)}
          disabled={isLoading}
        />
        <Button
          type="submit"
          fullWidth
          variant="contained"
          sx={{ mt: 3, mb: 2, py: 1.5, fontSize: '1.1rem' }}
          disabled={isLoading}
        >
          {isLoading ? <CircularProgress size={24} color="inherit" /> : '윤리 리스크 진단 시작'}
        </Button>
      </Box>
    </StyledPaper>
  );
}

interface DiagnosisResultCardProps {
  report: string | null;
  error: string | null;
}

function DiagnosisResultCard({ report, error }: DiagnosisResultCardProps) {
  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 3, whiteSpace: 'pre-wrap' }}>
        <Typography variant="h6">오류 발생</Typography>
        {error}
      </Alert>
    );
  }

  if (report === null) {
    return null; // 초기 상태 또는 로딩 후 결과 없을 때 (선택적 UI)
  }
  
  if (report === "") { // 빈 문자열 리포트는 "진행중" 또는 다른 메시지 표시 가능
      return (
        <StyledPaper elevation={3}>
            <Typography variant="h5" component="h2" gutterBottom>
                진단 결과 보고서
            </Typography>
            <Typography>진단 결과를 기다리고 있습니다...</Typography>
        </StyledPaper>
      )
  }


  return (
    <StyledPaper elevation={3}>
      <Typography variant="h5" component="h2" gutterBottom>
        진단 결과 보고서
      </Typography>
      <Box sx={{ 
        mt: 2, 
        p: 2, 
        border: '1px solid #e0e0e0', 
        borderRadius: '8px', 
        backgroundColor: '#f9f9f9',
        '& h1, & h2, & h3, & h4, & h5, & h6': { marginTop: '1em', marginBottom: '0.5em'},
        '& p': { marginBottom: '0.8em'},
        '& ul, & ol': { paddingLeft: '20px'},
        '& blockquote': { 
            borderLeft: `4px solid ${theme.palette.secondary.light}`, 
            paddingLeft: '15px', 
            marginLeft: 0, 
            fontStyle: 'italic',
            backgroundColor: '#f0f0f0',
            paddingY: '10px'
        },
        '& pre': {
            backgroundColor: '#2d2d2d',
            color: '#f8f8f2',
            padding: '15px',
            borderRadius: '6px',
            overflowX: 'auto'
        },
        '& code:not(pre > code)': { // 인라인 코드만 해당되도록 수정
            backgroundColor: '#e8eaf6',
            color: '#3f51b5',
            padding: '0.2em 0.4em',
            borderRadius: '3px',
        }
      }}>
        <ReactMarkdown>{report}</ReactMarkdown>
      </Box>
    </StyledPaper>
  );
}


function App() {
  const [report, setReport] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (data: { serviceName: string; serviceInfo: string }) => {
    console.log("Diagnosis submission started:", data); // API 호출 시작 로깅
    setIsLoading(true);
    setError(null);
    setReport(null); // 이전 리포트 초기화

    const inputData: DiagnosisInput = {
      service_name: data.serviceName,
      service_initial_info: data.serviceInfo,
    };

    try {
      console.log("Calling API with data:", inputData); // API 호출 직전 데이터 로깅
      const response = await callDiagnosisApi(inputData);
      console.log("API response received:", response); // API 응답 로깅

      if (response && response.report_content && typeof response.report_content === 'string') {
        setReport(response.report_content);
      } else if (response && typeof response === 'string') {
        setReport(response);
      } else {
        const errorMessage = '진단 결과를 가져왔으나, 보고서 내용을 찾을 수 없습니다. API 응답 형식을 확인해주세요.';
        console.error(errorMessage, response); // 에러 상세 로깅
        setError(errorMessage);
        // setReport(JSON.stringify(response, null, 2)); // 전체 응답을 보여주는 것은 디버깅용으로만
      }
    } catch (err) {
      let errorMessage = '알 수 없는 오류가 발생했습니다.';
      if (err instanceof Error) {
        errorMessage = err.message;
      }
      console.error("API call failed:", err); // API 호출 실패 에러 로깅
      setError(errorMessage);
      setReport(null);
    } finally {
      console.log("Diagnosis submission finished. Loading state set to false."); // API 호출 완료 로깅
      setIsLoading(false);
    }
  };


  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static" elevation={1}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
            AI 윤리성 리스크 진단 시스템
          </Typography>
        </Toolbar>
      </AppBar>
      <Container component="main" maxWidth="lg" sx={{ mt: 4, mb: 4 }}> {/* maxWidth 변경 */} 
        <Typography variant="h4" component="h1" gutterBottom align="center" sx={{ mb: 4 }}>
          AI 서비스 윤리 진단
        </Typography>
        
        <Grid container spacing={3}> {/* Grid 레이아웃 사용 */} 
          <Grid item xs={12} md={6}> {/* 진단 폼 영역 */} 
            <DiagnosisForm onSubmit={handleSubmit} isLoading={isLoading} />
          </Grid>
          <Grid item xs={12} md={6}> {/* 결과 카드 영역 */} 
            {isLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: '200px' }}>
                <CircularProgress size={50} />
              </Box>
            )}
            {!isLoading && <DiagnosisResultCard report={report} error={error} />} 
          </Grid>
          <Grid item xs={12}> {/* 로그 뷰어 전체 너비 */} 
            <LogViewer />
          </Grid>
        </Grid>

      </Container>
      <Box component="footer" sx={{ bgcolor: theme.palette.background.default, py: 3, mt: 'auto', borderTop: `1px solid ${theme.palette.divider}` }}> {/* 테마 색상 사용 */} 
        <Container maxWidth="lg">
          <Typography variant="body2" color="text.secondary" align="center">
            &copy; {new Date().getFullYear()} AI Ethics Diagnosis. All rights reserved.
          </Typography>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
