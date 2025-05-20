import { createTheme } from '@mui/material/styles';

// SK 그룹의 브랜드 컬러 시스템
const skColors = {
  primary: '#E4002B', // SK 시그니처 레드
  secondary: '#2D2D2D', // SK 다크 그레이
  background: '#F5F5F5', // SK 라이트 그레이
  textPrimary: '#212121',
  textSecondary: '#757575',
  white: '#FFFFFF',
};

// Material-UI 테마 생성
export const skTheme = createTheme({
  palette: {
    primary: {
      main: skColors.primary,
      contrastText: skColors.white,
    },
    secondary: {
      main: skColors.secondary,
      contrastText: skColors.white,
    },
    background: {
      default: skColors.background,
      paper: skColors.white,
    },
    text: {
      primary: skColors.textPrimary,
      secondary: skColors.textSecondary,
    },
  },
  typography: {
    fontFamily: [
      'Noto Sans KR',
      'Roboto',
      'sans-serif',
    ].join(','),
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
    },
    h3: {
      fontWeight: 500,
      fontSize: '1.75rem',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          padding: '12px 24px',
          fontWeight: 600,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
        },
      },
    },
  },
});