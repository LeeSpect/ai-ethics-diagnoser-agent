import React from 'react';
import { styled } from '@mui/material/styles';
import { AppBar, Toolbar, Typography, Container } from '@mui/material';
import { skTheme } from '../theme';

const StyledAppBar = styled(AppBar)({
  backgroundColor: skTheme.palette.primary.main,
  boxShadow: 'none',
  padding: '0.5rem 0',
});

const Logo = styled('div')({
  display: 'flex',
  alignItems: 'center',
  '& img': {
    height: '36px',
    marginRight: '16px',
  },
});

const MainContainer = styled(Container)({
  paddingTop: '2rem',
  paddingBottom: '2rem',
});

interface MainLayoutProps {
  children: React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  return (
    <>
      <StyledAppBar position="static">
        <Toolbar>
          <Logo>
            <Typography variant="h6" color="inherit">
              AI 윤리 진단 시스템
            </Typography>
          </Logo>
        </Toolbar>
      </StyledAppBar>
      <MainContainer maxWidth="lg">
        {children}
      </MainContainer>
    </>
  );
};