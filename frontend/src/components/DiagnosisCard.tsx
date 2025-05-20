import React from 'react';
import { Card, CardContent, Typography, Box, styled } from '@mui/material';
import { skTheme } from '../theme';

const StyledCard = styled(Card)({
  borderRadius: '12px',
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
  transition: 'transform 0.3s ease, box-shadow 0.3s ease',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)',
  },
});

const TitleBox = styled(Box)({
  borderBottom: `2px solid ${skTheme.palette.primary.main}`,
  paddingBottom: '0.5rem',
  marginBottom: '1rem',
});

interface DiagnosisCardProps {
  title: string;
  content: string;
}

export const DiagnosisCard: React.FC<DiagnosisCardProps> = ({ title, content }) => {
  return (
    <StyledCard>
      <CardContent>
        <TitleBox>
          <Typography variant="h6" color="primary">
            {title}
          </Typography>
        </TitleBox>
        <Typography variant="body1">
          {content}
        </Typography>
      </CardContent>
    </StyledCard>
  );
};