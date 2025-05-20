import React, { useState } from 'react';
import { Box, TextField, Button, Typography, styled } from '@mui/material';
import { skTheme } from '../theme';

const FormContainer = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  gap: '1.5rem',
  padding: '2rem',
  backgroundColor: skTheme.palette.background.paper,
  borderRadius: '12px',
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
});

const SubmitButton = styled(Button)({
  backgroundColor: skTheme.palette.primary.main,
  '&:hover': {
    backgroundColor: skTheme.palette.primary.dark,
  },
});

interface DiagnosisFormProps {
  onSubmit: (data: { serviceName: string; serviceInfo: string }) => void;
}

export const DiagnosisForm: React.FC<DiagnosisFormProps> = ({ onSubmit }) => {
  const [serviceName, setServiceName] = useState('');
  const [serviceInfo, setServiceInfo] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ serviceName, serviceInfo });
  };

  return (
    <form onSubmit={handleSubmit}>
      <FormContainer>
        <Typography variant="h5" color="primary">
          AI 서비스 윤리성 진단
        </Typography>
        <TextField
          label="서비스 이름"
          variant="outlined"
          fullWidth
          value={serviceName}
          onChange={(e) => setServiceName(e.target.value)}
          required
        />
        <TextField
          label="서비스 설명"
          variant="outlined"
          fullWidth
          multiline
          rows={4}
          value={serviceInfo}
          onChange={(e) => setServiceInfo(e.target.value)}
          required
        />
        <SubmitButton
          type="submit"
          variant="contained"
          size="large"
          fullWidth
        >
          진단 시작
        </SubmitButton>
      </FormContainer>
    </form>
  );
};