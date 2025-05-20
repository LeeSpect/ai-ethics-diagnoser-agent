import React, { useEffect, useState, useRef } from 'react';
import { Paper, Typography, List, ListItem, ListItemText, Chip, Box } from '@mui/material';
import { styled, useTheme } from '@mui/material/styles'; // useTheme 추가

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  extra?: string;
}

function parseLogMessage(logString: string): LogEntry {
  const match = logString.match(/^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]\s+(.*?)(?:\s+\((.*?)\))?$/);
  if (match) {
    return {
      timestamp: match[1],
      level: match[2],
      message: match[3],
      extra: match[4] || '',
    };
  }
  return { timestamp: new Date().toISOString().replace('T', ' ').split('.')[0] + ',000', level: 'RAW', message: logString };
}

const LogViewerContainer = styled(Paper)(({ theme }) => ({
  marginTop: theme.spacing(3),
  padding: theme.spacing(2),
  maxHeight: '400px',
  overflowY: 'auto',
  backgroundColor: theme.palette.mode === 'dark' ? '#232323' : '#fdfdfd', // 약간 더 어두운/밝은 배경
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius,
  fontFamily: '"Fira Code", "Menlo", "Monaco", "Consolas", "Courier New", monospace',
  fontSize: '0.875rem',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
}));

const LogListItemStyled = styled(ListItem)(({ theme }) => ({
  paddingTop: theme.spacing(0.75),
  paddingBottom: theme.spacing(0.75),
  borderBottom: `1px solid ${theme.palette.divider}`,
  '&:last-child': {
    borderBottom: 'none',
  },
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  }
}));

function getChipColorByLevel(level: string): "default" | "primary" | "secondary" | "error" | "info" | "success" | "warning" {
  switch (level.toUpperCase()) {
    case 'DEBUG': return 'default';
    case 'INFO': return 'info';
    case 'WARNING': return 'warning';
    case 'ERROR': return 'error';
    case 'CRITICAL': return 'error';
    default: return 'default';
  }
}

const LogViewer: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const ws = useRef<WebSocket | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const theme = useTheme(); // 테마 사용

  useEffect(() => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.hostname}:8001/ws/logs`;

    ws.current = new WebSocket(wsUrl);

    const connectTime = new Date().toLocaleString();

    ws.current.onopen = () => {
      console.log('Log WebSocket Connected');
      setLogs(prevLogs => [
        ...prevLogs,
        parseLogMessage(`${connectTime} [INFO] Log stream connected to ${wsUrl}`)
      ]);
    };

    ws.current.onmessage = (event) => {
      const rawLog = event.data as string;
      setLogs(prevLogs => {
        const newLogs = [...prevLogs, parseLogMessage(rawLog)];
        return newLogs.slice(-200);
      });
    };

    ws.current.onerror = (error) => {
      console.error('Log WebSocket Error:', error);
      setLogs(prevLogs => [
        ...prevLogs,
        parseLogMessage(`${new Date().toLocaleString()} [ERROR] Log stream connection error.`)
      ]);
    };

    ws.current.onclose = (event) => {
      console.log('Log WebSocket Disconnected:', event.reason, event.code);
      setLogs(prevLogs => [
        ...prevLogs,
        parseLogMessage(`${new Date().toLocaleString()} [INFO] Log stream disconnected. Code: ${event.code}`)
      ]);
    };

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <Box mt={4} mb={2}>
      <Typography variant="h5" gutterBottom sx={{ color: theme.palette.primary.dark, fontWeight: 'bold' }}>
        서버 활동 로그 (실시간)
      </Typography>
      <LogViewerContainer ref={logContainerRef} elevation={2}>
        {logs.length === 0 ? (
          <Typography sx={{ p: 3, textAlign: 'center', color: theme.palette.text.secondary }}>
            서버 로그를 기다리고 있습니다...
          </Typography>
        ) : (
          <List dense disablePadding>
            {logs.map((log, index) => (
              <LogListItemStyled key={index}>
                <Chip 
                  label={log.level}
                  size="small" 
                  color={getChipColorByLevel(log.level)}
                  sx={{ 
                    marginRight: 1.5, 
                    minWidth: 75, 
                    fontWeight: 'bold',
                    fontSize: '0.75rem'
                  }} 
                />
                <ListItemText 
                  primary={log.message}
                  secondary={`${log.timestamp}${log.extra ? ` - ${log.extra}` : ''}`}
                  primaryTypographyProps={{ 
                    sx: { 
                      wordBreak: 'break-word', 
                      fontSize: 'inherit',
                      color: log.level === 'ERROR' || log.level === 'CRITICAL' ? theme.palette.error.main : theme.palette.text.primary,
                    }
                  }}
                  secondaryTypographyProps={{ 
                    sx: { 
                      fontSize: '0.7rem', 
                      opacity: 0.7,
                      marginTop: '2px' 
                    }
                  }}
                />
              </LogListItemStyled>
            ))}
          </List>
        )}
      </LogViewerContainer>
    </Box>
  );
};

export default LogViewer; 