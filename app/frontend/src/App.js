import React, { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import {
    Container,
    Box,
    Tabs,
    Tab,
    Typography,
    Paper,
    Button,
    IconButton,
    TextField,
    Alert,
    CircularProgress,
    Grid,
    Card,
    CardContent,
    Chip,
    Divider,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    LinearProgress,
    Slider,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    List,
    ListItem,
    ListItemText,
    ThemeProvider,
    createTheme,
    Backdrop,
    Fade,
    Checkbox,
    FormControlLabel
} from '@mui/material';
import {
    Add as AddIcon,
    Delete as DeleteIcon,
    Upload as UploadIcon,
    PlayArrow as PlayIcon,
    Stop as StopIcon,
    Download as DownloadIcon,
    Refresh as RefreshIcon,
    ExpandMore as ExpandMoreIcon,
    CloudDownload as CloudDownloadIcon,
    Close as CloseIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import ReactPlayer from './react-player-config';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#3a6fec',
            light: '#3a6fec',
            dark: '#3a6fec',
            contrastText: '#ffffff',
        },
        secondary: {
            main: '#9198A1',
            light: '#C9D1D9',
            dark: '#6E7681',
            contrastText: '#ffffff',
        },
        background: {
            default: '#0D1117',
            paper: '#161B22',
        },
        text: {
            primary: '#E6EDF3',
            secondary: '#9198A1',
        },
        divider: '#30363D',
        error: {
            main: '#DC5145',
        },
        warning: {
            main: '#EB8B3A',
        },
        success: {
            main: '#3A6FEC',
        },
    },
    typography: {
        fontFamily: [
            'Helvetica Neue',
            'Helvetica',
            'Arial',
            'sans-serif'
        ].join(','),
        h1: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 300,
        },
        h2: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 300,
        },
        h3: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 400,
        },
        h4: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 400,
        },
        h5: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 500,
        },
        h6: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 500,
        },
        body1: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 400,
        },
        body2: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 400,
        },
        button: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 500,
        },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                body: {
                    background: 'radial-gradient(ellipse at center, #0D1117 0%, #1C2128 50%, #0A0D10 100%)',
                    minHeight: '100vh',
                },
                '*::-webkit-scrollbar': {
                    width: '8px',
                    height: '8px',
                },
                '*::-webkit-scrollbar-track': {
                    background: '#30363D',
                    borderRadius: '4px',
                },
                '*::-webkit-scrollbar-thumb': {
                    background: '#484F58',
                    borderRadius: '4px',
                    '&:hover': {
                        background: '#6E7681',
                    },
                },
                '*::-webkit-scrollbar-corner': {
                    background: '#30363D',
                },
                '*': {
                    scrollbarWidth: 'thin',
                    scrollbarColor: '#484F58 #30363D',
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundColor: '#161B22',
                    backgroundImage: 'none',
                    border: '1px solid #30363D',
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: '#161B22',
                    backgroundImage: 'none',
                    border: '1px solid #30363D',
                    '&:hover': {
                        borderColor: '#484F58',
                        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4), 0 4px 12px rgba(0, 0, 0, 0.2)',
                    },
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    borderRadius: '8px',
                    fontWeight: 500,
                },
                contained: {
                    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.3)',
                    '&:hover': {
                        boxShadow: '0 2px 6px rgba(0, 0, 0, 0.4)',
                    },
                },
                outlined: {
                    borderColor: '#30363D',
                    '&:hover': {
                        borderColor: '#3a6fec',
                        backgroundColor: 'rgba(255, 107, 53, 0.08)',
                    },
                },
            },
        },
        MuiTextField: {
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        backgroundColor: '#0D1117',
                        '& fieldset': {
                            borderColor: '#30363D',
                        },
                        '&:hover fieldset': {
                            borderColor: '#6E7681',
                        },
                        '&.Mui-focused fieldset': {
                            borderColor: '#3a6fec',
                        },
                    },
                },
            },
        },
        MuiSelect: {
            styleOverrides: {
                root: {
                    backgroundColor: '#0D1117',
                    '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#30363D',
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#6E7681',
                    },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#3a6fec',
                    },
                },
            },
        },
        MuiMenuItem: {
            styleOverrides: {
                root: {
                    backgroundColor: '#161B22',
                    '&:hover': {
                        backgroundColor: '#21262D',
                    },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(53, 100, 255, 0.12)',
                        '&:hover': {
                            backgroundColor: 'rgba(53, 124, 255, 0.2)',
                        },
                    },
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    backgroundColor: '#21262D',
                    color: '#E6EDF3',
                    '&.MuiChip-colorPrimary': {
                        backgroundColor: 'rgba(53, 134, 255, 0.2)',
                        color: '#3a6fec',
                    },
                },
                outlined: {
                    borderColor: '#30363D',
                    '&.MuiChip-colorPrimary': {
                        borderColor: '#3a6fec',
                        color: '#3a6fec'
                    },
                },
            },
        },
        MuiAccordion: {
            styleOverrides: {
                root: {
                    backgroundColor: '#161B22',
                    border: '1px solid #30363D',
                    '&:before': {
                        display: 'none',
                    },
                    '&.Mui-expanded': {
                        margin: 0,
                    },
                },
            },
        },
        MuiAccordionSummary: {
            styleOverrides: {
                root: {
                    backgroundColor: '#21262D',
                    '&:hover': {
                        backgroundColor: '#262C36',
                    },
                },
            },
        },
        MuiDialog: {
            styleOverrides: {
                paper: {
                    backgroundColor: '#161B22',
                    border: '1px solid #30363D',
                    borderRadius: 12,
                    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
                },
            },
        },
        MuiDialogTitle: {
            styleOverrides: {
                root: {
                    backgroundColor: '#21262D',
                    borderBottom: '1px solid #30363D',
                    color: '#F0F6FC',
                    fontWeight: 600,
                    fontSize: '1.25rem',
                },
            },
        },
        MuiDialogContent: {
            styleOverrides: {
                root: {
                    backgroundColor: '#161B22',
                    color: '#C9D1D9',
                },
            },
        },
        MuiDialogActions: {
            styleOverrides: {
                root: {
                    backgroundColor: '#161B22',
                    borderTop: '1px solid #30363D',
                    padding: '16px 24px',
                },
            },
        },
        MuiListItem: {
            styleOverrides: {
                root: {
                    '&:hover': {
                        backgroundColor: '#21262D',
                    },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(53, 147, 255, 0.12)',
                        '&:hover': {
                            backgroundColor: 'rgba(53, 124, 255, 0.2)',
                        },
                    },
                },
            },
        },
        MuiCheckbox: {
            styleOverrides: {
                root: {
                    color: '#6E7681',
                    '&.Mui-checked': {
                        color: '#3a6fec',
                    },
                    '&:hover': {
                        backgroundColor: 'rgba(58, 111, 236, 0.08)',
                    },
                },
            },
        },
        MuiFormControlLabel: {
            styleOverrides: {
                label: {
                    color: '#C9D1D9',
                    fontSize: '0.875rem',
                },
            },
        },
        MuiSlider: {
            styleOverrides: {
                root: {
                    color: '#3a6fec',
                },
                rail: {
                    backgroundColor: '#30363D',
                },
                track: {
                    backgroundColor: '#3a6fec',
                },
                thumb: {
                    backgroundColor: '#3a6fec',
                    '&:hover': {
                        boxShadow: '0 0 0 8px rgba(53, 134, 255, 0.16)',
                    },
                },
            },
        },
        MuiLinearProgress: {
            styleOverrides: {
                root: {
                    backgroundColor: '#30363D',
                },
                bar: {
                    backgroundColor: '#3a6fec',
                },
            },
        },
        MuiCircularProgress: {
            styleOverrides: {
                root: {
                    color: '#3a6fec',
                },
            },
        },
        MuiTabs: {
            styleOverrides: {
                root: {
                    '& .MuiTabs-indicator': {
                        backgroundColor: '#3a6fec',
                    },
                },
            },
        },
        MuiTab: {
            styleOverrides: {
                root: {
                    color: '#9198A1',
                    '&.Mui-selected': {
                        color: '#3a6fec',
                    },
                    '&:hover': {
                        color: '#E6EDF3',
                    },
                },
            },
        },
        MuiBackdrop: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                },
            },
        },
        MuiDivider: {
            styleOverrides: {
                root: {
                    borderColor: '#30363D',
                },
            },
        },
        MuiIconButton: {
            styleOverrides: {
                root: {
                    color: '#9198A1',
                    '&:hover': {
                        backgroundColor: 'rgba(255, 107, 53, 0.08)',
                        color: '#3a6fec',
                    },
                },
            },
        },
        MuiContainer: {
            styleOverrides: {
                root: {
                    backgroundColor: 'transparent',
                    background: 'transparent',
                },
            },
        },
    },
});

function TabPanel({ children, value, index, ...other }) {
    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`simple-tabpanel-${index}`}
            aria-labelledby={`simple-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={{
                    p: 2,
                    background: 'linear-gradient(135deg, #161B22 0%, #0D1117 100%)',
                    flex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    minHeight: 0,
                    overflow: 'auto',
                    maxHeight: 'calc(100vh - 200px)',
                    '&::-webkit-scrollbar': {
                        width: '8px',
                    },
                    '&::-webkit-scrollbar-track': {
                        background: '#30363D',
                        borderRadius: '4px',
                    },
                    '&::-webkit-scrollbar-thumb': {
                        background: '#484F58',
                        borderRadius: '4px',
                        '&:hover': {
                            background: '#6E7681',
                        },
                    },
                }}>
                    {children}
                </Box>
            )}
        </div>
    );
}

function AudioUploadRow({ index, data, onChange, onRemove }) {
    const [audioFile, setAudioFile] = useState(null);
    const [audioUrl, setAudioUrl] = useState('');

    useEffect(() => {
        if (!data.file && !data.audioUrl) {
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
            setAudioFile(null);
            setAudioUrl('');
        }
    }, [data.file, data.audioUrl, audioUrl]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        accept: {
            'audio/*': ['.mp3', '.wav', '.flac', '.m4a', '.aac']
        },
        multiple: false,
        onDrop: (acceptedFiles) => {
            const file = acceptedFiles[0];
            setAudioFile(file);
            setAudioUrl(URL.createObjectURL(file));
            onChange(index, { ...data, file, audioUrl: URL.createObjectURL(file) });
        }
    });

    return (
        <Card sx={{
            mb: 2,
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06)',
            borderRadius: 2,
            border: 'none',
            '&:hover': {
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15), 0 4px 12px rgba(0, 0, 0, 0.1)',
                transform: 'translateY(-2px)',
                transition: 'all 0.3s ease'
            },
            transition: 'all 0.3s ease'
        }}>
            <CardContent>
                <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} sm={4}>
                        <Box
                            {...getRootProps()}
                            sx={{
                                border: '2px dashed #30363D',
                                borderRadius: 2,
                                p: 2,
                                textAlign: 'center',
                                cursor: 'pointer',
                                backgroundColor: '#0D1117',
                                '&:hover': { 
                                    borderColor: 'primary.main',
                                    backgroundColor: '#161B22'
                                },
                                backgroundColor: isDragActive ? 'action.hover' : 'background.paper'
                            }}
                        >
                            <input {...getInputProps()} />
                            {audioFile ? (
                                <Box>
                                    <Typography variant="body2" color="textSecondary">
                                        {audioFile.name}
                                    </Typography>
                                    {audioUrl && (
                                        <Suspense fallback={<div>Loading player...</div>}>
                                            <ReactPlayer
                                                url={audioUrl}
                                                controls
                                                width="100%"
                                                height="60px"
                                                style={{ marginTop: 8 }}
                                            />
                                        </Suspense>
                                    )}
                                </Box>
                            ) : (
                                <Box>
                                    <UploadIcon sx={{ fontSize: 40, color: 'text.secondary' }} />
                                    <Typography variant="body2" color="textSecondary">
                                        {isDragActive ? 'Drop audio here' : 'Click or drag audio file'}
                                    </Typography>
                                </Box>
                            )}
                        </Box>
                    </Grid>

                    <Grid item xs={12} sm={6}>
                        <TextField
                            fullWidth
                            multiline
                            rows={3}
                            label={`Prompt/Annotation ${index + 1}`}
                            placeholder="Describe this audio file..."
                            value={data.prompt || ''}
                            onChange={(e) => onChange(index, { ...data, prompt: e.target.value })}
                            variant="outlined"
                        />
                    </Grid>

                    <Grid item xs={12} sm={2}>
                        <IconButton
                            color="error"
                            onClick={() => onRemove(index)}
                            sx={{ alignSelf: 'flex-start' }}
                        >
                            <DeleteIcon />
                        </IconButton>
                    </Grid>
                </Grid>
            </CardContent>
        </Card>
    );
}

function formatDuration(seconds) {
    const sec = Math.floor(seconds % 60);
    const min = Math.floor((seconds / 60) % 60);
    const hr = Math.floor(seconds / 3600);
    return [hr, min, sec]
        .map((v, i) => (i === 0 ? v : v.toString().padStart(2, '0')))
        .join(':');
}

function TrainingMonitor({
    isTraining,
    trainingProgress,
    trainingStatus,
    trainingHistory,
    trainingStartTime,
    trainingError,
    trainingConfig,
    systemStatus
}) {
    const getElapsedTime = () => {
        if (!trainingStartTime) return 0;
        return Math.floor((Date.now() - trainingStartTime) / 1000);
    };

    const getEstimatedTimeRemaining = () => {
        if (!trainingStartTime || trainingProgress === 0) return null;
        const elapsed = getElapsedTime();
        const estimatedTotal = (elapsed / trainingProgress) * 100;
        return Math.max(0, estimatedTotal - elapsed);
    };

    const getProgressColor = () => {
        if (trainingError) return 'error';
        if (trainingProgress === 100) return 'success';
        return 'primary';
    };

    return (
        <Paper sx={{
            p: 3,
            mb: 2,
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4), 0 4px 12px rgba(0, 0, 0, 0.3)',
            borderRadius: 2,
            border: '1px solid #30363D',
            background: 'linear-gradient(135deg, #161B22 0%, #21262D 100%)',
            '&:hover': {
                boxShadow: '0 12px 32px rgba(0, 0, 0, 0.5), 0 6px 16px rgba(0, 0, 0, 0.4)',
                transform: 'translateY(-2px)',
                transition: 'all 0.3s ease'
            },
            transition: 'all 0.3s ease'
        }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box
                    sx={{
                        width: 12,
                        height: 12,
                        borderRadius: '50%',
                        backgroundColor: isTraining ? 'success.main' : trainingError ? 'error.main' : 'grey.500',
                        mr: 1,
                        animation: isTraining ? 'pulse 2s infinite' : 'none',
                        '@keyframes pulse': {
                            '0%': { opacity: 1 },
                            '50%': { opacity: 0.5 },
                            '100%': { opacity: 1 }
                        }
                    }}
                />
                <Typography variant="h6" sx={{ flex: 1 }}>
                    Training Monitor
                </Typography>
                {isTraining && (
                    <Chip
                        label="Live"
                        size="small"
                        color="success"
                        sx={{ fontSize: '0.7rem' }}
                    />
                )}
            </Box>

            <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Progress</Typography>
                    <Typography variant="body2">{trainingProgress}%</Typography>
                </Box>
                <LinearProgress
                    variant="determinate"
                    value={trainingProgress}
                    color={getProgressColor()}
                    sx={{ height: 8, borderRadius: 4 }}
                />
            </Box>

            {trainingStatus?.device_info && (
                <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                        <strong>Device Used for Training</strong>
                    </Typography>
                    <Typography variant="body2">
                        Device: {trainingStatus.device_info.device} ({trainingStatus.device_info.memory_gb?.toFixed(2)}GB VRAM)
                    </Typography>
                    <Typography variant="body2" color="textSecondary" sx={{ fontSize: '0.8rem', mt: 0.5 }}>
                        Info: {trainingStatus.device_info.type === 'cuda' ? 'CUDA GPU available and selected for training' :
                            trainingStatus.device_info.type === 'cpu' ? 'Using CPU (no CUDA GPU available or compatible)' :
                                'Using MPS (Apple Silicon GPU)'}
                    </Typography>
                </Box>
            )}

            <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Current Epoch</Typography>
                    <Typography variant="body1">
                        {trainingStatus?.current_epoch || 0} / {trainingConfig.epochs}
                    </Typography>
                </Grid>
                <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Current Step / Total Steps</Typography>
                    <Typography variant="body1" color="primary">
                        {trainingStatus?.current_step !== undefined ?
                            `${trainingStatus.current_step} / ${trainingStatus?.current_step !== undefined && trainingStatus?.current_epoch !== undefined ?
                                (trainingStatus.current_epoch * (trainingStatus.total_steps_per_epoch || 35) + trainingStatus.current_step) :
                                trainingStatus.current_step}` :
                            'N/A'}
                    </Typography>
                </Grid>
                <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Checkpoints Saved</Typography>
                    <Typography variant="body1">
                        {trainingStatus?.checkpoints_saved || 0}
                    </Typography>
                </Grid>
                <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Current Loss</Typography>
                    <Typography variant="body1">
                        {trainingStatus?.loss ? parseFloat(trainingStatus.loss).toFixed(4) : 'N/A'}
                    </Typography>
                </Grid>

            </Grid>

            {trainingStatus?.loss_history && trainingStatus.loss_history.length > 0 && (
                <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                        <strong>Loss History</strong>
                    </Typography>
                    <Box sx={{ height: 200, width: '100%' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={trainingStatus.loss_history}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis
                                    dataKey="time"
                                    tickFormatter={(value) => `${Math.floor(value / 60)}:${(value % 60).toString().padStart(2, '0')}`}
                                    label={{ value: 'Time (min:sec)', position: 'insideBottom', offset: -5 }}
                                />
                                <YAxis
                                    label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                                />
                                <Tooltip
                                    labelFormatter={(value) => `Time: ${Math.floor(value / 60)}:${(value % 60).toString().padStart(2, '0')}`}
                                    formatter={(value, name) => [value.toFixed(4), 'Loss']}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="loss"
                                    stroke="#8884d8"
                                    strokeWidth={2}
                                    dot={{ r: 2 }}
                                    activeDot={{ r: 4 }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </Box>
                </Box>
            )}

            {trainingError && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                        <strong>Training Error:</strong> {trainingError}
                    </Typography>
                </Alert>
            )}

        </Paper>
    );
}

function ModelUnwrapButton({ model, onUnwrap, onRefresh }) {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleUnwrap = async () => {
        setLoading(true);
        setResult(null);
        setError(null);

        try {
            const response = await axios.post('/api/unwrap-model', {
                model_config: model.configPath,
                ckpt_path: model.ckptPath,
                name: model.name + '_unwrapped'
            });
            setResult(response.data);
            if (onUnwrap) onUnwrap(response.data);
            if (onRefresh) onRefresh(); // Refresh model list after unwrapping
        } catch (err) {
            console.error('Unwrap error:', err);
            setError(err.response?.data?.error || err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ marginTop: 8 }}>
            <Button
                variant="outlined"
                color="primary"
                size="small"
                startIcon={<CloudDownloadIcon />}
                onClick={handleUnwrap}
                disabled={loading}
            >
                {loading ? 'Unwrapping...' : 'Unwrap for Inference'}
            </Button>
            {result && result.unwrapped_path && (
                <div style={{ marginTop: 4 }}>
                    <a href={`file://${result.unwrapped_path}`} target="_blank" rel="noopener noreferrer">
                        Download Unwrapped Model
                    </a>
                </div>
            )}
            {error && (
                <div style={{ color: '#DB5044', marginTop: 4 }}>{error}</div>
            )}
        </div>
    );
}

function CheckpointManager({ model, onRefresh }) {
    const [loadingStates, setLoadingStates] = useState({});
    const [error, setError] = useState(null);
    const [expandedCheckpoint, setExpandedCheckpoint] = useState(null);

    const handleUnwrapCheckpoint = async (checkpoint) => {
        const checkpointId = checkpoint.path;
        setLoadingStates(prev => ({ ...prev, [checkpointId]: { unwrapping: true } }));
        setError(null);
        try {
            const response = await axios.post('/api/unwrap-model', {
                model_config: model.config_path,
                ckpt_path: checkpoint.path,
                name: `${checkpoint.name}_unwrapped`
            });
            setError(null);
            alert(`Checkpoint "${checkpoint.name}" unwrapped successfully!`);
            onRefresh();
        } catch (err) {
            setError(`Failed to unwrap ${checkpoint.name}: ${err.response?.data?.error || err.message}`);
        } finally {
            setLoadingStates(prev => ({ ...prev, [checkpointId]: { unwrapping: false } }));
        }
    };

    const handleDeleteCheckpoint = async (checkpoint) => {
        if (!confirm(`Are you sure you want to delete the wrapped checkpoint "${checkpoint.name}"? This action cannot be undone.`)) {
            return;
        }
        const checkpointId = checkpoint.path;
        setLoadingStates(prev => ({ ...prev, [checkpointId]: { deleting: true } }));
        setError(null);
        try {
            await axios.post('/api/delete-checkpoint', {
                checkpoint_path: checkpoint.path
            });
            alert(`Checkpoint "${checkpoint.name}" deleted successfully.`);
            onRefresh();
        } catch (err) {
            setError(`Failed to delete ${checkpoint.name}: ${err.response?.data?.error || err.message}`);
        } finally {
            setLoadingStates(prev => ({ ...prev, [checkpointId]: { deleting: false } }));
        }
    };

    const checkpoints = model.checkpoints || [];

    return (
        <Paper sx={{
            p: 2,
            mb: 2,
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4), 0 4px 12px rgba(0, 0, 0, 0.3)',
            borderRadius: 2,
            border: '1px solid #30363D',
            background: 'linear-gradient(135deg, #161B22 0%, #21262D 100%)',
            '&:hover': {
                boxShadow: '0 12px 32px rgba(0, 0, 0, 0.5), 0 6px 16px rgba(0, 0, 0, 0.4)',
                transform: 'translateY(-2px)',
                transition: 'all 0.3s ease'
            },
            transition: 'all 0.3s ease'
        }}>
            <Typography variant="h6" gutterBottom>
                Checkpoint Management for {model.name}
            </Typography>

            {checkpoints.length === 0 ? (
                <Typography variant="body2" color="textSecondary" gutterBottom>
                    No checkpoints found for this model.
                </Typography>
            ) : (
                <>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                        <strong>Available Checkpoints:</strong> {checkpoints.length}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                        <strong>Unwrapped Models:</strong> {model.unwrapped_models?.length || 0}
                    </Typography>

                    {/* Individual Checkpoint Cards */}
                    <Box sx={{ mt: 2 }}>
                        {checkpoints.map((checkpoint, index) => {
                            const checkpointId = checkpoint.path;
                            const isUnwrapping = loadingStates[checkpointId]?.unwrapping;
                            const isDeleting = loadingStates[checkpointId]?.deleting;

                            const hasUnwrappedVersion = model.unwrapped_models?.some(unwrapped =>
                                unwrapped.name.includes(checkpoint.name) ||
                                checkpoint.name.includes(unwrapped.name.replace('_unwrapped', ''))
                            );

                            return (
                                <Card key={index} sx={{
                                    mb: 1,
                                    p: 1,
                                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04)',
                                    borderRadius: 1.5,
                                    border: 'none',
                                    '&:hover': {
                                        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.12), 0 2px 6px rgba(0, 0, 0, 0.08)',
                                        transform: 'translateY(-1px)',
                                        transition: 'all 0.2s ease'
                                    },
                                    transition: 'all 0.2s ease'
                                }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                        <Box sx={{ flex: 1 }}>
                                            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                                {checkpoint.name}
                                                {hasUnwrappedVersion && (
                                                    <Chip
                                                        label="Unwrapped"
                                                        size="small"
                                                        color="success"
                                                        sx={{ ml: 1, fontSize: '0.7rem' }}
                                                    />
                                                )}
                                            </Typography>
                                            <Typography variant="caption" color="textSecondary">
                                                Size: {checkpoint.size_mb} MB
                                            </Typography>
                                            {checkpoint.epoch !== undefined && (
                                                <Typography variant="caption" color="textSecondary" sx={{ ml: 1 }}>
                                                    | Epoch: {checkpoint.epoch}
                                                </Typography>
                                            )}
                                            {checkpoint.step !== undefined && (
                                                <Typography variant="caption" color="textSecondary" sx={{ ml: 1 }}>
                                                    | Step: {checkpoint.step}
                                                </Typography>
                                            )}
                                        </Box>
                                        <Box sx={{ display: 'flex', gap: 1 }}>
                                            {!hasUnwrappedVersion && (
                                                <Button
                                                    variant="outlined"
                                                    color="primary"
                                                    size="small"
                                                    startIcon={<CloudDownloadIcon />}
                                                    onClick={() => handleUnwrapCheckpoint(checkpoint)}
                                                    disabled={isUnwrapping || isDeleting}
                                                >
                                                    {isUnwrapping ? 'Unwrapping...' : 'Unwrap'}
                                                </Button>
                                            )}

                                            {hasUnwrappedVersion && (
                                                <Button
                                                    variant="outlined"
                                                    color="error"
                                                    size="small"
                                                    startIcon={<DeleteIcon />}
                                                    onClick={() => handleDeleteCheckpoint(checkpoint)}
                                                    disabled={isDeleting}
                                                >
                                                    {isDeleting ? 'Deleting Wrapped...' : 'Delete Wrapped Checkpoint'}
                                                </Button>
                                            )}
                                        </Box>
                                    </Box>
                                </Card>
                            );
                        })}
                    </Box>
                </>
            )}

            {error && (
                <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>
            )}
        </Paper>
    );
}

function GeneratedFragmentsWindow({ fragments, onDownload }) {
    const [playingFragment, setPlayingFragment] = useState(null);
    const audioRefs = useRef({});

    const handlePlayPause = (fragment) => {
        const audio = audioRefs.current[fragment.id];
        if (!audio) return;

        if (playingFragment === fragment.id) {
            audio.pause();
            setPlayingFragment(null);
        } else {
            if (playingFragment && audioRefs.current[playingFragment]) {
                audioRefs.current[playingFragment].pause();
            }
            audio.play();
            setPlayingFragment(fragment.id);
        }
    };

    const setAudioRef = useCallback((fragmentId, audioElement) => {
        if (audioElement) {
            audioRefs.current[fragmentId] = audioElement;
        }
    }, []);

    return (
        <Paper
            variant="outlined"
            sx={{
                p: 2,
                height: 240,
                display: 'flex',
                flexDirection: 'column'
            }}
        >
            <Typography variant="h6" gutterBottom>
                Generated Fragments ({fragments.length})
            </Typography>

            {fragments.length === 0 ? (
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        height: '100%',
                        color: 'text.secondary'
                    }}
                >
                    <Typography variant="body2">
                        No fragments generated yet
                    </Typography>
                </Box>
            ) : (
                <List
                    sx={{
                        flex: 1,
                        overflow: 'auto',
                        maxHeight: 180,
                        '& .MuiListItem-root': {
                            border: '1px solid',
                            borderColor: 'divider',
                            borderRadius: 1,
                            mb: 1,
                            '&:last-child': {
                                mb: 0
                            }
                        }
                    }}
                >
                    {fragments.slice().reverse().map((fragment, index) => (
                        <ListItem
                            key={fragment.id}
                            sx={{
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'stretch',
                                py: 1
                            }}
                        >
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                                <Box sx={{ flex: 1, minWidth: 0 }}>
                                    <Typography
                                        variant="subtitle2"
                                        sx={{
                                            fontWeight: 'bold',
                                            overflow: 'hidden',
                                            textOverflow: 'ellipsis',
                                            display: '-webkit-box',
                                            WebkitLineClamp: 2,
                                            WebkitBoxOrient: 'vertical'
                                        }}
                                    >
                                        {fragment.prompt}
                                    </Typography>
                                    <Typography variant="caption" color="textSecondary">
                                        {fragment.duration}s • {fragment.timestamp}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', gap: 1, flexShrink: 0 }}>
                                    <IconButton
                                        size="small"
                                        onClick={() => handlePlayPause(fragment)}
                                        color={playingFragment === fragment.id ? "primary" : "default"}
                                        sx={{
                                            border: '1px solid',
                                            borderColor: playingFragment === fragment.id ? 'primary.main' : 'divider'
                                        }}
                                    >
                                        {playingFragment === fragment.id ? <StopIcon /> : <PlayIcon />}
                                    </IconButton>
                                    <Button
                                        size="small"
                                        variant="outlined"
                                        startIcon={<DownloadIcon />}
                                        onClick={() => onDownload(fragment)}
                                    >
                                        Download
                                    </Button>
                                </Box>
                            </Box>

                            <audio
                                ref={el => setAudioRef(fragment.id, el)}
                                src={fragment.audioUrl}
                                onEnded={() => setPlayingFragment(null)}
                                onPause={() => setPlayingFragment(null)}
                                style={{ display: 'none' }}
                            />
                        </ListItem>
                    ))}
                </List>
            )}
        </Paper>
    );
}

function WelcomePage({ open, onClose }) {
    const [titleVisible, setTitleVisible] = useState(false);
    const [textVisible, setTextVisible] = useState(false);

    useEffect(() => {
        if (open) {
            const titleTimer = setTimeout(() => {
                setTitleVisible(true);
            }, 500);

            const textTimer = setTimeout(() => {
                setTextVisible(true);
            }, 1500);

            return () => {
                clearTimeout(titleTimer);
                clearTimeout(textTimer);
            };
        } else {
            setTitleVisible(false);
            setTextVisible(false);
        }
    }, [open]);

    if (!open) return null;

    return (
        <Backdrop
            open={open}
            onClick={onClose}
            sx={{
                zIndex: 9999,
                
                background: 'linear-gradient(135deg, #0D1117 0%, #1C2128 100%)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'fixed',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: '85vw',
                height: '90%',
                cursor: 'pointer',
                borderRadius: '15px',
                overflow: 'hidden'
            }}
            >

            <Box
                sx={{ textAlign: 'center', maxWidth: 800, px: 4 }}
                onClick={(e) => e.stopPropagation()}
                
            >
                <Fade in={titleVisible} timeout={800}>
                    <Box sx={{
                        width: 120,
                        height: 120,
                        backgroundImage: 'url(/fragmenta_icon_1024.png)',
                        backgroundSize: 'cover',
                        backgroundPosition: 'center',
                        borderRadius: 3,                     
                        filter: 'drop-shadow(0 8px 16px rgba(0, 0, 0, 0.4))',
                        mx: 'auto',
                        mb: 1
                    }} />
                </Fade>

                <Fade in={titleVisible} timeout={1000}>
                    <Typography
                        variant="h2"
                        component="h1"
                        sx={{
                            fontFamily: '"Bitcount Single", "IBM Plex Mono", "JetBrains Mono", "Space Mono", "Courier New", monospace',
                            fontWeight: 400,
                            color: 'primary.main',
                            mb: 4,
                            fontSize: { xs: '2.5rem', sm: '3.5rem', md: '4rem' },
                            letterSpacing: '0.02em'
                        }}
                    >
                        Welcome to Fragmenta!
                    </Typography>
                </Fade>

                <Fade in={textVisible} timeout={1000}>
                    <Box>
                        <Typography
                            variant="h5"
                            sx={{
                                color: 'text.secondary',
                                mb: 1,
                                fontWeight: 400,
                                lineHeight: 1.6
                            }}
                        >
                            An End-to-End Pipeline to Fine-Tune and Use Text-to-Audio Models.
                        </Typography>


                                                <Typography
                            variant="body1"
                            sx={{
                                color: 'text.secondary',
                                mb: 8,
                                lineHeight: 1.8,
                                fontSize: '1.1rem'
                            }}
                        >

                            Made for composers and audio creators.

                        </Typography>
                        <Typography
                            variant="body2"
                            sx={{
                                color: 'text.secondary',
                                opacity: 0.6,
                                fontSize: '0.8rem',
                            }}
                        >
                            @2025 Misagh Azimi
                        </Typography>
                        <Typography
                            variant="body2"
                            sx={{
                                color: 'text.secondary',
                                opacity: 0.6,
                                fontSize: '0.8rem',
                                fontStyle: 'italic',
                            }}
                        >
                            Version 0.0.1
                        </Typography>
                        <Button
                            variant="contained"
                            onClick={onClose}
                            sx={{
                                mt: 4,
                                mb: 2,
                                px: 4,
                                py: 1.5,
                                borderRadius: 2,
                                textTransform: 'none',
                                fontSize: '1.1rem',
                                fontWeight: 500
                            }}
                        >
                            Get Started
                        </Button>
                        <Typography
                            variant="body2"
                            sx={{
                                color: 'text.secondary',
                                opacity: 0.7,
                                fontSize: '0.9rem'
                            }}
                        >
                            or click anywhere to continue
                        </Typography>
                    </Box>
                </Fade>
            </Box>
        </Backdrop>
    );
}

function App() {
    const [tabValue, setTabValue] = useState(0);
    const [uploadRows, setUploadRows] = useState([
        { file: null, prompt: '', audioUrl: '' }
    ]);
    const [processingStatus, setProcessingStatus] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [processedCount, setProcessedCount] = useState(0);
    const [chunksPreview, setChunksPreview] = useState([]);

    const [showWelcomePage, setShowWelcomePage] = useState(true);

    const [trainingConfig, setTrainingConfig] = useState({
        epochs: 50,
        checkpointSteps: 100,
        batchSize: 4,
        learningRate: 1e-4,
        modelName: 'my_fine_tuned_model',
        baseModel: 'stable-audio-open-small',
        saveWrappedCheckpoint: false
    });
    const [isTraining, setIsTraining] = useState(false);
    const [trainingProgress, setTrainingProgress] = useState(0);
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [trainingHistory, setTrainingHistory] = useState([]);
    const [trainingStartTime, setTrainingStartTime] = useState(null);
    const [trainingError, setTrainingError] = useState(null);

    const [generationPrompt, setGenerationPrompt] = useState('');
    const [generationDuration, setGenerationDuration] = useState(10);
    const [generatedAudio, setGeneratedAudio] = useState(null);
    const [generatedAudioBlob, setGeneratedAudioBlob] = useState(null);
    const [isGenerating, setIsGenerating] = useState(false);
    const [generationProgress, setGenerationProgress] = useState(0);
    const [selectedModel, setSelectedModel] = useState('');
    const [selectedUnwrappedModel, setSelectedUnwrappedModel] = useState('');
    const [outputCounter, setOutputCounter] = useState(0);
    const [generatedFragments, setGeneratedFragments] = useState([]);

    const generateFileName = () => {
        return `fragmenta_output${outputCounter.toString().padStart(3, '0')}.wav`;
    };

    const downloadAudio = () => {
        if (generatedAudioBlob) {
            const url = URL.createObjectURL(generatedAudioBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = generateFileName();
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }
    };

    const downloadFragment = (fragment) => {
        const link = document.createElement('a');
        link.href = fragment.audioUrl;
        link.download = fragment.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const [systemStatus, setSystemStatus] = useState(null);
    const [availableModels, setAvailableModels] = useState([]);
    const [gpuMemoryStatus, setGpuMemoryStatus] = useState(null);
    const [isUpdatingGpuMemory, setIsUpdatingGpuMemory] = useState(false);
    const [baseModels, setBaseModels] = useState([
        {
            name: 'stable-audio-open-small',
            displayName: 'Stable Audio Open Small (Recommended)',
            description: 'Faster - Lower memory usage',
            type: 'base',
            path: '/models/pretrained/stable-audio-open-small-model.safetensors',
            configPath: '/models/config/model_config_small.json',
            downloaded: false
        },
        {
            name: 'stable-audio-open-1.0',
            displayName: 'Stable Audio Open 1.0',
            description: 'Higher quality - Requires more memory',
            type: 'base',
            path: '/models/pretrained/stable-audio-open-model.safetensors',
            configPath: '/models/config/model_config.json',
            downloaded: false
        }
    ]);

    const [showStartFreshDialog, setShowStartFreshDialog] = useState(false);
    const [isStartingFresh, setIsStartingFresh] = useState(false);
    const [uploadKey, setUploadKey] = useState(0);
    const [isFreeingGPU, setIsFreeingGPU] = useState(false);
    const [showFreeGPUDialog, setShowFreeGPUDialog] = useState(false);

    useEffect(() => {
        setSelectedUnwrappedModel('');
    }, [selectedModel]);

    useEffect(() => {
        console.log('Model changed:', selectedModel);
    }, [selectedModel]);

    const getMaxDuration = () => {
        if (!selectedModel) return 10;

        const baseModel = baseModels.find(m => m.name === selectedModel);
        if (baseModel) {
            if (baseModel.name === 'stable-audio-open-small') {
                return 11;
            } else if (baseModel.name === 'stable-audio-open-1.0') {
                return 47;
            }
        }

        const model = availableModels.find(m => m.name === selectedModel);
        if (model && selectedUnwrappedModel) {
            const selectedUnwrapped = model.unwrapped_models?.find(u => u.path === selectedUnwrappedModel);
            if (selectedUnwrapped) {
                const sizeMB = selectedUnwrapped.size_mb || 0;
                return sizeMB < 2000 ? 11 : 47;
            }
        }

        return 10;
    };

    useEffect(() => {
        const maxDuration = getMaxDuration();
        if (generationDuration > maxDuration) {
            setGenerationDuration(maxDuration);
        }
    }, [selectedModel, selectedUnwrappedModel]);

    const handleTabChange = (event, newValue) => {
        setTabValue(newValue);
    };

    const addUploadRow = () => {
        setUploadRows([...uploadRows, { file: null, prompt: '', audioUrl: '' }]);
    };

    const removeUploadRow = (index) => {
        const newRows = uploadRows.filter((_, i) => i !== index);
        setUploadRows(newRows);
    };

    const updateUploadRow = (index, data) => {
        const newRows = [...uploadRows];
        newRows[index] = data;
        setUploadRows(newRows);
    };

    const fetchSystemStatus = async () => {
        try {
            const response = await axios.get('/api/status');
            setSystemStatus(response.data);
        } catch (error) {
            console.error('Error fetching system status:', error);
        }
    };

    const fetchAvailableModels = async () => {
        try {
            const response = await axios.get('/api/models');
            console.log('Fetched models:', response.data.models);
            setAvailableModels(response.data.models || []);
        } catch (error) {
            console.error('Error fetching available models:', error);
        }
    };

    const fetchBaseModelsStatus = async () => {
        try {
            const response = await axios.get('/api/base-models/status');
            const baseModelsStatus = response.data.base_models;
            
            setBaseModels(prevModels => 
                prevModels.map(model => ({
                    ...model,
                    downloaded: baseModelsStatus[model.name]?.downloaded || false
                }))
            );
        } catch (error) {
            console.error('Error fetching base models status:', error);
        }
    };

    const refreshAllModels = async () => {
        await Promise.all([
            fetchAvailableModels(),
            fetchBaseModelsStatus()
        ]);
    };

    const fetchGpuMemoryStatus = async () => {
        try {
            setIsUpdatingGpuMemory(true);
            const response = await axios.get('/api/gpu-memory-status');
            console.log('GPU Memory Response:', response.data);
            setGpuMemoryStatus(response.data.memory_info);
        } catch (error) {
            console.error('Error fetching GPU memory status:', error.response?.data?.error || error.message || error);
            setGpuMemoryStatus(null);
        } finally {
            setIsUpdatingGpuMemory(false);
        }
    };

    useEffect(() => {
        fetchSystemStatus();
        fetchAvailableModels();
        fetchBaseModelsStatus();
        fetchGpuMemoryStatus();
    }, []);

    useEffect(() => {
        const interval = setInterval(() => {
            fetchGpuMemoryStatus();
        }, isTraining ? 2000 : 10000);

        return () => clearInterval(interval);
    }, [isTraining]);

    useEffect(() => {
        let statusInterval;

        if (isTraining) {
            statusInterval = setInterval(async () => {
                try {
                    const statusResponse = await axios.get('/api/training-status');
                    const currentStatus = statusResponse.data;
                    setTrainingStatus(currentStatus);

                    if (currentStatus.progress !== undefined) {
                        setTrainingProgress(prevProgress => {
                            if (currentStatus.progress >= prevProgress && (prevProgress > 0 || currentStatus.progress > 0)) {
                                return currentStatus.progress;
                            }
                            return prevProgress;
                        });
                    }

                    setTrainingHistory(prev => {
                        const newEntry = {
                            timestamp: Date.now(),
                            progress: currentStatus.progress || 0,
                            current_epoch: currentStatus.current_epoch || 0,
                            current_step: currentStatus.current_step || 0,
                            loss: currentStatus.loss,
                            checkpoints_saved: currentStatus.checkpoints_saved || 0,
                            is_training: currentStatus.is_training,
                            message: currentStatus.error ||
                                (currentStatus.progress > 0 ? `Progress: ${currentStatus.progress}%` : 'Starting...')
                        };

                        const lastEntry = prev[prev.length - 1];
                        if (!lastEntry ||
                            lastEntry.progress !== newEntry.progress ||
                            lastEntry.current_epoch !== newEntry.current_epoch ||
                            lastEntry.current_step !== newEntry.current_step ||
                            lastEntry.loss !== newEntry.loss ||
                            lastEntry.checkpoints_saved !== newEntry.checkpoints_saved ||
                            lastEntry.message !== newEntry.message) {
                            return [...prev, newEntry];
                        }
                        return prev;
                    });

                    if (currentStatus.is_training) {
                        setTrainingProgress(currentStatus.progress || 0);
                    } else {
                        setIsTraining(false);
                        if (currentStatus.error) {
                            setTrainingError(currentStatus.error);
                            setProcessingStatus(`Training failed: ${currentStatus.error}`);
                        } else {
                            setProcessingStatus('Training completed successfully!');
                            setTrainingProgress(100);
                        }
                        setTimeout(() => {
                            fetchSystemStatus();
                            fetchAvailableModels();
                        }, 0);
                    }
                } catch (statusError) {
                    console.error('Error fetching training status:', statusError);
                    setTrainingError('Failed to fetch training status');
                }
            }, 2000);
        }

        return () => {
            if (statusInterval) {
                clearInterval(statusInterval);
            }
        };
    }, [isTraining]);

    const processFiles = async () => {
        setIsProcessing(true);
        setProcessingStatus('Processing files...');

        try {
            const formData = new FormData();

            uploadRows.forEach((row, index) => {
                if (row.file && row.prompt) {
                    formData.append(`file_${index}`, row.file);
                    formData.append(`prompt_${index}`, row.prompt);
                }
            });

            const response = await axios.post('/api/process-files', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            setProcessingStatus(response.data.message);
            setProcessedCount(response.data.processed_count);
            setChunksPreview(response.data.chunks_preview || []);

            setUploadRows([{ file: null, prompt: '', audioUrl: '' }]);

            fetchSystemStatus();
        } catch (error) {
            setProcessingStatus(`Error: ${error.response?.data?.error || error.message}`);
        } finally {
            setIsProcessing(false);
        }
    };

    const startTraining = async () => {
        setIsTraining(true);
        setTrainingProgress(0);
        setTrainingError(null);
        setTrainingStartTime(Date.now());
        setTrainingHistory([]);

        try {
            const response = await axios.post('/api/start-training', trainingConfig);
            setProcessingStatus('Training started successfully!');
        } catch (error) {
            const errorData = error.response?.data;
            const errorMessage = errorData?.error || error.message;
            
            if (errorData?.checkpoint_warning) {
                setTrainingError(errorMessage);
                setProcessingStatus(errorMessage);
            } else {
                setTrainingError(errorMessage);
                setProcessingStatus(`Training error: ${errorMessage}`);
            }
            setIsTraining(false);
        }
    };

    const stopTraining = async () => {
        try {
            const response = await axios.post('/api/stop-training');
            setProcessingStatus('Training stopped gracefully');
            setIsTraining(false);
            setTrainingProgress(0);
            setTrainingError(null);
        } catch (error) {
            setTrainingError(error.response?.data?.error || error.message);
            setProcessingStatus(`Stop training error: ${error.response?.data?.error || error.message}`);
        }
    };

    const generateAudio = async () => {
        if (!generationPrompt.trim()) {
            setProcessingStatus('Please enter a prompt');
            return;
        }

        let requestData = {
            prompt: generationPrompt,
            duration: generationDuration
        };

        console.log('=== FRONTEND DEBUG: MODEL SELECTION ===');
        console.log('selectedModel:', selectedModel);
        console.log('selectedUnwrappedModel:', selectedUnwrappedModel);
        console.log('baseModels:', baseModels);
        console.log('availableModels:', availableModels);

        const baseModel = baseModels.find(m => m.name === selectedModel);
        if (baseModel) {
            requestData.model_name = selectedModel;
            console.log('FRONTEND: Using base model:', selectedModel);
            console.log('FRONTEND: Base model details:', baseModel);
        } else if (selectedUnwrappedModel) {
            requestData.unwrapped_model_path = selectedUnwrappedModel;
            console.log('FRONTEND: Using unwrapped model:', selectedUnwrappedModel);

            const parentModel = availableModels.find(m => m.name === selectedModel);
            console.log('FRONTEND: Parent model info:', parentModel);
        } else {
            console.log('FRONTEND: No model selected!');
            setProcessingStatus('Please select a model');
            return;
        }

        console.log('FRONTEND: Final request data:', requestData);

        setIsGenerating(true);
        setGenerationProgress(0);
        setProcessingStatus('Starting audio generation...');

        const progressInterval = setInterval(() => {
            setGenerationProgress(prev => {
                if (prev >= 90) return prev;
                const newProgress = prev + Math.random() * 10;
                setProcessingStatus(`Generating audio... ${Math.round(newProgress)}%`);
                return newProgress;
            });
        }, 500);

        try {
            console.log('FRONTEND: Sending request to /api/generate with data:', requestData);
            const response = await axios.post('/api/generate', requestData, {
                responseType: 'blob'
            });

            clearInterval(progressInterval);
            setGenerationProgress(100);

            const audioUrl = URL.createObjectURL(response.data);
            setGeneratedAudio(audioUrl);
            setGeneratedAudioBlob(response.data);

            const newFragment = {
                id: Date.now(),
                prompt: generationPrompt,
                duration: generationDuration,
                audioUrl: audioUrl,
                audioBlob: response.data,
                filename: generateFileName(),
                timestamp: new Date().toLocaleString()
            };

            setGeneratedFragments(prev => [...prev, newFragment]);

            setOutputCounter(prev => prev + 1);
            setProcessingStatus('Audio generated successfully!');

            setTimeout(() => {
                setGenerationProgress(0);
            }, 2000);

        } catch (error) {
            clearInterval(progressInterval);
            setGenerationProgress(0);
            console.log('FRONTEND: Generation error:', error);
            console.log('FRONTEND: Error response:', error.response);
            setProcessingStatus(`Generation error: ${error.response?.data?.error || error.message}`);
        } finally {
            setIsGenerating(false);
        }
    };

    const handleStartFresh = async () => {
        setIsStartingFresh(true);
        setShowStartFreshDialog(false);

        try {
            const response = await axios.post('/api/start-fresh');

            setUploadRows([{ file: null, prompt: '', audioUrl: '' }]);
            setProcessedCount(0);
            setChunksPreview([]);
            setGeneratedAudio(null);
            setGeneratedAudioBlob(null);
            setGeneratedFragments([]);
            setProcessingStatus('');
            setGenerationPrompt('');
            setUploadKey(prev => prev + 1);

            setProcessingStatus(response.data.message);

            fetchSystemStatus();

        } catch (error) {
            setProcessingStatus(`Start fresh error: ${error.response?.data?.error || error.message}`);
        } finally {
            setIsStartingFresh(false);
        }
    };

    const handleFreeGPUMemory = async () => {
        setIsFreeingGPU(true);
        setShowFreeGPUDialog(false);
        try {
            const response = await axios.post('/api/free-gpu-memory');
            setProcessingStatus(`GPU Memory Freed: ${response.data.message}`);

            if (response.data.memory_info && response.data.memory_info.cuda) {
                const mem = response.data.memory_info.cuda;
                setProcessingStatus(`GPU Memory Freed: ${mem.free.toFixed(2)}GB free of ${mem.total.toFixed(2)}GB total`);
            }

            fetchGpuMemoryStatus();
        } catch (error) {
            setProcessingStatus(`Free GPU Memory error: ${error.response?.data?.error || error.message}`);
        } finally {
            setIsFreeingGPU(false);
        }
    };

    const getSelectedModelDisplayName = () => {
        console.log('=== GETTING DISPLAY NAME ===');
        console.log('selectedModel:', selectedModel);
        console.log('selectedUnwrappedModel:', selectedUnwrappedModel);

        if (!selectedModel) {
            console.log('No selectedModel, returning empty string');
            return '';
        }

        const baseModel = baseModels.find(m => m.name === selectedModel);
        if (baseModel) {
            console.log('Found base model:', baseModel.displayName);
            return baseModel.displayName;
        }

        const model = availableModels.find(m => m.name === selectedModel);
        if (model && selectedUnwrappedModel) {
            const selectedUnwrapped = model.unwrapped_models?.find(u => u.path === selectedUnwrappedModel);
            if (selectedUnwrapped) {
                const displayName = `${model.name} (${selectedUnwrapped.name})`;
                console.log('Generated fine-tuned display name:', displayName);
                return displayName;
            }
        }

        console.log('Using fallback name:', selectedModel);
        return selectedModel;
    };

    const allAvailableModels = [
        ...baseModels,
        ...availableModels
    ];

    const handleModelChange = (event) => {
        const newSelectedModel = event.target.value;
        setSelectedModel(newSelectedModel);

        setSelectedUnwrappedModel('');
    };

    return (
        <ThemeProvider theme={theme}>
            <Box sx={{
                minHeight: '100vh',
                background: 'transparent',
                backgroundColor: '#0D1117',
                overflow: 'auto',
                position: 'relative',
                display: 'flex',
                flexDirection: 'column'
            }}>
                <WelcomePage
                    open={showWelcomePage}
                    onClose={() => {
                        setShowWelcomePage(false);
                        
                        axios.post('http://127.0.0.1:5001/api/welcome-page-closed')
                            .then(() => {
                                console.log('Welcome page closure signal sent successfully');
                            })
                            .catch((error) => {
                                console.error('Failed to signal welcome page closure:', error);
                            });
                    }}
                />

            <Container maxWidth={false} sx={{
                py: 2,
                px: 2,
                minHeight: '100vh',
                display: 'flex',
                flexDirection: 'column',
                backgroundColor: 'transparent',
                background: 'transparent',
                borderBottomLeftRadius: '15px',
                borderBottomRightRadius: '15px',
                overflow: 'visible',
                boxSizing: 'border-box',
                width: '100%',
                maxWidth: '100%',
                filter: showWelcomePage ? 'blur(8px)' : 'none',
                transition: 'filter 0.3s ease-in-out'
            }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                    <Box sx={{ 
                        position: 'relative',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 2
                    }}>
                        {/* Logo */}
                        <Box sx={{
                            width: 60,
                            height: 60,
                            backgroundImage: 'url(/fragmenta_icon_1024.png)',
                            backgroundSize: 'cover',
                            backgroundPosition: 'center',
                            borderRadius: 2,
                            filter: 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3))'
                        }} />
                        
                        {/* Title */}
                        <Box>
                            <Typography variant="h4" component="h1" sx={{ 
                                color: 'text.primary',
                                fontFamily: '"Bitcount Single", "IBM Plex Mono", "JetBrains Mono", "Space Mono", "Courier New", monospace',
                                fontWeight: 400,
                                letterSpacing: '0.02em',
                                textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)'
                            }}>
                                Fragmenta
                            </Typography>
                        </Box>
                    </Box>

                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                        {/* Action Buttons - Left Side */}
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                            <Button
                                variant="contained"
                                color="primary"
                                size="small"
                                startIcon={<RefreshIcon />}
                                onClick={() => setShowFreeGPUDialog(true)}
                                disabled={isFreeingGPU || !(gpuMemoryStatus && gpuMemoryStatus.cuda)}
                                sx={{
                                    fontSize: '0.65rem',
                                    py: 0.25,
                                    px: 1,
                                    minWidth: 90,
                                    height: 28,
                                    opacity: !(gpuMemoryStatus && gpuMemoryStatus.cuda) ? 0.5 : 1
                                }}
                            >
                                {isFreeingGPU ? 'Freeing...' : 'Free GPU'}
                            </Button>
                            <Button
                                variant="contained"
                                color="error"
                                size="small"
                                startIcon={<RefreshIcon />}
                                onClick={() => setShowStartFreshDialog(true)}
                                disabled={isStartingFresh}
                                sx={{
                                    fontSize: '0.65rem',
                                    py: 0.25,
                                    px: 1,
                                    minWidth: 90,
                                    height: 28
                                }}
                            >
                                {isStartingFresh ? 'Starting...' : 'Fresh Start'}
                            </Button>
                        </Box>

                        {/* GPU Memory Status - Right Side */}
                        <Box sx={{
                            p: 1.5,
                            bgcolor: 'background.paper',
                            borderRadius: 2,
                            border: '1px solid',
                            borderColor: 'divider',
                            minWidth: 240,
                            position: 'relative',
                            overflow: 'hidden'
                        }}>
                            {gpuMemoryStatus && gpuMemoryStatus.cuda ? (
                                <>
                                    {/* Status Indicator */}
                                    <Box sx={{
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        right: 0,
                                        height: 3,
                                        bgcolor: gpuMemoryStatus.cuda.free > 2 ? 'success.main' :
                                            gpuMemoryStatus.cuda.free > 0.5 ? 'warning.main' : 'error.main'
                                    }} />

                                    {/* Header */}
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                                        <Typography variant="caption" color="textSecondary" sx={{ fontWeight: 500 }}>
                                            GPU Memory
                                        </Typography>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                            <Box sx={{
                                                width: 6,
                                                height: 6,
                                                borderRadius: '50%',
                                                bgcolor: gpuMemoryStatus.cuda.free > 2 ? 'success.main' :
                                                    gpuMemoryStatus.cuda.free > 0.5 ? 'warning.main' : 'error.main',
                                                animation: 'pulse 2s infinite',
                                                '@keyframes pulse': {
                                                    '0%': { opacity: 1 },
                                                    '50%': { opacity: 0.5 },
                                                    '100%': { opacity: 1 }
                                                }
                                            }} />
                                            <Typography variant="caption" color="textSecondary">
                                                {gpuMemoryStatus.cuda.free > 2 ? 'Good' :
                                                    gpuMemoryStatus.cuda.free > 0.5 ? 'Low' : 'Critical'}
                                            </Typography>
                                        </Box>
                                    </Box>

                                    {/* Memory Bar */}
                                    <Box sx={{ mb: 1 }}>
                                        <Box sx={{
                                            position: 'relative',
                                            width: '100%',
                                            height: 6,
                                            bgcolor: 'grey.200',
                                            borderRadius: 3,
                                            overflow: 'hidden'
                                        }}>
                                            {/* Used Memory */}
                                            <Box
                                                sx={{
                                                    position: 'absolute',
                                                    top: 0,
                                                    left: 0,
                                                    height: '100%',
                                                    width: `${Math.min((gpuMemoryStatus.cuda.allocated / gpuMemoryStatus.cuda.total) * 100, 100)}%`,
                                                    bgcolor: 'error.main',
                                                    borderRadius: 3,
                                                    transition: 'width 0.3s ease-in-out'
                                                }}
                                            />
                                            {/* Cached Memory */}
                                            <Box
                                                sx={{
                                                    position: 'absolute',
                                                    top: 0,
                                                    left: 0,
                                                    height: '100%',
                                                    width: `${Math.min(((gpuMemoryStatus.cuda.allocated + gpuMemoryStatus.cuda.cached) / gpuMemoryStatus.cuda.total) * 100, 100)}%`,
                                                    bgcolor: 'warning.main',
                                                    borderRadius: 3,
                                                    transition: 'width 0.3s ease-in-out'
                                                }}
                                            />
                                        </Box>
                                    </Box>

                                    {/* Memory Details */}
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <Typography variant="caption" color="primary" sx={{ fontWeight: 'bold' }}>
                                            {gpuMemoryStatus.cuda.free.toFixed(1)}GB free
                                        </Typography>
                                        <Typography variant="caption" color="textSecondary">
                                            {gpuMemoryStatus.cuda.total.toFixed(1)}GB total
                                        </Typography>
                                    </Box>
                                </>
                            ) : (
                                <>
                                    {/* Status Indicator - No GPU */}
                                    <Box sx={{
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        right: 0,
                                        height: 3,
                                        bgcolor: 'warning.main'
                                    }} />

                                    {/* Header */}
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                                        <Typography variant="caption" color="textSecondary" sx={{ fontWeight: 500 }}>
                                            GPU Status
                                        </Typography>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                            <Box sx={{
                                                width: 6,
                                                height: 6,
                                                borderRadius: '50%',
                                                bgcolor: 'warning.main'
                                            }} />
                                            <Typography variant="caption" color="warning.main">
                                                No GPU
                                            </Typography>
                                        </Box>
                                    </Box>

                                    {/* No GPU Message */}
                                    <Typography variant="caption" color="textSecondary" sx={{ display: 'block', textAlign: 'center' }}>
                                        No CUDA GPU detected
                                    </Typography>
                                    <Typography variant="caption" color="textSecondary" sx={{ display: 'block', textAlign: 'center', mt: 0.5 }}>
                                        Using CPU for processing
                                    </Typography>
                                </>
                            )}
                        </Box>
                    </Box>
                </Box>

                {/* Main Content with Sidebar Layout */}
                <Box sx={{ 
                    display: 'flex', 
                    width: '100%',
                    flex: 1,
                    gap: 1,
                    borderRadius: 3,
                    minHeight: 0
                }}>
                    {/* Left Sidebar with Vertical Tabs */}
                    <Paper sx={{
                        width: 180,
                        backgroundColor: 'background.paper',
                        borderRadius: 2,
                        overflow: 'hidden',
                        display: 'flex',
                        flexDirection: 'column',
                        height: '100%'
                    }}>
                        <Tabs
                            value={tabValue}
                            onChange={handleTabChange}
                            orientation="vertical"
                            aria-label="main navigation tabs"
                            sx={{
                                height: '100%',
                                '& .MuiTabs-indicator': {
                                    left: 0,
                                    width: 9,
                                    backgroundColor: 'primary.main',
                                    borderRadius: '0 2px 2px 0'
                                },
                                '& .MuiTab-root': {
                                    alignItems: 'flex-start',
                                    textAlign: 'left',
                                    minHeight: 48,
                                    fontSize: '0.9rem',
                                    fontWeight: 500,
                                    textTransform: 'none',
                                    color: 'text.secondary',
                                    px: 2,
                                    py: 1.5,
                                    '&.Mui-selected': {
                                        color: 'primary.main',
                                        fontWeight: 600,
                                        backgroundColor: 'rgba(53, 157, 255, 0.1)'
                                    },
                                    '&:hover': {
                                        color: 'text.primary',
                                        backgroundColor: 'rgba(53, 147, 255, 0.05)'
                                    }
                                }
                            }}
                        >
                            <Tab label="Data Processing" />
                            <Tab label="Training" />
                            <Tab label="Generation" />
                        </Tabs>
                    </Paper>

                    {/* Main Content Area */}
                    <Paper sx={{
                        flex: 1,
                        backgroundColor: 'background.paper',
                        borderRadius: 2,
                        display: 'flex',
                        flexDirection: 'column',
                        minHeight: '500px', // Set minimum height instead of fixed height
                        maxHeight: 'calc(100vh - 160px)', // Allow content to be scrollable
                        overflow: 'hidden'
                    }}>

                    {/* Data Processing Tab */}
                    <TabPanel value={tabValue} index={0}>
                        <Grid container spacing={3} sx={{ 
                            flex: 1,
                            minHeight: 0,
                            flexWrap: 'wrap', // Allow wrapping for better responsive behavior
                            alignItems: 'stretch'
                        }}>
                            <Grid item xs={12} md={8} sx={{ 
                                display: 'flex', 
                                flexDirection: 'column',
                                minHeight: 0,
                                overflow: 'hidden'
                            }}>
                                <Box sx={{
                                    flex: 1,
                                    overflow: 'auto',
                                    pr: 1,
                                    maxHeight: 'calc(100vh - 280px)', // Ensure scrolling works properly
                                    '&::-webkit-scrollbar': {
                                        width: '6px',
                                    },
                                    '&::-webkit-scrollbar-track': {
                                        background: '#30363D',
                                        borderRadius: '3px',
                                    },
                                    '&::-webkit-scrollbar-thumb': {
                                        background: '#484F58',
                                        borderRadius: '3px',
                                        '&:hover': {
                                            background: '#6E7681',
                                        },
                                    },
                                }}>
                                    <Typography variant="h5" gutterBottom>
                                        Upload Audio Files with Annotations
                                    </Typography>

                                    {uploadRows.map((row, index) => (
                                        <AudioUploadRow
                                            key={`${uploadKey}-${index}`}
                                            index={index}
                                            data={row}
                                            onChange={updateUploadRow}
                                            onRemove={removeUploadRow}
                                        />
                                    ))}

                                    <Button
                                        variant="outlined"
                                        startIcon={<AddIcon />}
                                    onClick={addUploadRow}
                                    sx={{ mb: 3 }}
                                >
                                    Add Another Row
                                </Button>

                                <Button
                                    variant="contained"
                                    size="large"
                                    onClick={processFiles}
                                    disabled={isProcessing}
                                    startIcon={isProcessing ? <CircularProgress size={20} /> : <UploadIcon />}
                                    fullWidth
                                >
                                    {isProcessing ? 'Processing...' : 'Process Files'}
                                </Button>
                                </Box>
                            </Grid>

                            <Grid item xs={12} md={4}>
                                <Typography variant="h5" gutterBottom>
                                    Processing Status
                                </Typography>

                                {processingStatus && (
                                    <Alert severity="info" sx={{ mb: 2 }}>
                                        {processingStatus}
                                    </Alert>
                                )}

                                {systemStatus && (
                                    <Paper sx={{
                                        p: 2,
                                        mb: 2,
                                        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4), 0 4px 12px rgba(0, 0, 0, 0.3)',
                                        borderRadius: 2,
                                        border: '1px solid #30363D',
                                        background: 'linear-gradient(135deg, #161B22 0%, #21262D 100%)',
                                        '&:hover': {
                                            boxShadow: '0 12px 32px rgba(0, 0, 0, 0.5), 0 6px 16px rgba(0, 0, 0, 0.4)',
                                            transform: 'translateY(-2px)',
                                            transition: 'all 0.3s ease'
                                        },
                                        transition: 'all 0.3s ease'
                                    }}>
                                        <Typography variant="h6" gutterBottom>System Status</Typography>
                                        <Typography variant="body2">Raw Files: {systemStatus.raw_files}</Typography>
                                        <Typography variant="body2">Processed Segments: {systemStatus.processed_segments}</Typography>
                                        <Typography variant="body2" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                                            Total Duration: {formatDuration(systemStatus.total_duration || 0)}
                                        </Typography>
                                        <Typography variant="body2">
                                            Custom Metadata: {systemStatus.has_metadata_json ? 'Yes' : 'Not Found'}
                                        </Typography>
                                        {systemStatus.raw_file_names && systemStatus.raw_file_names.length > 0 && (
                                            <Box sx={{ mt: 1 }}>
                                                <Typography variant="body2" color="textSecondary">
                                                    Recent files: {systemStatus.raw_file_names.join(', ')}
                                                </Typography>
                                            </Box>
                                        )}
                                    </Paper>
                                )}

                            </Grid>
                        </Grid>
                    </TabPanel>

                    {/* Training Tab */}
                    <TabPanel value={tabValue} index={1}>
                        <Grid container spacing={3} alignItems="stretch" sx={{ 
                            height: '100%',
                            flexWrap: 'wrap' // Allow wrapping for better responsive behavior
                        }}>
                            <Grid item xs={12} md={6} sx={{ 
                                display: 'flex', 
                                flexDirection: 'column',
                                overflow: 'hidden'
                            }}>
                                <Box sx={{
                                    flex: 1,
                                    overflow: 'auto',
                                    pr: 1,
                                    maxHeight: 'calc(100vh - 280px)', // Ensure scrolling works properly
                                    '&::-webkit-scrollbar': {
                                        width: '6px',
                                    },
                                    '&::-webkit-scrollbar-track': {
                                        background: '#30363D',
                                        borderRadius: '3px',
                                    },
                                    '&::-webkit-scrollbar-thumb': {
                                        background: '#484F58',
                                        borderRadius: '3px',
                                        '&:hover': {
                                            background: '#6E7681',
                                        },
                                    },
                                }}>
                                    <Typography variant="h5" gutterBottom>
                                        Training Configuration
                                    </Typography>

                                    <FormControl fullWidth sx={{ mb: 2 }}>
                                        <InputLabel id="base-model-select-label">Base Model</InputLabel>
                                        <Select
                                            labelId="base-model-select-label"
                                            id="base-model-select"
                                            value={trainingConfig.baseModel}
                                            label="Base Model"
                                            onChange={(e) => setTrainingConfig({
                                                ...trainingConfig,
                                                baseModel: e.target.value
                                            })}
                                        >
                                        <MenuItem value="stable-audio-open-1.0">
                                            <Box>
                                                <Typography variant="body1">Stable Audio Open 1.0</Typography>
                                                <Typography variant="caption" color="textSecondary">
                                                    Full model (838M parameters)
                                                </Typography>
                                                {(() => {
                                                    const model = baseModels.find(m => m.name === 'stable-audio-open-1.0');
                                                    return model?.downloaded ? (
                                                        <Typography variant="caption" color="success.main" display="block">
                                                            Downloaded and ready
                                                        </Typography>
                                                    ) : (
                                                        <Typography variant="caption" color="error.main" display="block">
                                                            Not downloaded
                                                        </Typography>
                                                    );
                                                })()}
                                            </Box>
                                        </MenuItem>
                                        <MenuItem value="stable-audio-open-small">
                                            <Box>
                                                <Typography variant="body1">Stable Audio Open Small</Typography>
                                                <Typography variant="caption" color="textSecondary">
                                                    Small model (faster training)
                                                </Typography>
                                                {(() => {
                                                    const model = baseModels.find(m => m.name === 'stable-audio-open-small');
                                                    return model?.downloaded ? (
                                                        <Typography variant="caption" color="success.main" display="block">
                                                            Downloaded and ready
                                                        </Typography>
                                                    ) : (
                                                        <Typography variant="caption" color="error.main" display="block">
                                                            Not downloaded
                                                        </Typography>
                                                    );
                                                })()}
                                            </Box>
                                        </MenuItem>
                                    </Select>
                                </FormControl>

                                <TextField
                                    fullWidth
                                    label="Fine-tuned Model Name"
                                    value={trainingConfig.modelName}
                                    onChange={(e) => setTrainingConfig({
                                        ...trainingConfig,
                                        modelName: e.target.value
                                    })}
                                    sx={{ mb: 2 }}
                                />

                                <Accordion sx={{ mb: 2 }}>
                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                        <Typography variant="h6">Advanced Settings</Typography>
                                    </AccordionSummary>
                                    <AccordionDetails sx={{
                                        maxHeight: '400px',
                                        overflowY: 'auto',
                                        overflowX: 'hidden',
                                        '&::-webkit-scrollbar': {
                                            width: '8px',
                                        },
                                        '&::-webkit-scrollbar-track': {
                                            background: '#30363D',
                                            borderRadius: '4px',
                                        },
                                        '&::-webkit-scrollbar-thumb': {
                                            background: '#484F58',
                                            borderRadius: '4px',
                                            '&:hover': {
                                                background: '#6E7681',
                                            },
                                        },
                                    }}>
                                        <Grid container spacing={3}>
                                            {/* Row 1 */}
                                            <Grid item xs={6}>
                                                <Typography gutterBottom>Epochs</Typography>
                                                <Slider
                                                    value={trainingConfig.epochs}
                                                    onChange={(e, value) => setTrainingConfig({
                                                        ...trainingConfig,
                                                        epochs: value
                                                    })}
                                                    min={1}
                                                    max={1000}
                                                    marks
                                                    valueLabelDisplay="auto"
                                                    sx={{ mb: 2 }}
                                                />
                                            </Grid>

                                            <Grid item xs={6}>
                                                <Typography gutterBottom>Checkpoint Interval</Typography>
                                                <Slider
                                                    value={trainingConfig.checkpointSteps}
                                                    onChange={(e, value) => setTrainingConfig({
                                                        ...trainingConfig,
                                                        checkpointSteps: value
                                                    })}
                                                    min={10}
                                                    max={1000}
                                                    step={10}
                                                    marks
                                                    valueLabelDisplay="auto"
                                                    sx={{ mb: 2 }}
                                                />
                                            </Grid>

                                            {/* Row 2 */}
                                            <Grid item xs={6}>
                                                <Typography gutterBottom>Batch Size</Typography>
                                                <Slider
                                                    value={trainingConfig.batchSize}
                                                    onChange={(e, value) => setTrainingConfig({
                                                        ...trainingConfig,
                                                        batchSize: value
                                                    })}
                                                    min={1}
                                                    max={16}
                                                    marks
                                                    valueLabelDisplay="auto"
                                                    sx={{ mb: 2 }}
                                                />
                                            </Grid>

                                            <Grid item xs={6}>
                                                <Typography gutterBottom>Learning Rate</Typography>
                                                <Slider
                                                    value={trainingConfig.learningRate}
                                                    onChange={(e, value) => setTrainingConfig({
                                                        ...trainingConfig,
                                                        learningRate: value
                                                    })}
                                                    min={1e-6}
                                                    max={1e-3}
                                                    step={1e-6}
                                                    marks
                                                    valueLabelDisplay="auto"
                                                    sx={{ mb: 2 }}
                                                />
                                            </Grid>

                                        </Grid>
                                    </AccordionDetails>
                                </Accordion>



                                <Box sx={{ display: 'flex', gap: 2 }}>
                                    <Button
                                        variant="contained"
                                        size="large"
                                        onClick={startTraining}
                                        disabled={isTraining || (() => {
                                            // Check if the selected base model is downloaded
                                            const baseModel = baseModels.find(m => m.name === trainingConfig.baseModel);
                                            return baseModel ? !baseModel.downloaded : false;
                                        })()}
                                        startIcon={isTraining ? <CircularProgress size={20} /> : <PlayIcon />}
                                        sx={{ flex: 1 }}
                                    >
                                        {isTraining ? 'Training...' : 'Start Training'}
                                    </Button>
                                    <Button
                                        variant="outlined"
                                        color="error"
                                        size="large"
                                        onClick={stopTraining}
                                        disabled={!isTraining}
                                        startIcon={<StopIcon />}
                                        sx={{ flex: 1 }}
                                    >
                                        Stop Training
                                    </Button>
                                </Box>
                                
                                {/* Warning when base model is not downloaded */}
                                {(() => {
                                    const baseModel = baseModels.find(m => m.name === trainingConfig.baseModel);
                                    if (baseModel && !baseModel.downloaded) {
                                        return (
                                            <Alert 
                                                severity="error" 
                                                sx={{ 
                                                    mt: 2,
                                                    backgroundColor: 'rgba(219, 80, 68, 0)',
                                                    border: '1px solid #DB5044',
                                                    borderRadius: 2,
                                                    '& .MuiAlert-icon': {
                                                        color: '#DB5044'
                                                    }
                                                }}
                                            >
                                                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                    The selected base model "{baseModel.displayName}" is not downloaded.
                                                    Please use the File Authentication menu to download it before training.
                                                </Typography>
                                            </Alert>
                                        );
                                    }
                                    return null;
                                })()}
                                </Box>
                            </Grid>

                            <Grid item xs={12} md={6} sx={{ 
                                display: 'flex', 
                                flexDirection: 'column',
                                overflow: 'hidden'
                            }}>
                                <Box sx={{
                                    flex: 1,
                                    overflow: 'auto',
                                    pl: 1,
                                    '&::-webkit-scrollbar': {
                                        width: '6px',
                                    },
                                    '&::-webkit-scrollbar-track': {
                                        background: '#30363D',
                                        borderRadius: '3px',
                                    },
                                    '&::-webkit-scrollbar-thumb': {
                                        background: '#484F58',
                                        borderRadius: '3px',
                                        '&:hover': {
                                            background: '#6E7681',
                                        },
                                    },
                                }}>
                                    <Typography variant="h5" gutterBottom>
                                        Training Monitor
                                    </Typography>

                                    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                                    <TrainingMonitor
                                        isTraining={isTraining}
                                        trainingProgress={trainingProgress}
                                        trainingStatus={trainingStatus}
                                        trainingHistory={trainingHistory}
                                        trainingStartTime={trainingStartTime}
                                        trainingError={trainingError}
                                        trainingConfig={trainingConfig}
                                        systemStatus={systemStatus}
                                    />
                                </Box>
                                </Box>
                            </Grid>
                        </Grid>
                    </TabPanel>

                    {/* Generation Tab */}
                    <TabPanel value={tabValue} index={2}>
                        <Grid container spacing={3} sx={{ 
                            height: '100%',
                            flexWrap: 'wrap' // Allow wrapping for better responsive behavior
                        }}>
                            <Grid item xs={12} md={6} sx={{ 
                                display: 'flex', 
                                flexDirection: 'column',
                                overflow: 'hidden'
                            }}>
                                <Box sx={{
                                    flex: 1,
                                    overflow: 'auto',
                                    pr: 1,
                                    maxHeight: 'calc(100vh - 280px)', // Ensure scrolling works properly
                                    '&::-webkit-scrollbar': {
                                        width: '6px',
                                    },
                                    '&::-webkit-scrollbar-track': {
                                        background: '#30363D',
                                        borderRadius: '3px',
                                    },
                                    '&::-webkit-scrollbar-thumb': {
                                        background: '#484F58',
                                        borderRadius: '3px',
                                        '&:hover': {
                                            background: '#6E7681',
                                        },
                                    },
                                }}>
                                    <Typography variant="h5" gutterBottom>
                                        Audio Generation
                                    </Typography>

                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                                    <FormControl fullWidth variant="outlined">
                                        <Select
                                            labelId="model-select-label"
                                            id="model-select"
                                            value={selectedModel || ''}
                                            label="Select Model"
                                            onChange={(event) => {
                                                console.log('Model dropdown selected:', event.target.value, typeof event.target.value);
                                                handleModelChange(event);
                                            }}
                                            displayEmpty
                                        >
                                            <MenuItem value="" disabled>
                                                <em>Select a model</em>
                                            </MenuItem>
                                            {/* Base Models Section */}
                                            <MenuItem disabled>
                                                <Typography variant="subtitle2" color="textSecondary">
                                                    ── Base Models (Ready for Generation) ──
                                                </Typography>
                                            </MenuItem>
                                            {baseModels.map((model) => (
                                                <MenuItem key={model.name} value={String(model.name)}>
                                                    <Box>
                                                        <Typography variant="body1">{model.displayName}</Typography>
                                                        <Typography variant="caption" color="textSecondary">
                                                            {model.description}
                                                        </Typography>
                                                        <Typography variant="caption" color="success.main" display="block">
                                                            Ready for inference
                                                        </Typography>
                                                    </Box>
                                                </MenuItem>
                                            ))}
                                            {/* Fine-tuned Models Section */}
                                            {availableModels.length > 0 && (
                                                <MenuItem disabled>
                                                    <Typography variant="subtitle2" color="textSecondary">
                                                        ── Fine-tuned Models ──
                                                    </Typography>
                                                </MenuItem>
                                            )}
                                            {availableModels.map((model) => (
                                                <MenuItem key={model.name} value={String(model.name)} disabled={false}>
                                                    <Box>
                                                        <Typography variant="body1">{model.name}</Typography>
                                                        <Typography variant="caption" color="textSecondary">
                                                            {model.has_checkpoint ? 'Checkpoint' : 'No Checkpoint'} |
                                                            {model.unwrapped_models && model.unwrapped_models.length > 0
                                                                ? ` ${model.unwrapped_models.length} unwrapped models`
                                                                : ' No unwrapped models'}
                                                        </Typography>
                                                    </Box>
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                    <IconButton
                                        onClick={fetchAvailableModels}
                                        title="Refresh Models"
                                        sx={{ minWidth: 40 }}
                                    >
                                        <RefreshIcon />
                                    </IconButton>
                                </Box>

                                {/* Unwrapped Model Selection for Fine-tuned Models */}
                                {selectedModel && availableModels.find(m => m.name === selectedModel)?.unwrapped_models?.length > 0 && (
                                    (() => {
                                        const unwrappedModels = availableModels.find(m => m.name === selectedModel)?.unwrapped_models || [];
                                        const validPaths = unwrappedModels.map(u => String(u.path));
                                        // Only allow the value if it's in the list, otherwise set to ''
                                        const safeSelected = validPaths.includes(selectedUnwrappedModel) ? selectedUnwrappedModel : '';
                                        return (
                                            <>
                                                <FormControl fullWidth sx={{ mb: 2 }} variant="outlined">
                                                    <Select
                                                        key={selectedModel}
                                                        labelId="unwrapped-model-select-label"
                                                        id="unwrapped-model-select"
                                                        value={safeSelected}
                                                        label="Select Checkpoint"
                                                        onChange={(e) => {
                                                            console.log('Selected checkpoint:', e.target.value, typeof e.target.value);
                                                            setSelectedUnwrappedModel(String(e.target.value));
                                                        }}
                                                        displayEmpty
                                                    >
                                                        <MenuItem value="" disabled>
                                                            <em>Select a checkpoint</em>
                                                        </MenuItem>
                                                        {unwrappedModels.map((unwrapped, index) => (
                                                            <MenuItem key={index} value={String(unwrapped.path)}>
                                                                <Box>
                                                                    <Typography variant="body1">{unwrapped.name}</Typography>
                                                                    <Typography variant="caption" color="textSecondary">
                                                                        Size: {unwrapped.size_mb} MB
                                                                    </Typography>
                                                                    <Typography variant="body2" color="success.main" display="block">
                                                                        Ready for inference
                                                                    </Typography>
                                                                </Box>
                                                            </MenuItem>
                                                        ))}
                                                    </Select>
                                                </FormControl>
                                            </>
                                        );
                                    })()
                                )}

                                <TextField
                                    fullWidth
                                    multiline
                                    minRows={1}
                                    maxRows={4}
                                    label="Generation Prompt"
                                    placeholder="Describe the audio you want to generate..."
                                    value={generationPrompt}
                                    onChange={(e) => setGenerationPrompt(e.target.value)}
                                    sx={{ mb: 3 }}
                                />

                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                                    <Typography variant="body2" color="textSecondary">
                                        Desired Duration (seconds):
                                    </Typography>
                                    <Slider
                                        value={generationDuration}
                                        onChange={(e, value) => setGenerationDuration(value)}
                                        min={1}
                                        max={getMaxDuration()}
                                        step={1}
                                        marks
                                        valueLabelDisplay="auto"
                                    />
                                    <Typography variant="body2" color="textSecondary">
                                        {generationDuration}s
                                    </Typography>
                                </Box>



                                {isGenerating ? (
                                    <Box sx={{ mb: 3 }}>
                                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                            <CircularProgress size={20} sx={{ mr: 1 }} />
                                            <Typography variant="body2" color="textSecondary">
                                                Generating audio... {Math.round(generationProgress)}%
                                            </Typography>
                                        </Box>
                                        <LinearProgress
                                            variant="determinate"
                                            value={generationProgress}
                                            sx={{ height: 8, borderRadius: 4 }}
                                        />
                                        <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
                                            This may take 30-60 seconds depending on the prompt length
                                        </Typography>
                                    </Box>
                                ) : (
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        fullWidth
                                        onClick={generateAudio}
                                        disabled={!selectedModel || !generationPrompt.trim() || (() => {
                                            // Check if selected model is a base model and if it's downloaded
                                            const baseModel = baseModels.find(m => m.name === selectedModel);
                                            if (baseModel) {
                                                return !baseModel.downloaded;
                                            }
                                            // For fine-tuned models, allow if they have checkpoints
                                            return false;
                                        })()}
                                        sx={{ mb: 2 }}
                                    >
                                        Generate Audio
                                    </Button>
                                )}
                                
                                {/* Warnings for model issues */}
                                {selectedModel &&
                                    availableModels.find(m => m.name === selectedModel) &&
                                    availableModels.find(m => m.name === selectedModel)?.unwrapped_models?.length > 0 &&
                                    !selectedUnwrappedModel && (
                                        <Alert severity="warning" sx={{ mt: 2 }}>
                                            Please select a checkpoint for the selected fine-tuned model before generating audio.
                                        </Alert>
                                    )}
                                
                                {/* Warning when base model is not downloaded */}
                                {(() => {
                                    const baseModel = baseModels.find(m => m.name === selectedModel);
                                    if (baseModel && !baseModel.downloaded) {
                                        return (
                                            <Alert 
                                                severity="error" 
                                                sx={{ 
                                                    mt: 2,
                                                    backgroundColor: 'rgba(219, 80, 68, 0)',
                                                    border: '1px solid #DB5044',
                                                    borderRadius: 2,
                                                    '& .MuiAlert-icon': {
                                                        color: '#DB5044'
                                                    }
                                                }}
                                            >
                                                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                    The selected base model "{baseModel.displayName}" is not downloaded.
                                                    Please use the Authentication menu to download it before generating audio.
                                                </Typography>
                                            </Alert>
                                        );
                                    }
                                    return null;
                                })()}
                                </Box>
                            </Grid>

                            <Grid item xs={12} md={6} sx={{ 
                                display: 'flex', 
                                flexDirection: 'column',
                                overflow: 'hidden'
                            }}>
                                <Box sx={{
                                    flex: 1,
                                    overflow: 'auto',
                                    pl: 1,
                                    maxHeight: 'calc(100vh - 280px)', // Ensure scrolling works properly
                                    '&::-webkit-scrollbar': {
                                        width: '6px',
                                    },
                                    '&::-webkit-scrollbar-track': {
                                        background: '#30363D',
                                        borderRadius: '3px',
                                    },
                                    '&::-webkit-scrollbar-thumb': {
                                        background: '#484F58',
                                        borderRadius: '3px',
                                        '&:hover': {
                                            background: '#6E7681',
                                        },
                                    },
                                }}>
                                    <Typography variant="h5" gutterBottom>
                                        Selected Model
                                    </Typography>

                                    <Paper sx={{
                                    p: 2,
                                    mb: 2,
                                    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4), 0 4px 12px rgba(0, 0, 0, 0.3)',
                                    borderRadius: 2,
                                    border: '1px solid #30363D',
                                    background: 'linear-gradient(135deg, #161B22 0%, #21262D 100%)',
                                    '&:hover': {
                                        boxShadow: '0 12px 32px rgba(0, 0, 0, 0.5), 0 6px 16px rgba(0, 0, 0, 0.4)',
                                        transform: 'translateY(-2px)',
                                        transition: 'all 0.3s ease'
                                    },
                                    transition: 'all 0.3s ease'
                                }}>
                                    {selectedModel ? (
                                        (() => {
                                            // Check if it's a base model
                                            const baseModel = baseModels.find(m => m.name === selectedModel);
                                            if (baseModel) {
                                                const maxDuration = getMaxDuration();
                                                return (
                                                    <Box>
                                                        <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                                                            {baseModel.displayName}
                                                        </Typography>
                                                        <Typography variant="body2" color="textSecondary">
                                                            Type: Base Model
                                                        </Typography>
                                                        <Typography variant="body2" color="textSecondary">
                                                            {baseModel.description}
                                                        </Typography>
                                                        {baseModel.downloaded ? (
                                                            <Typography variant="body2" color="success.main" sx={{ fontWeight: 'bold' }}>
                                                                Ready for inference
                                                            </Typography>
                                                        ) : (
                                                            <Typography variant="body2" color="error.main" >
                                                                Model not downloaded
                                                            </Typography>
                                                        )}
                                                    </Box>
                                                );
                                            }

                                            // Check if it's a fine-tuned model
                                            const model = availableModels.find(m => m.name === selectedModel);
                                            if (model) {
                                                const maxDuration = getMaxDuration();
                                                return (
                                                    <Box>
                                                        <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                                                            {model.name}
                                                        </Typography>
                                                        <Typography variant="body2" color="textSecondary">
                                                            Type: Fine-tuned Model
                                                        </Typography>
                                                        <Typography variant="body2" color="textSecondary">
                                                            Path: {model.path}
                                                        </Typography>
                                                        <Typography variant="body2" color="textSecondary">
                                                            Checkpoint: {model.has_checkpoint ? 'Available' : 'Missing'}
                                                        </Typography>

                                                        {model.unwrapped_models && model.unwrapped_models.length > 0 && (
                                                            <Box sx={{ mt: 2 }}>
                                                                <Typography variant="subtitle2" color="primary" gutterBottom>
                                                                    Selected Unwrapped Model for Generation
                                                                </Typography>
                                                                {selectedUnwrappedModel ? (
                                                                    (() => {
                                                                        const selectedUnwrapped = model.unwrapped_models.find(u => u.path === selectedUnwrappedModel);
                                                                        if (selectedUnwrapped) {
                                                                            const isLargeModel = selectedUnwrapped.size_mb >= 2000;
                                                                            return (
                                                                                <Box>
                                                                                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                                                                        {selectedUnwrapped.name}
                                                                                    </Typography>
                                                                                    <Typography variant="caption" color="textSecondary">
                                                                                        Size: {selectedUnwrapped.size_mb} MB
                                                                                    </Typography>
                                                                                    <Typography variant="body2" color="primary.main" sx={{ fontWeight: 'bold' }}>
                                                                                        Max Duration: {maxDuration} seconds ({isLargeModel ? 'Large Model' : 'Small Model'})
                                                                                    </Typography>
                                                                                    <Typography variant="body2" color="success.main">
                                                                                        Ready for inference
                                                                                    </Typography>
                                                                                </Box>
                                                                            );
                                                                        }
                                                                        return null;
                                                                    })()
                                                                ) : (
                                                                    <Typography variant="caption" color="error">
                                                                        No checkpoint selected.
                                                                    </Typography>
                                                                )}
                                                            </Box>
                                                        )}
                                                    </Box>
                                                );
                                            }

                                            return (
                                                <Typography variant="body2" color="textSecondary">
                                                    Model not found
                                                </Typography>
                                            );
                                        })()
                                    ) : (
                                        <Typography variant="body2" color="textSecondary">
                                            Please select a model to generate audio
                                        </Typography>
                                    )}
                                </Paper>

                                {/* Checkpoint Management Section */}
                                {selectedModel && availableModels.find(m => m.name === selectedModel) && (
                                    <CheckpointManager
                                        model={availableModels.find(m => m.name === selectedModel)}
                                        onRefresh={refreshAllModels}
                                    />
                                )}

                                <Typography variant="h5" gutterBottom>
                                    Generated Fragments
                                </Typography>

                                <GeneratedFragmentsWindow
                                    fragments={generatedFragments}
                                    onDownload={downloadFragment}
                                />
                                </Box>
                            </Grid>
                        </Grid>
                    </TabPanel>
                    </Paper>
                </Box>

                {/* Start Fresh Confirmation Dialog */}
                <Dialog
                    open={showStartFreshDialog}
                    onClose={() => setShowStartFreshDialog(false)}
                    aria-labelledby="start-fresh-dialog-title"
                >
                    <DialogTitle id="start-fresh-dialog-title">
                        Start Fresh - Delete All Data
                    </DialogTitle>
                    <DialogContent>
                        <Typography sx={{ mt: 3 }}>
                            This will permanently delete all uploaded audio files, processed segments, and metadata files.
                            This action cannot be undone.
                        </Typography>
                        <Typography variant="body2" color="error" sx={{ mt: 2 }}>
                            Are you sure you want to continue?
                        </Typography>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={() => setShowStartFreshDialog(false)}>
                            Cancel
                        </Button>
                        <Button
                            onClick={handleStartFresh}
                            color="error"
                            variant="contained"
                            disabled={isStartingFresh}
                        >
                            {isStartingFresh ? 'Deleting...' : 'Delete All Data'}
                        </Button>
                    </DialogActions>
                </Dialog>

                {/* Free GPU Memory Confirmation Dialog */}
                <Dialog
                    open={showFreeGPUDialog}
                    onClose={() => setShowFreeGPUDialog(false)}
                    aria-labelledby="free-gpu-dialog-title"
                >
                    <DialogTitle id="free-gpu-dialog-title">
                        Free GPU Memory
                    </DialogTitle>
                    <DialogContent>
                        <Typography sx={{ mt: 3 }}>
                            This will stop all running processes and free GPU memory. Any active training will be stopped immediately.
                        </Typography>
                        <Typography variant="body2" color="warning.main" sx={{ mt: 2 }}>
                            Are you sure you want to continue?
                        </Typography>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={() => setShowFreeGPUDialog(false)}>
                            Cancel
                        </Button>
                        <Button
                            onClick={handleFreeGPUMemory}
                            color="primary"
                            variant="contained"
                            disabled={isFreeingGPU}
                        >
                            {isFreeingGPU ? 'Freeing...' : 'Free GPU Memory'}
                        </Button>
                    </DialogActions>
                </Dialog>
            </Container>
            </Box>
        </ThemeProvider>
    );
}

export default App; 