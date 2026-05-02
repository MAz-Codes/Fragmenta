import React, { useState, useEffect, useMemo, useRef, Suspense, lazy } from 'react';
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
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    LinearProgress,
    Slider,
    FormControl,
    Select,
    MenuItem,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    FormControlLabel,
    Switch,
    CssBaseline,
    ThemeProvider,
    useMediaQuery
} from '@mui/material';
import {
    Plus as AddIcon,
    Database as UploadIcon,
    Play as PlayIcon,
    Square as StopIcon,
    Cpu as ActivityIcon,
    SlidersHorizontal as SlidersIcon,
    Music as SparklesIcon,
    RefreshCw as RefreshIcon,
    ChevronDown as ExpandMoreIcon,
    CloudDownload as CloudDownloadIcon,
    FolderOpen as FolderOpenIcon,
    Info as InfoIcon,
    BookOpen as BookOpenIcon,
    Moon as MoonIcon,
    Sun as SunIcon,
    Piano as PerformanceIcon,
    AlertCircle as AlertIcon
} from 'lucide-react';
import api from './api';
import HfAuthDialog from './components/HfAuthDialog';
import TabPanel from './components/TabPanel';
import AudioUploadRow from './components/AudioUploadRow';
import BulkAnnotatePanel from './components/BulkAnnotatePanel';
import CsvImportPanel from './components/CsvImportPanel';
import TrainingMonitor from './components/TrainingMonitor';
import ModelUnwrapButton from './components/ModelUnwrapButton';
import CheckpointManager from './components/CheckpointManager';
import GeneratedFragmentsWindow from './components/GeneratedFragmentsWindow';
import WelcomePage from './components/WelcomePage';
import { clearPerformanceSession } from './components/usePerformanceSession';
import { formatDuration } from './utils/format';
import theme, { appStyles, lightTheme } from './theme';

const PerformancePanel = lazy(() => import('./components/PerformancePanel'));

const COLOR_MODE_STORAGE_KEY = 'fragmenta-color-mode';
const HIDE_WELCOME_PAGE_KEY = 'fragmenta-hide-welcome';
const PERFORMANCE_ENABLED_KEY = 'fragmenta-performance-enabled';

function App() {
    const [tabValue, setTabValue] = useState(0);
    const [uploadRows, setUploadRows] = useState([
        { file: null, prompt: '', audioUrl: '' }
    ]);
    const [processingStatus, setProcessingStatus] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [processedCount, setProcessedCount] = useState(0);
    const [chunksPreview, setChunksPreview] = useState([]);

    const [showWelcomePage, setShowWelcomePage] = useState(
        () => window.localStorage.getItem(HIDE_WELCOME_PAGE_KEY) !== 'true'
    );
    const [performanceEnabled, setPerformanceEnabled] = useState(
        () => window.localStorage.getItem(PERFORMANCE_ENABLED_KEY) === 'true'
    );
    const togglePerformance = () => {
        setPerformanceEnabled((prev) => {
            const next = !prev;
            window.localStorage.setItem(PERFORMANCE_ENABLED_KEY, next ? 'true' : 'false');
            if (!next && tabValue === 3) setTabValue(0);
            if (next) setTabValue(3);
            return next;
        });
    };
    const [authDialogOpen, setAuthDialogOpen] = useState(false);
    const [showInfoDialog, setShowInfoDialog] = useState(false);
    const [isOpeningDocumentation, setIsOpeningDocumentation] = useState(false);
    const [colorMode, setColorMode] = useState(() => {
        if (typeof window === 'undefined') {
            return 'dark';
        }

        const savedMode = window.localStorage.getItem(COLOR_MODE_STORAGE_KEY);
        if (savedMode === 'light' || savedMode === 'dark') {
            return savedMode;
        }

        return 'dark';
    });

    const [trainingConfig, setTrainingConfig] = useState({
        epochs: 30,
        checkpointSteps: 500,
        checkpointAuto: true,
        batchSize: 4,
        learningRate: 1e-4,
        modelName: 'my_fine_tuned_model',
        baseModel: '',
        saveWrappedCheckpoint: false,
        precision: 'auto'
    });
    const [checkpointPreview, setCheckpointPreview] = useState(null);
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
    const [generatedFragments, setGeneratedFragments] = useState([]);
    const [currentFilename, setCurrentFilename] = useState('');
    const [cfgScale, setCfgScale] = useState(7.0);
    const [steps, setSteps] = useState(250);
    const [batchCount, setBatchCount] = useState(1);
    const [randomSeed, setRandomSeed] = useState(true);
    const [seedValue, setSeedValue] = useState('');
    const generationAbortRef = useRef(null);
    const stopGenerationRef = useRef(false);

    const slugifyPrompt = (text, maxLen = 40) => {
        const slug = (text || '').trim().toLowerCase()
            .replace(/[^a-z0-9]+/g, '_')
            .replace(/_+/g, '_')
            .replace(/^_|_$/g, '');
        return (slug.slice(0, maxLen) || 'untitled');
    };

    const buildFragmentFilename = (prompt, timestampStr, batchIndex, batchTotal) => {
        const suffix = batchTotal > 1 ? `_${batchIndex}` : '';
        return `fragmenta_${timestampStr}_${slugifyPrompt(prompt)}${suffix}.wav`;
    };

    const downloadAudio = () => {
        if (generatedAudioBlob) {
            const url = URL.createObjectURL(generatedAudioBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = currentFilename || 'fragmenta_output.wav';
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
    const [isStatusLoading, setIsStatusLoading] = useState(false);
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
    // Bumping this key forces the performance panel to remount, which is how
    // we flush its in-memory session state on Fresh Start (clearing localStorage
    // alone wouldn't reset the mounted panel's useState mirrors).
    const [performanceResetKey, setPerformanceResetKey] = useState(0);
    const [isFreeingGPU, setIsFreeingGPU] = useState(false);
    const [showFreeGPUDialog, setShowFreeGPUDialog] = useState(false);
    const [modelWarning, setModelWarning] = useState({
        open: false,
        title: '',
        message: '',
        canOpenModels: false,
    });
    const appTheme = useMemo(
        () => (colorMode === 'light' ? lightTheme : theme),
        [colorMode]
    );
    const isCompactLayout = useMediaQuery(appTheme.breakpoints.down('md'));
    const isIconOnlySidebar = useMediaQuery(appTheme.breakpoints.between('md', 'lg'));

    useEffect(() => {
        setSelectedUnwrappedModel('');
    }, [selectedModel]);

    useEffect(() => {
        console.log('Model changed:', selectedModel);
    }, [selectedModel]);

    // Resolve the base model identity for the currently-selected entry. Works
    // for both base-model selections (selectedModel === 'stable-audio-open-...')
    // and fine-tunes (where the API returns base_model from training_metadata).
    const resolvedBaseModel = (() => {
        if (!selectedModel) return null;
        if (selectedModel === 'stable-audio-open-small' || selectedModel === 'stable-audio-open-1.0') {
            return selectedModel;
        }
        const model = availableModels.find(m => m.name === selectedModel);
        if (model?.base_model) return model.base_model;
        // Legacy fine-tunes without base_model metadata: fall back to the
        // unwrapped-file size heuristic.
        if (model && selectedUnwrappedModel) {
            const u = model.unwrapped_models?.find(x => x.path === selectedUnwrappedModel);
            if (u) return (u.size_mb || 0) < 2000 ? 'stable-audio-open-small' : 'stable-audio-open-1.0';
        }
        return null;
    })();

    const isSmallModel = resolvedBaseModel === 'stable-audio-open-small';

    const getMaxDuration = () => {
        if (!selectedModel) return 10;
        if (resolvedBaseModel === 'stable-audio-open-small') return 11;
        if (resolvedBaseModel === 'stable-audio-open-1.0') return 47;
        return 10;
    };

    useEffect(() => {
        const maxDuration = getMaxDuration();
        if (generationDuration > maxDuration) {
            setGenerationDuration(maxDuration);
        }
        // The distilled small model is hard-coded to 8 steps + pingpong sampler
        // at the backend regardless of slider value; snap the slider so the UI
        // reflects what will actually run.
        if (isSmallModel && steps !== 8) {
            setSteps(8);
        }
    }, [selectedModel, selectedUnwrappedModel, isSmallModel]);

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
        setIsStatusLoading(true);
        try {
            const response = await api.get('/api/status');
            setSystemStatus(response.data);
        } catch (error) {
            console.error('Error fetching system status:', error);
        } finally {
            setIsStatusLoading(false);
        }
    };

    const fetchAvailableModels = async () => {
        try {
            const response = await api.get('/api/models');
            console.log('Fetched models:', response.data.models);
            setAvailableModels(response.data.models || []);
        } catch (error) {
            console.error('Error fetching available models:', error);
        }
    };

    const fetchBaseModelsStatus = async () => {
        try {
            const response = await api.get('/api/base-models/status');
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
            const response = await api.get('/api/gpu-memory-status');
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
        // Debounced preview of total_steps + resolved checkpoint cadence so the
        // user sees what "Auto" picks before launching training.
        const handle = setTimeout(async () => {
            try {
                const { checkpointAuto, ...rest } = trainingConfig;
                const { data } = await api.post('/api/training/checkpoint-preview', {
                    ...rest,
                    checkpointSteps: checkpointAuto ? null : trainingConfig.checkpointSteps,
                });
                setCheckpointPreview(data);
            } catch {
                setCheckpointPreview(null);
            }
        }, 300);
        return () => clearTimeout(handle);
    }, [
        trainingConfig.epochs,
        trainingConfig.batchSize,
        trainingConfig.checkpointSteps,
        trainingConfig.checkpointAuto,
    ]);

    useEffect(() => {
        let statusInterval;

        if (isTraining) {
            statusInterval = setInterval(async () => {
                try {
                    const statusResponse = await api.get('/api/training-status');
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

            const response = await api.post('/api/process-files', formData);

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
        const selectedBaseModel = baseModels.find(m => m.name === trainingConfig.baseModel);
        if (!selectedBaseModel) {
            showModelWarning({
                title: 'Base Model Required',
                message: 'Please select a base model before starting training.',
                canOpenModels: false,
            });
            return;
        }

        if (!selectedBaseModel.downloaded) {
            showModelWarning({
                title: 'Base Model Not Downloaded',
                message: `The selected base model "${selectedBaseModel.displayName}" is not downloaded.`,
                canOpenModels: true,
            });
            return;
        }

        setIsTraining(true);
        setTrainingProgress(0);
        setTrainingError(null);
        setTrainingStartTime(Date.now());
        setTrainingHistory([]);

        await api.post('/api/bulk-annotate/unload-clap').catch(() => {});

        try {
            const { checkpointAuto, ...rest } = trainingConfig;
            const payload = {
                ...rest,
                checkpointSteps: checkpointAuto ? null : trainingConfig.checkpointSteps,
            };
            const response = await api.post('/api/start-training', payload);
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
            const response = await api.post('/api/stop-training');
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

        const baseRequestData = {
            prompt: generationPrompt,
            duration: generationDuration,
            cfg_scale: cfgScale,
            steps: steps
        };

        const baseModel = baseModels.find(m => m.name === selectedModel);
        if (baseModel) {
            if (!baseModel.downloaded) {
                showModelWarning({
                    title: 'Base Model Not Downloaded',
                    message: `The selected base model "${baseModel.displayName}" is not downloaded.`,
                    canOpenModels: true,
                });
                return;
            }

            baseRequestData.model_name = selectedModel;
        } else if (selectedUnwrappedModel) {
            baseRequestData.unwrapped_model_path = selectedUnwrappedModel;
        } else {
            setProcessingStatus('Please select a model');
            return;
        }

        const parsedSeed = parseInt(seedValue, 10);
        if (!randomSeed && (Number.isNaN(parsedSeed) || parsedSeed < 0)) {
            setProcessingStatus('Please enter a non-negative integer seed, or enable Random Seed');
            return;
        }

        const totalRuns = Math.max(1, Math.min(10, batchCount));

        await api.post('/api/bulk-annotate/unload-clap').catch(() => {});

        stopGenerationRef.current = false;
        const abortController = new AbortController();
        generationAbortRef.current = abortController;

        setIsGenerating(true);
        setGenerationProgress(0);

        let progressInterval;
        const startProgressTicker = () => {
            progressInterval = setInterval(() => {
                setGenerationProgress(prev => {
                    if (prev >= 90) return prev;
                    return prev + Math.random() * 3;
                });
            }, 1000);
        };
        const stopProgressTicker = () => {
            if (progressInterval) {
                clearInterval(progressInterval);
                progressInterval = null;
            }
        };

        const now = new Date();
        const pad = (n) => String(n).padStart(2, '0');
        const batchTimestamp =
            `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
            `_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;

        let stoppedEarly = false;
        let completedRuns = 0;
        try {
            for (let i = 0; i < totalRuns; i++) {
                if (stopGenerationRef.current) {
                    stoppedEarly = true;
                    break;
                }
                const batchIndex = i + 1;
                const runLabel = totalRuns > 1 ? ` (${batchIndex}/${totalRuns})` : '';
                setProcessingStatus(`Generating audio${runLabel}...`);
                setGenerationProgress(0);
                startProgressTicker();

                const seedForRun = randomSeed
                    ? Math.floor(Math.random() * 0xffffffff)
                    : parsedSeed;

                const requestData = {
                    ...baseRequestData,
                    seed: seedForRun,
                    batch_index: batchIndex,
                    batch_total: totalRuns
                };

                const response = await api.post('/api/generate', requestData, {
                    responseType: 'blob',
                    signal: abortController.signal
                });

                stopProgressTicker();
                setGenerationProgress(100);

                const audioUrl = URL.createObjectURL(response.data);
                const fragmentFilename = buildFragmentFilename(
                    generationPrompt, batchTimestamp, batchIndex, totalRuns
                );

                setGeneratedAudio(audioUrl);
                setGeneratedAudioBlob(response.data);
                setCurrentFilename(fragmentFilename);

                const newFragment = {
                    id: Date.now() + i,
                    prompt: generationPrompt,
                    duration: generationDuration,
                    cfgScale,
                    steps,
                    seed: seedForRun,
                    batchIndex,
                    batchTotal: totalRuns,
                    audioUrl,
                    audioBlob: response.data,
                    filename: fragmentFilename,
                    timestamp: new Date().toLocaleString()
                };

                setGeneratedFragments(prev => [...prev, newFragment]);
                completedRuns += 1;
            }

            if (stoppedEarly) {
                setProcessingStatus(
                    `Generation stopped after ${completedRuns}/${totalRuns} fragment${completedRuns === 1 ? '' : 's'}.`
                );
            } else {
                setProcessingStatus(totalRuns > 1
                    ? `Generated ${totalRuns} fragments successfully!`
                    : 'Audio generated successfully!');
            }

            setTimeout(() => {
                setGenerationProgress(0);
            }, 2000);

        } catch (error) {
            stopProgressTicker();
            setGenerationProgress(0);
            const wasAborted = error?.name === 'CanceledError'
                || error?.name === 'AbortError'
                || error?.code === 'ERR_CANCELED'
                || stopGenerationRef.current;
            if (wasAborted) {
                setProcessingStatus(
                    `Generation stopped after ${completedRuns}/${totalRuns} fragment${completedRuns === 1 ? '' : 's'}.`
                );
            } else {
                setProcessingStatus(`Generation error: ${error.response?.data?.error || error.message}`);
            }
        } finally {
            stopProgressTicker();
            setIsGenerating(false);
            generationAbortRef.current = null;
            stopGenerationRef.current = false;
        }
    };

    const stopGeneration = () => {
        stopGenerationRef.current = true;
        api.post('/api/stop-generation').catch(() => {});
        if (generationAbortRef.current) {
            try { generationAbortRef.current.abort(); } catch (_) {}
        }
        setProcessingStatus('Stopping generation…');
    };

    const handleStartFresh = async () => {
        setIsStartingFresh(true);
        setShowStartFreshDialog(false);

        try {
            const response = await api.post('/api/start-fresh');

            setUploadRows([{ file: null, prompt: '', audioUrl: '' }]);
            setProcessedCount(0);
            setChunksPreview([]);
            setGeneratedAudio(null);
            setGeneratedAudioBlob(null);
            setGeneratedFragments([]);
            setProcessingStatus('');
            setGenerationPrompt('');
            setUploadKey(prev => prev + 1);

            // Wipe persisted performance session and force-remount the panel so
            // its in-memory state resets to defaults along with localStorage.
            // (MIDI mappings and other app preferences are intentionally kept.)
            clearPerformanceSession();
            setPerformanceResetKey(prev => prev + 1);

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
            const response = await api.post('/api/free-gpu-memory');
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

    const handleOpenOutputFolder = async () => {
        try {
            const response = await api.post('/api/open-output-folder');
            if (!response.data.success) {
                setProcessingStatus(`Open output folder error: ${response.data.error || 'Unknown error'}`);
            }
        } catch (error) {
            setProcessingStatus(`Open output folder error: ${error.response?.data?.error || error.message}`);
        }
    };

    const handleOpenDocumentation = async (docKey = 'about') => {
        try {
            setIsOpeningDocumentation(true);
            const response = await api.post('/api/open-documentation', { doc_key: docKey });
            if (!response.data.success) {
                setProcessingStatus(`Open documentation error: ${response.data.error || 'Unknown error'}`);
                return;
            }
            if (response.data.message) {
                setProcessingStatus(response.data.message);
            }
        } catch (error) {
            setProcessingStatus(`Open documentation error: ${error.response?.data?.error || error.message}`);
        } finally {
            setIsOpeningDocumentation(false);
        }
    };

    const toggleColorMode = () => {
        setColorMode((prevMode) => {
            const nextMode = prevMode === 'light' ? 'dark' : 'light';
            if (typeof window !== 'undefined') {
                window.localStorage.setItem(COLOR_MODE_STORAGE_KEY, nextMode);
            }
            return nextMode;
        });
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

        const selectedBaseModel = baseModels.find(m => m.name === newSelectedModel);
        if (selectedBaseModel && !selectedBaseModel.downloaded) {
            showModelWarning({
                title: 'Base Model Not Downloaded',
                message: `The selected base model "${selectedBaseModel.displayName}" is not downloaded.`,
                canOpenModels: true,
            });
        }
    };

    const handleTrainingBaseModelChange = (event) => {
        const newBaseModel = event.target.value;
        setTrainingConfig({
            ...trainingConfig,
            baseModel: newBaseModel,
        });

        const selectedBaseModel = baseModels.find(m => m.name === newBaseModel);
        if (selectedBaseModel && !selectedBaseModel.downloaded) {
            showModelWarning({
                title: 'Base Model Not Downloaded',
                message: `The selected base model "${selectedBaseModel.displayName}" is not downloaded.`,
                canOpenModels: true,
            });
        }
    };

    const showModelWarning = ({ title, message, canOpenModels = false }) => {
        setModelWarning({
            open: true,
            title,
            message,
            canOpenModels,
        });
    };

    const closeModelWarning = () => {
        setModelWarning(prev => ({ ...prev, open: false }));
    };

    const handleOpenModelsFromWarning = () => {
        closeModelWarning();
        setAuthDialogOpen(true);
    };

    const getTrainingIndicatorState = () => {
        if (trainingError) {
            return { status: 'error', label: 'Error', animate: false };
        }
        if (isTraining) {
            return { status: 'live', label: 'Live', animate: true };
        }
        if (trainingProgress === 100) {
            return { status: 'complete', label: 'Complete', animate: false };
        }
        return { status: 'idle', label: 'Idle', animate: false };
    };

    const trainingIndicatorState = getTrainingIndicatorState();

    return (
        <ThemeProvider theme={appTheme}>
            <CssBaseline />
            <Box sx={appStyles.root}>
                <WelcomePage
                    open={showWelcomePage}
                    onClose={(dontShowAgain) => {
                        setShowWelcomePage(false);
                        if (dontShowAgain) {
                            window.localStorage.setItem(HIDE_WELCOME_PAGE_KEY, 'true');
                        }

                        api.post('/api/welcome-page-closed')
                            .then(() => {
                                console.log('Welcome page closure signal sent successfully');
                            })
                            .catch((error) => {
                                console.error('Failed to signal welcome page closure:', error);
                            });
                    }}
                />

                <Container maxWidth={false} sx={appStyles.container(showWelcomePage)}>
                    <Box sx={appStyles.headerRow}>
                        <Box sx={appStyles.headerBrand}>
                            {/* Logo */}
                            <Box sx={appStyles.logo} />

                            {/* Title */}
                            <Box>
                                <Typography variant="h4" component="h1" sx={appStyles.title}>
                                    Fragmenta
                                </Typography>

                            </Box>
                        </Box>

                        <Box sx={appStyles.headerActionsContainer(isCompactLayout)}>
                            <Box sx={appStyles.headerActionsGrid(isCompactLayout)}>
                                <Button
                                    variant="outlined"
                                    color="secondary"
                                    size="small"
                                    startIcon={<CloudDownloadIcon />}
                                    onClick={() => setAuthDialogOpen(true)}
                                    sx={appStyles.headerActionButton}
                                >
                                    Get Models
                                </Button>
                                <Button
                                    variant="contained"
                                    color="primary"
                                    size="small"
                                    startIcon={<RefreshIcon />}
                                    onClick={() => setShowFreeGPUDialog(true)}
                                    disabled={isFreeingGPU || !(gpuMemoryStatus && gpuMemoryStatus.cuda)}
                                    sx={appStyles.headerActionButtonWithOpacity(Boolean(gpuMemoryStatus && gpuMemoryStatus.cuda))}
                                >
                                    {isFreeingGPU ? 'Freeing...' : 'Free GPU'}
                                </Button>
                                <Button
                                    variant="outlined"
                                    color="secondary"
                                    size="small"
                                    startIcon={<FolderOpenIcon />}
                                    onClick={handleOpenOutputFolder}
                                    sx={appStyles.headerActionButton}
                                >
                                    Outputs
                                </Button>
                                <Button
                                    variant="contained"
                                    color="error"
                                    size="small"
                                    startIcon={<RefreshIcon />}
                                    onClick={() => setShowStartFreshDialog(true)}
                                    disabled={isStartingFresh}
                                    sx={appStyles.headerActionButton}
                                >
                                    {isStartingFresh ? 'Starting...' : 'Fresh Start'}
                                </Button>
                            </Box>

                            <Box sx={appStyles.gpuCard(isCompactLayout)}>
                                {gpuMemoryStatus && gpuMemoryStatus.cuda ? (
                                    <>
                                        <Box sx={appStyles.gpuHeaderRow}>
                                            <Typography variant="caption" color="textSecondary" sx={appStyles.gpuLabel}>
                                                GPU Memory
                                            </Typography>
                                            <Box sx={appStyles.gpuStatusGroup}>
                                                <Box
                                                    sx={appStyles.gpuStatusDot(
                                                        gpuMemoryStatus.cuda.free > 2 ? 'good' : gpuMemoryStatus.cuda.free > 0.5 ? 'low' : 'critical'
                                                    )}
                                                />
                                                <Typography variant="caption" color="textSecondary">
                                                    {gpuMemoryStatus.cuda.free > 2 ? 'Good' :
                                                        gpuMemoryStatus.cuda.free > 0.5 ? 'Low' : 'Critical'}
                                                </Typography>
                                            </Box>
                                        </Box>

                                        <Box sx={appStyles.gpuUsageWrap}>
                                            <Box sx={appStyles.gpuUsageTrack}>
                                                <Box
                                                    sx={appStyles.gpuUsageFill(
                                                        `${Math.min((gpuMemoryStatus.cuda.allocated / gpuMemoryStatus.cuda.total) * 100, 100)}%`,
                                                        'error.main'
                                                    )}
                                                />
                                                <Box
                                                    sx={appStyles.gpuUsageFill(
                                                        `${Math.min(((gpuMemoryStatus.cuda.allocated + gpuMemoryStatus.cuda.cached) / gpuMemoryStatus.cuda.total) * 100, 100)}%`,
                                                        'warning.main'
                                                    )}
                                                />
                                            </Box>
                                        </Box>

                                        <Box sx={appStyles.gpuFooterRow}>
                                            <Typography variant="caption" color="primary" sx={appStyles.gpuFreeText}>
                                                {gpuMemoryStatus.cuda.free.toFixed(1)}GB free
                                            </Typography>
                                            <Typography variant="caption" color="textSecondary">
                                                {gpuMemoryStatus.cuda.total.toFixed(1)}GB total
                                            </Typography>
                                        </Box>
                                    </>
                                ) : (
                                    <>
                                        <Box sx={appStyles.gpuHeaderRow}>
                                            <Typography variant="caption" color="textSecondary" sx={appStyles.gpuLabel}>
                                                GPU Status
                                            </Typography>
                                            <Box sx={appStyles.gpuStatusGroup}>
                                                <Box sx={appStyles.gpuStatusDot('low', false)} />
                                                <Typography variant="caption" color="warning.main">
                                                    No GPU
                                                </Typography>
                                            </Box>
                                        </Box>

                                        <Typography variant="caption" color="textSecondary" sx={appStyles.centeredCaption}>
                                            No CUDA GPU detected
                                        </Typography>
                                        <Typography variant="caption" color="textSecondary" sx={appStyles.centeredCaptionWithMargin}>
                                            Using CPU for processing
                                        </Typography>
                                    </>
                                )}
                            </Box>
                        </Box>
                    </Box>

                    {/* Main Content with Sidebar Layout */}
                    <Box sx={appStyles.mainLayout}>
                        {/* Left Sidebar with Vertical Tabs */}
                        <Paper sx={appStyles.navPaper}>
                            <Tabs
                                value={tabValue}
                                onChange={handleTabChange}
                                orientation={isCompactLayout ? 'horizontal' : 'vertical'}
                                aria-label="main navigation tabs"
                                sx={appStyles.navigationTabs(isCompactLayout, isIconOnlySidebar)}
                            >
                                <Tab icon={<UploadIcon size={20} />} iconPosition={isIconOnlySidebar ? 'top' : 'start'} label={isIconOnlySidebar ? undefined : 'Data Processing'} />
                                <Tab icon={<ActivityIcon size={20} />} iconPosition={isIconOnlySidebar ? 'top' : 'start'} label={isIconOnlySidebar ? undefined : 'Training'} />
                                <Tab icon={<SparklesIcon size={20} />} iconPosition={isIconOnlySidebar ? 'top' : 'start'} label={isIconOnlySidebar ? undefined : 'Generation'} />
                                <Tab
                                    icon={<PerformanceIcon size={20} />}
                                    iconPosition={isIconOnlySidebar ? 'top' : 'start'}
                                    label={isIconOnlySidebar ? undefined : (
                                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                                            Performance
                                            <Switch
                                                size="small"
                                                checked={performanceEnabled}
                                                onChange={() => {}}
                                                onClick={(e) => { e.stopPropagation(); togglePerformance(); }}
                                                sx={{ transform: 'scale(0.75)' }}
                                            />
                                        </Box>
                                    )}
                                    sx={{ opacity: performanceEnabled ? 1 : 0.5, transition: 'opacity 0.2s' }}
                                />
                            </Tabs>
                        </Paper>

                        {/* Main Content Area */}
                        <Paper sx={appStyles.mainContentPaper}>

                            {/* Data Processing Tab */}
                            <TabPanel value={tabValue} index={0}>
                                <Grid container spacing={{ xs: 2, sm: 2.5, md: 3 }} sx={appStyles.dataProcessingGrid}>
                                    <Grid item xs={12} md={8} sx={appStyles.primaryPaneItem}>
                                        <Box sx={appStyles.primaryPaneContent}>
                                            <Paper sx={{ p: 2 }} variant="outlined">
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                                                    <UploadIcon size={20} />
                                                    <Typography variant="h6">Manual Annotation</Typography>
                                                </Box>
                                                <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                                                    Upload audio files one by one and annotate them yourself.
                                                    Use this when you want full control over every annotation.
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
                                                    sx={appStyles.addRowButton}
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
                                                    {isProcessing ? 'Saving…' : 'Save to dataset'}
                                                </Button>
                                            </Paper>

                                            <BulkAnnotatePanel key={`bulk-${uploadKey}`} onCommitted={fetchSystemStatus} />

                                            <CsvImportPanel key={`csv-${uploadKey}`} onCommitted={fetchSystemStatus} />
                                        </Box>
                                    </Grid>

                                    <Grid item xs={12} md={4}>

                                        {processingStatus && (
                                            <Alert severity="info" sx={appStyles.sectionInfoAlert}>
                                                {processingStatus}
                                            </Alert>
                                        )}

                                        {!systemStatus && (
                                            <Paper sx={appStyles.elevatedInfoCard}>
                                                <Box sx={appStyles.sectionCardHeader}>
                                                    <Box component="span" sx={appStyles.sectionCardIcon}>
                                                        <FolderOpenIcon size={20} />
                                                    </Box>
                                                    <Typography variant="h6" sx={appStyles.sectionCardTitle}>Dataset Status</Typography>
                                                </Box>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, py: 1 }}>
                                                    <CircularProgress size={18} />
                                                    <Typography variant="body2" color="textSecondary">
                                                        Scanning dataset…
                                                    </Typography>
                                                </Box>
                                            </Paper>
                                        )}

                                        {systemStatus && (
                                            <Paper sx={appStyles.elevatedInfoCard}>
                                                <Box sx={appStyles.sectionCardHeader}>
                                                    <Box component="span" sx={appStyles.sectionCardIcon}>
                                                        <FolderOpenIcon size={20} />
                                                    </Box>
                                                    <Typography variant="h6" sx={appStyles.sectionCardTitle}>Dataset Status</Typography>
                                                    {isStatusLoading && (
                                                        <CircularProgress size={14} sx={{ ml: 1 }} />
                                                    )}
                                                </Box>
                                                <Typography variant="body2">Raw Files: {systemStatus.raw_files}</Typography>
                                                <Typography variant="body2" sx={appStyles.emphasizedPrimaryBody2}>
                                                    Total Duration: {formatDuration(systemStatus.total_duration || 0)}
                                                </Typography>
                                                <Typography variant="body2">
                                                    Custom Metadata: {systemStatus.has_metadata_json ? 'Yes' : 'Not Found'}
                                                </Typography>
                                                {systemStatus.raw_file_names && systemStatus.raw_file_names.length > 0 && (
                                                    <Box sx={appStyles.recentFilesBlock}>
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
                                <Grid container spacing={{ xs: 2, sm: 2.5, md: 3 }} alignItems="stretch" sx={appStyles.responsiveGrid}>
                                    <Grid item xs={12} md={6} sx={appStyles.secondaryPaneItem}>
                                        <Box sx={appStyles.primaryPaneContent}>
                                            <Paper sx={appStyles.elevatedInfoCard}>
                                                <Box sx={appStyles.sectionCardHeader}>
                                                    <Box component="span" sx={appStyles.sectionCardIcon}>
                                                        <SlidersIcon size={20} />
                                                    </Box>
                                                    <Typography variant="h6" sx={appStyles.sectionCardTitle}>Training Configuration</Typography>
                                                </Box>

                                                <FormControl fullWidth sx={appStyles.formControlMarginBottom}>
                                                    <Select
                                                        id="base-model-select"
                                                        value={trainingConfig.baseModel}
                                                        onChange={handleTrainingBaseModelChange}
                                                        displayEmpty
                                                        inputProps={{ 'aria-label': 'Base model' }}
                                                    >
                                                        <MenuItem value="" disabled>
                                                            <em>Select a model</em>
                                                        </MenuItem>
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
                                                    sx={appStyles.fieldMarginBottom}
                                                />

                                                <Accordion sx={appStyles.accordionMarginBottom}>
                                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                                        <Typography variant="h6">Advanced Settings</Typography>
                                                    </AccordionSummary>
                                                    <AccordionDetails sx={appStyles.advancedSettingsDetails}>
                                                        <Grid container spacing={{ xs: 2, sm: 2.5, md: 3 }}>
                                                            <Grid item xs={12}>
                                                                <Typography gutterBottom>Epochs</Typography>
                                                                <Box sx={appStyles.sliderRow}>
                                                                    <Slider
                                                                        value={trainingConfig.epochs}
                                                                        onChange={(e, value) => setTrainingConfig({
                                                                            ...trainingConfig,
                                                                            epochs: value
                                                                        })}
                                                                        min={1}
                                                                        max={1000}
                                                                        valueLabelDisplay="auto"
                                                                        sx={appStyles.sliderFlexGrow}
                                                                    />
                                                                    <TextField
                                                                        type="number"
                                                                        value={trainingConfig.epochs}
                                                                        onChange={(e) => {
                                                                            const val = parseInt(e.target.value) || 1;
                                                                            setTrainingConfig({
                                                                                ...trainingConfig,
                                                                                epochs: Math.max(1, Math.min(1000, val))
                                                                            });
                                                                        }}
                                                                        inputProps={{ min: 1, max: 1000, step: 1 }}
                                                                        sx={appStyles.sliderInputSmall}
                                                                        size="small"
                                                                    />
                                                                </Box>
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                                                                    <Typography>Checkpoint Interval (steps)</Typography>
                                                                    <FormControlLabel
                                                                        sx={{ m: 0 }}
                                                                        control={
                                                                            <Switch
                                                                                size="small"
                                                                                checked={trainingConfig.checkpointAuto}
                                                                                onChange={(e) => setTrainingConfig({
                                                                                    ...trainingConfig,
                                                                                    checkpointAuto: e.target.checked,
                                                                                    ...(e.target.checked ? {} : {
                                                                                        checkpointSteps: checkpointPreview?.checkpoint_every || trainingConfig.checkpointSteps,
                                                                                    }),
                                                                                })}
                                                                            />
                                                                        }
                                                                        label="Auto"
                                                                        labelPlacement="start"
                                                                    />
                                                                </Box>
                                                                <Box sx={appStyles.sliderRow}>
                                                                    <Slider
                                                                        value={
                                                                            trainingConfig.checkpointAuto
                                                                                ? (checkpointPreview?.checkpoint_every || trainingConfig.checkpointSteps)
                                                                                : trainingConfig.checkpointSteps
                                                                        }
                                                                        onChange={(e, value) => setTrainingConfig({
                                                                            ...trainingConfig,
                                                                            checkpointSteps: value
                                                                        })}
                                                                        min={10}
                                                                        max={Math.max(1000, checkpointPreview?.total_steps || 0)}
                                                                        step={10}
                                                                        valueLabelDisplay="auto"
                                                                        disabled={trainingConfig.checkpointAuto}
                                                                        sx={appStyles.sliderFlexGrow}
                                                                    />
                                                                    <TextField
                                                                        type="number"
                                                                        value={
                                                                            trainingConfig.checkpointAuto
                                                                                ? (checkpointPreview?.checkpoint_every ?? '')
                                                                                : trainingConfig.checkpointSteps
                                                                        }
                                                                        onChange={(e) => {
                                                                            const val = parseInt(e.target.value) || 10;
                                                                            const cap = Math.max(1000, checkpointPreview?.total_steps || 0);
                                                                            setTrainingConfig({
                                                                                ...trainingConfig,
                                                                                checkpointSteps: Math.max(10, Math.min(cap, val))
                                                                            });
                                                                        }}
                                                                        inputProps={{ min: 10, step: 10 }}
                                                                        sx={appStyles.sliderInputSmall}
                                                                        size="small"
                                                                        disabled={trainingConfig.checkpointAuto}
                                                                    />
                                                                </Box>
                                                                {checkpointPreview?.valid && checkpointPreview.checkpoint_every > 0 && (
                                                                    <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 0.5 }}>
                                                                        ≈ {Math.max(1, Math.round(checkpointPreview.total_steps / checkpointPreview.checkpoint_every))} checkpoints across {checkpointPreview.total_steps} total steps
                                                                        {trainingConfig.checkpointAuto ? ' (auto)' : ''}
                                                                    </Typography>
                                                                )}
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Typography gutterBottom>Learning Rate</Typography>
                                                                <Box sx={appStyles.sliderRow}>
                                                                    <Slider
                                                                        value={trainingConfig.learningRate}
                                                                        onChange={(e, value) => setTrainingConfig({
                                                                            ...trainingConfig,
                                                                            learningRate: value
                                                                        })}
                                                                        min={1e-6}
                                                                        max={1e-3}
                                                                        step={1e-6}
                                                                        valueLabelDisplay="auto"
                                                                        sx={appStyles.sliderFlexGrow}
                                                                    />
                                                                    <TextField
                                                                        type="number"
                                                                        value={trainingConfig.learningRate}
                                                                        onChange={(e) => {
                                                                            const val = parseFloat(e.target.value) || 1e-6;
                                                                            setTrainingConfig({
                                                                                ...trainingConfig,
                                                                                learningRate: Math.max(1e-6, Math.min(1e-3, val))
                                                                            });
                                                                        }}
                                                                        inputProps={{ min: 1e-6, max: 1e-3, step: 1e-6 }}
                                                                        sx={appStyles.sliderInputMedium}
                                                                        size="small"
                                                                    />
                                                                </Box>
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Typography gutterBottom>Batch Size</Typography>
                                                                <Box sx={appStyles.sliderRow}>
                                                                    <Slider
                                                                        value={trainingConfig.batchSize}
                                                                        onChange={(e, value) => setTrainingConfig({
                                                                            ...trainingConfig,
                                                                            batchSize: value
                                                                        })}
                                                                        min={1}
                                                                        max={32}
                                                                        step={1}
                                                                        valueLabelDisplay="auto"
                                                                        sx={appStyles.sliderFlexGrow}
                                                                    />
                                                                    <TextField
                                                                        type="number"
                                                                        value={trainingConfig.batchSize}
                                                                        onChange={(e) => {
                                                                            const val = parseInt(e.target.value, 10) || 1;
                                                                            setTrainingConfig({
                                                                                ...trainingConfig,
                                                                                batchSize: Math.max(1, Math.min(32, val))
                                                                            });
                                                                        }}
                                                                        inputProps={{ min: 1, max: 32, step: 1 }}
                                                                        sx={appStyles.sliderInputSmall}
                                                                        size="small"
                                                                    />
                                                                </Box>
                                                                <Typography variant="caption" color="textSecondary">
                                                                    Lower this if you hit CUDA out-of-memory; raise it for faster training on large GPUs.
                                                                </Typography>
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Typography gutterBottom>Precision</Typography>
                                                                <FormControl fullWidth size="small">
                                                                    <Select
                                                                        value={trainingConfig.precision}
                                                                        onChange={(e) => setTrainingConfig({
                                                                            ...trainingConfig,
                                                                            precision: e.target.value
                                                                        })}
                                                                    >
                                                                        <MenuItem value="auto">Auto (recommended)</MenuItem>
                                                                        <MenuItem value="bf16-mixed">bf16-mixed (Ampere+ GPUs)</MenuItem>
                                                                        <MenuItem value="16-mixed">16-mixed (older GPUs)</MenuItem>
                                                                        <MenuItem value="32">32 (full precision, highest VRAM)</MenuItem>
                                                                    </Select>
                                                                </FormControl>
                                                                <Typography variant="caption" color="textSecondary">
                                                                    Auto picks bf16-mixed on modern CUDA, 16-mixed on older cards, fp32 on CPU/MPS.
                                                                </Typography>
                                                            </Grid>

                                                        </Grid>
                                                    </AccordionDetails>
                                                </Accordion>



                                                <Box sx={appStyles.trainingActionRow}>
                                                    <Button
                                                        variant="contained"
                                                        size="large"
                                                        onClick={startTraining}
                                                        disabled={isTraining || !trainingConfig.baseModel || (() => {
                                                            // Check if the selected base model is downloaded
                                                            const baseModel = baseModels.find(m => m.name === trainingConfig.baseModel);
                                                            return baseModel ? !baseModel.downloaded : true;
                                                        })()}
                                                        startIcon={isTraining ? <CircularProgress size={20} /> : <PlayIcon />}
                                                        sx={appStyles.actionButtonFlexGrow}
                                                    >
                                                        {isTraining ? 'Training...' : 'Start'}
                                                    </Button>
                                                    <Button
                                                        variant="outlined"
                                                        color="error"
                                                        size="large"
                                                        onClick={stopTraining}
                                                        disabled={!isTraining}
                                                        startIcon={<StopIcon />}
                                                        sx={appStyles.actionButtonFlexGrow}
                                                    >
                                                        Stop
                                                    </Button>
                                                </Box>
                                            </Paper>
                                        </Box>
                                    </Grid>

                                    <Grid item xs={12} md={6} sx={appStyles.secondaryPaneItem}>
                                        <Box sx={appStyles.secondaryPaneContent}>
                                            <Box sx={appStyles.trainingMonitorWrap}>
                                                <TrainingMonitor
                                                    trainingProgress={trainingProgress}
                                                    trainingStatus={trainingStatus}
                                                    trainingHistory={trainingHistory}
                                                    trainingStartTime={trainingStartTime}
                                                    trainingError={trainingError}
                                                    trainingConfig={trainingConfig}
                                                    systemStatus={systemStatus}
                                                    indicatorState={trainingIndicatorState}
                                                />
                                            </Box>
                                        </Box>
                                    </Grid>
                                </Grid>
                            </TabPanel>

                            {/* Generation Tab */}
                            <TabPanel value={tabValue} index={2}>
                                <Grid container spacing={{ xs: 2, sm: 2.5, md: 3 }} sx={appStyles.responsiveGrid}>
                                    <Grid item xs={12} md={6} sx={appStyles.secondaryPaneItem}>
                                        <Box sx={appStyles.primaryPaneContent}>
                                            <Paper sx={appStyles.elevatedInfoCard}>
                                                <Box sx={appStyles.sectionCardHeader}>
                                                    <Box component="span" sx={appStyles.sectionCardIcon}>
                                                        <SparklesIcon size={20} />
                                                    </Box>
                                                    <Typography variant="h6" sx={appStyles.sectionCardTitle}>Audio Generation</Typography>
                                                </Box>

                                                <Box sx={appStyles.generationModelRow}>
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
                                                                        {model.downloaded ? (
                                                                            <Typography variant="caption" color="success.main" display="block">
                                                                                Ready for inference
                                                                            </Typography>
                                                                        ) : (
                                                                            <Typography variant="caption" color="error.main" display="block">
                                                                                Not downloaded
                                                                            </Typography>
                                                                        )}
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
                                                        sx={appStyles.refreshModelsButton}
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
                                                                <FormControl fullWidth sx={appStyles.formControlMarginBottom} variant="outlined">
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
                                                    sx={appStyles.fieldMarginBottomLarge}
                                                />

                                                <Box sx={appStyles.durationRow}>
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

                                                <Accordion sx={appStyles.accordionMarginBottom}>
                                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                                        <Typography variant="h6">Advanced Settings</Typography>
                                                    </AccordionSummary>
                                                    <AccordionDetails sx={appStyles.advancedSettingsDetails}>
                                                        <Grid container spacing={{ xs: 2, sm: 2.5, md: 3 }}>
                                                            <Grid item xs={12}>
                                                                <Typography gutterBottom>CFG Scale</Typography>
                                                                <Box sx={appStyles.sliderRow}>
                                                                    <Slider
                                                                        value={isSmallModel ? 1.0 : cfgScale}
                                                                        onChange={(e, value) => setCfgScale(value)}
                                                                        min={0.1}
                                                                        max={20}
                                                                        step={0.1}
                                                                        valueLabelDisplay="auto"
                                                                        disabled={isSmallModel}
                                                                        sx={appStyles.sliderFlexGrow}
                                                                    />
                                                                    <TextField
                                                                        type="number"
                                                                        value={isSmallModel ? 1.0 : cfgScale}
                                                                        onChange={(e) => {
                                                                            const val = parseFloat(e.target.value);
                                                                            if (Number.isNaN(val)) return;
                                                                            setCfgScale(Math.max(0.1, Math.min(20, val)));
                                                                        }}
                                                                        inputProps={{ min: 0.1, max: 20, step: 0.1 }}
                                                                        disabled={isSmallModel}
                                                                        sx={appStyles.sliderInputSmall}
                                                                        size="small"
                                                                    />
                                                                </Box>
                                                                {isSmallModel && (
                                                                    <Typography variant="caption" color="textSecondary">
                                                                        Locked at 1.0 for the distilled small model.
                                                                    </Typography>
                                                                )}
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Typography gutterBottom>Inference Steps</Typography>
                                                                <Box sx={appStyles.sliderRow}>
                                                                    <Slider
                                                                        value={steps}
                                                                        onChange={(e, value) => setSteps(value)}
                                                                        min={50}
                                                                        max={250}
                                                                        step={null}
                                                                        marks={[
                                                                            { value: 50, label: '50' },
                                                                            { value: 100, label: '100' },
                                                                            { value: 150, label: '150' },
                                                                            { value: 200, label: '200' },
                                                                            { value: 250, label: '250' },
                                                                        ]}
                                                                        valueLabelDisplay="auto"
                                                                        disabled={isSmallModel}
                                                                        sx={appStyles.sliderFlexGrow}
                                                                    />
                                                                </Box>
                                                                {isSmallModel && (
                                                                    <Typography variant="caption" color="textSecondary">
                                                                        Locked at 8 steps (pingpong sampler) for the distilled small model.
                                                                    </Typography>
                                                                )}
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Typography gutterBottom>Batch Generation (per prompt)</Typography>
                                                                <Box sx={appStyles.sliderRow}>
                                                                    <Slider
                                                                        value={batchCount}
                                                                        onChange={(e, value) => setBatchCount(value)}
                                                                        min={1}
                                                                        max={10}
                                                                        step={1}
                                                                        marks
                                                                        valueLabelDisplay="auto"
                                                                        sx={appStyles.sliderFlexGrow}
                                                                    />
                                                                    <TextField
                                                                        type="number"
                                                                        value={batchCount}
                                                                        onChange={(e) => {
                                                                            const val = parseInt(e.target.value, 10) || 1;
                                                                            setBatchCount(Math.max(1, Math.min(10, val)));
                                                                        }}
                                                                        inputProps={{ min: 1, max: 10, step: 1 }}
                                                                        sx={appStyles.sliderInputSmall}
                                                                        size="small"
                                                                    />
                                                                </Box>
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Typography gutterBottom>Seed</Typography>
                                                                <Box sx={appStyles.sliderRow}>
                                                                    <FormControlLabel
                                                                        control={
                                                                            <Switch
                                                                                checked={randomSeed}
                                                                                onChange={(e) => setRandomSeed(e.target.checked)}
                                                                            />
                                                                        }
                                                                        label="Random"
                                                                    />
                                                                    <TextField
                                                                        type="number"
                                                                        placeholder="e.g. 42"
                                                                        value={seedValue}
                                                                        onChange={(e) => setSeedValue(e.target.value)}
                                                                        disabled={randomSeed}
                                                                        inputProps={{ min: 0, max: 4294967295, step: 1 }}
                                                                        sx={appStyles.sliderFlexGrow}
                                                                        size="small"
                                                                    />
                                                                </Box>
                                                                <Typography variant="caption" color="textSecondary">
                                                                    {randomSeed
                                                                        ? 'A new random seed is used for each generation in the batch.'
                                                                        : 'The same seed is used for every generation in the batch.'}
                                                                </Typography>
                                                            </Grid>
                                                        </Grid>
                                                    </AccordionDetails>
                                                </Accordion>



                                                {isGenerating ? (
                                                    <Box sx={appStyles.generatingWrap}>
                                                        <Box sx={appStyles.generatingHeader}>
                                                            <CircularProgress size={20} sx={appStyles.generatingSpinner} />
                                                            <Typography variant="body2" color="textSecondary">
                                                                Generating audio... {Math.round(generationProgress)}%
                                                            </Typography>
                                                        </Box>
                                                        <LinearProgress
                                                            variant="determinate"
                                                            value={generationProgress}
                                                            sx={appStyles.generatingProgress}
                                                        />
                                                        <Typography variant="caption" color="textSecondary" sx={appStyles.generatingHint}>
                                                            Generation time may vary considerably depending on your hardware.
                                                        </Typography>
                                                        <Button
                                                            variant="outlined"
                                                            color="error"
                                                            fullWidth
                                                            startIcon={<StopIcon size={16} />}
                                                            onClick={stopGeneration}
                                                            disabled={stopGenerationRef.current}
                                                            sx={{ mt: 1.5 }}
                                                        >
                                                            Stop
                                                        </Button>
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
                                                        sx={appStyles.generateButton}
                                                    >
                                                        Generate Audio
                                                    </Button>
                                                )}

                                            {/* Warnings for model issues */}
                                                {selectedModel &&
                                                    availableModels.find(m => m.name === selectedModel) &&
                                                    availableModels.find(m => m.name === selectedModel)?.unwrapped_models?.length > 0 &&
                                                    !selectedUnwrappedModel && (
                                                        <Alert severity="warning" sx={appStyles.warningAlertTop}>
                                                            Please select a checkpoint for the selected fine-tuned model before generating audio.
                                                        </Alert>
                                                    )}
                                            </Paper>
                                        </Box>
                                    </Grid>

                                    <Grid item xs={12} md={6} sx={appStyles.secondaryPaneItem}>
                                        <Box sx={appStyles.secondaryPaneContent}>
                                            <Paper sx={appStyles.selectedModelCard}>
                                                <Box sx={appStyles.sectionCardHeader}>
                                                    <Box component="span" sx={appStyles.sectionCardIcon}>
                                                        <InfoIcon size={20} />
                                                    </Box>
                                                    <Typography variant="h6" sx={appStyles.sectionCardTitle}>Selected Model</Typography>
                                                </Box>
                                                {selectedModel ? (
                                                    (() => {
                                                        // Check if it's a base model
                                                        const baseModel = baseModels.find(m => m.name === selectedModel);
                                                        if (baseModel) {
                                                            return (
                                                                <Box>
                                                                    <Typography variant="body1" sx={appStyles.boldBodyText}>
                                                                        {baseModel.displayName}
                                                                    </Typography>
                                                                    {baseModel.downloaded ? (
                                                                        <Typography variant="body2" color="success.main" sx={appStyles.boldBodyText}>
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
                                                            const selectedUnwrapped = selectedUnwrappedModel
                                                                ? model.unwrapped_models?.find(u => u.path === selectedUnwrappedModel)
                                                                : null;
                                                            return (
                                                                <Box>
                                                                    <Typography variant="body1" sx={appStyles.boldBodyText}>
                                                                        {model.name}
                                                                    </Typography>
                                                                    <Typography variant="caption" color="textSecondary" sx={appStyles.selectedModelMetaText}>
                                                                        {model.checkpoints?.length || 0} wrapped checkpoints
                                                                    </Typography>

                                                                    {selectedUnwrapped ? (
                                                                        <Box sx={appStyles.unwrappedInfoWrap}>
                                                                            <Typography variant="body2" sx={appStyles.boldBodyText}>
                                                                                Using: {selectedUnwrapped.name}
                                                                            </Typography>
                                                                            <Typography variant="caption" color="textSecondary" sx={appStyles.selectedModelMetaText}>
                                                                                {selectedUnwrapped.size_mb} MB
                                                                            </Typography>
                                                                            <Typography variant="body2" color="primary.main" sx={appStyles.boldBodyText}>
                                                                                Max Duration: {maxDuration} seconds
                                                                            </Typography>
                                                                        </Box>
                                                                    ) : (
                                                                        <Typography variant="caption" color="error" sx={appStyles.unwrappedInfoWrap}>
                                                                            Select a checkpoint to generate audio.
                                                                        </Typography>
                                                                    )}

                                                                    <CheckpointManager
                                                                        model={model}
                                                                        onRefresh={refreshAllModels}
                                                                    />
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

                                            <GeneratedFragmentsWindow
                                                fragments={generatedFragments}
                                                onDownload={downloadFragment}
                                            />
                                        </Box>
                                    </Grid>
                                </Grid>
                            </TabPanel>

                            <TabPanel value={tabValue} index={3} keepMounted>
                                {performanceEnabled ? (
                                    <Suspense fallback={
                                        <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
                                            <CircularProgress size={28} />
                                        </Box>
                                    }>
                                        <PerformancePanel
                                            key={performanceResetKey}
                                            selectedModel={selectedModel}
                                            selectedUnwrappedModel={selectedUnwrappedModel}
                                            availableModels={availableModels}
                                            baseModels={baseModels}
                                            onSelectModel={setSelectedModel}
                                            onSelectUnwrappedModel={setSelectedUnwrappedModel}
                                            onRefreshModels={fetchAvailableModels}
                                            steps={steps}
                                            onStepsChange={setSteps}
                                            randomSeed={randomSeed}
                                            seedValue={seedValue}
                                            onRandomSeedChange={setRandomSeed}
                                            onSeedValueChange={setSeedValue}
                                            onPresetLoaded={() => setPerformanceResetKey(prev => prev + 1)}
                                        />
                                    </Suspense>
                                ) : (
                                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', py: 8, gap: 2 }}>
                                        <AlertIcon size={48} color="#FFB74D" />
                                        <Typography variant="body1" color="textSecondary" align="center">
                                            Performance mode is turned off. Toggle on from the sidebar if you wish to enter performance mode.
                                        </Typography>
                                    </Box>
                                )}
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
                            <Typography sx={appStyles.dialogBodyText}>
                                This will permanently delete all uploaded audio files, processed segments, and metadata files.
                                This action cannot be undone.
                            </Typography>
                            <Typography variant="body2" color="error" sx={appStyles.dialogErrorText}>
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
                            <Typography sx={appStyles.dialogBodyText}>
                                This will stop all running processes and free GPU memory. Any active training will be stopped immediately.
                            </Typography>
                            <Typography variant="body2" color="warning.main" sx={appStyles.dialogErrorText}>
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

                    <Dialog
                        open={modelWarning.open}
                        onClose={closeModelWarning}
                        aria-labelledby="model-warning-dialog-title"
                    >
                        <DialogTitle id="model-warning-dialog-title">
                            {modelWarning.title || 'Model Warning'}
                        </DialogTitle>
                        <DialogContent>
                            <Typography sx={appStyles.dialogBodyText}>
                                {modelWarning.message}
                            </Typography>
                            {modelWarning.canOpenModels && (
                                <Typography variant="body2" color="warning.main" sx={appStyles.dialogErrorText}>
                                    Use "Get Models" to authenticate and download the required model.
                                </Typography>
                            )}
                        </DialogContent>
                        <DialogActions>
                            <Button onClick={closeModelWarning}>
                                Close
                            </Button>
                            {modelWarning.canOpenModels && (
                                <Button
                                    onClick={handleOpenModelsFromWarning}
                                    color="primary"
                                    variant="contained"
                                >
                                    Get Models
                                </Button>
                            )}
                        </DialogActions>
                    </Dialog>
                </Container>
            </Box>

            <IconButton
                aria-label={colorMode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
                title={colorMode === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode'}
                onClick={toggleColorMode}
                sx={appStyles.modeToggleButton}
            >
                {colorMode === 'light' ? <MoonIcon size={18} /> : <SunIcon size={18} />}
            </IconButton>

            <IconButton
                aria-label="Open about and documentation"
                title="About & Documentation"
                onClick={() => setShowInfoDialog(true)}
                sx={appStyles.infoButton}
            >
                <InfoIcon size={18} />
            </IconButton>

            <Dialog
                open={showInfoDialog}
                onClose={() => setShowInfoDialog(false)}
                aria-labelledby="about-documentation-dialog-title"
                maxWidth="sm"
                fullWidth
            >
                <DialogTitle id="about-documentation-dialog-title">
                    <Box sx={appStyles.infoDialogTitleRow}>
                        <BookOpenIcon size={18} />
                        <span>About & Documentation</span>
                    </Box>
                </DialogTitle>
                <DialogContent>
                    <Typography sx={appStyles.infoDialogIntro}>
                        Fragmenta is an open source, local-first pipeline to fine-tune, train, generate and perform with text-to-audio diffusion models.
                        Made by the composer and researcher Misagh Azimi. 
                    </Typography>

                    <Typography variant="subtitle2" sx={appStyles.infoDialogSectionTitle}>
                        Resources
                    </Typography>

                    <Box sx={appStyles.infoDialogActionStack}>
                        <Button
                            variant="contained"
                            fullWidth
                            size="small"
                            startIcon={<InfoIcon size={16} />}
                            onClick={() => handleOpenDocumentation('about')}
                            disabled={isOpeningDocumentation}
                            sx={appStyles.infoDocButton}
                        >
                            About
                        </Button>
                        <Button
                            variant="outlined"
                            fullWidth
                            size="small"
                            startIcon={<BookOpenIcon size={16} />}
                            onClick={() => handleOpenDocumentation('documentation')}
                            disabled={isOpeningDocumentation}
                            sx={appStyles.infoDocButton}
                        >
                            Documentation
                        </Button>
                        <Button
                            variant="outlined"
                            fullWidth
                            size="small"
                            disabled
                            sx={appStyles.infoDocButton}
                        >
                            Tutorials (Coming soon...)
                        </Button>
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setShowInfoDialog(false)}>
                        Close
                    </Button>
                </DialogActions>
            </Dialog>

            <HfAuthDialog
                open={authDialogOpen}
                onClose={(success) => {
                    setAuthDialogOpen(false);
                    if (success) {
                        refreshAllModels();
                    }
                }}
            />
        </ThemeProvider>
    );
}

export default App; 