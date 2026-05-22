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
    Menu,
    ListItemIcon,
    ListItemText,
    Divider,
    Snackbar,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    FormControlLabel,
    Switch,
    CssBaseline,
    ThemeProvider,
    useMediaQuery,
    ToggleButton,
    ToggleButtonGroup,
    Tooltip,
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
    AlertCircle as AlertIcon,
    Wand2 as WandIcon,
    Trash2 as DeleteIcon,
    Menu as MenuIcon,
} from 'lucide-react';
import api from './api';
import HfAuthDialog from './components/HfAuthDialog';
import TabPanel from './components/TabPanel';
import AudioUploadRow from './components/AudioUploadRow';
import BulkAnnotatePanel from './components/BulkAnnotatePanel';
import CsvImportPanel from './components/CsvImportPanel';
import TrainingMonitor from './components/TrainingMonitor';
import ModelUnwrapButton from './components/ModelUnwrapButton';
import LoraCheckpointManager from './components/LoraCheckpointManager';
import CheckpointManagerWindow from './components/CheckpointManagerWindow';
import LoraStack from './components/LoraStack';
import EditPanel from './components/EditPanel';
import GeneratedFragmentsWindow from './components/GeneratedFragmentsWindow';
import WelcomePage from './components/WelcomePage';
import { clearPerformanceSession } from './components/usePerformanceSession';
import { formatDuration } from './utils/format';
import theme, { appStyles, lightTheme } from './theme';

const PerformancePanel = lazy(() => import('./components/PerformancePanel'));

const COLOR_MODE_STORAGE_KEY = 'fragmenta-color-mode';
const HIDE_WELCOME_PAGE_KEY = 'fragmenta-hide-welcome-v2';
const PERFORMANCE_ENABLED_KEY = 'fragmenta-performance-enabled';

function App() {
    const [tabValue, setTabValue] = useState(0);
    // Lags behind tabValue by ~fadeDuration so content swap happens
    // while the panel is invisible (cross-fade between pages).
    const [displayedTab, setDisplayedTab] = useState(0);
    const TAB_FADE_MS = 180;
    // Header sticky chrome only kicks in once the page has scrolled.
    const [isScrolled, setIsScrolled] = useState(false);
    // Measure the header's actual rendered height so the fixed nav
    // rail can be pinned at exactly the first card's top edge.
    const headerRef = useRef(null);
    const [navTopPx, setNavTopPx] = useState(94);
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
    const [checkpointMgrOpen, setCheckpointMgrOpen] = useState(false);
    const [generationModelSelectOpen, setGenerationModelSelectOpen] = useState(false);
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
        mode: 'lora',                         // SA3 is LoRA-only; field kept for back-compat
        steps: 5000,                          // SA3 trains by step count, not epochs
        checkpointSteps: 500,
        checkpointAuto: true,
        batchSize: 1,
        learningRate: 1e-4,
        modelName: 'my_lora',
        baseModel: 'sa3-small-music-base',    // only *-base checkpoints are valid targets
        precision: 'bf16',
        duration: 30.0,                       // max clip seconds per sample

        loraRank: 16,
        loraAlpha: 16,
        loraDropout: 0,
        adapterType: 'dora-rows',             // SA3 upstream default
    });
    const [checkpointPreview, setCheckpointPreview] = useState(null);
    const [suggestionDialog, setSuggestionDialog] = useState({ open: false, data: null, loading: false });
    const [showRationale, setShowRationale] = useState(false);
    const [isTraining, setIsTraining] = useState(false);
    const [trainingProgress, setTrainingProgress] = useState(0);
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [trainingHistory, setTrainingHistory] = useState([]);
    const [trainingStartTime, setTrainingStartTime] = useState(null);
    const [trainingError, setTrainingError] = useState(null);

    const [generationPrompt, setGenerationPrompt] = useState('');
    const [negativePrompt, setNegativePrompt] = useState('');
    const [loraStack, setLoraStack] = useState([]);   // [{path, strength}]
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
    const [availableLoras, setAvailableLoras] = useState([]);
    const [selectedLora, setSelectedLora] = useState('');
    const [loraMultiplier, setLoraMultiplier] = useState(1.0);
    const generationAbortRef = useRef(null);
    const stopGenerationRef = useRef(false);

    // Turn a LoRAW checkpoint filename like
    //   .../epoch=29-step=1410.ckpt   →   "Epoch 29 · step 1410"
    // so the checkpoint picker reads as something a musician parses, not a path.
    const parseCheckpointLabel = (filepath) => {
        const name = (filepath || '').split('/').pop() || filepath || '';
        const m = name.match(/epoch=(\d+)-step=(\d+)/);
        if (m) return `Epoch ${m[1]} · step ${m[2]}`;
        return name.replace(/\.ckpt$/i, '');
    };

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
        { name: 'sa3-small-music', displayName: 'Small - Music',     description: 'CPU/GPU · ≤ 120s',         kind: 'post-trained', downloaded: false },
        { name: 'sa3-small-sfx',   displayName: 'Small - SFX',       description: 'CPU/GPU · ≤ 120s',         kind: 'post-trained', downloaded: false },
        { name: 'sa3-medium',      displayName: 'Medium',            description: 'CUDA + Flash-Attn · ≤ 380s', kind: 'post-trained', downloaded: false },
        { name: 'sa3-small-music-base', displayName: 'Small - Music (Base)', description: 'CPU/GPU · ≤ 120s',         kind: 'base', downloaded: false },
        { name: 'sa3-small-sfx-base',   displayName: 'Small - SFX (Base)',   description: 'CPU/GPU · ≤ 120s',         kind: 'base', downloaded: false },
        { name: 'sa3-medium-base',      displayName: 'Medium (Base)',        description: 'CUDA + Flash-Attn · ≤ 380s', kind: 'base', downloaded: false },
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
    // Vertical icon-only mode: between the compact (horizontal) threshold
    // and a custom upper bound. The MUI `lg` breakpoint at 1200 was too
    // eager — labels collapsed while there was still plenty of room.
    const isIconOnlySidebar = useMediaQuery('(min-width: 900px) and (max-width: 1099.95px)');
    // Mobile/very-small width — the nav rail goes horizontal (compact)
    // AND drops the text labels, matching the icon-only treatment used
    // on mid-size vertical.
    const isMobileLayout = useMediaQuery(appTheme.breakpoints.down('sm'));
    // Dock collapses to a hamburger at the same threshold where the nav
    // rail flips horizontal — keeps the chrome transition unified.
    const isDockCollapsed = isCompactLayout;
    const [dockMenuAnchor, setDockMenuAnchor] = useState(null);

    useEffect(() => {
        setSelectedUnwrappedModel('');
    }, [selectedModel]);

    useEffect(() => {
        console.log('Model changed:', selectedModel);
        // Clear the selected LoRA on any model change — a LoRA is bound to a
        // specific base, and the dropdown re-filters by resolvedBaseModel.
        setSelectedLora('');
    }, [selectedModel]);

    // Resolve the base SA3 model identity for the currently-selected entry.
    // For a direct base pick it's selectedModel itself; for a fine-tune we
    // read base_model from the training_metadata exposed by /api/models.
    const resolvedBaseModel = (() => {
        if (!selectedModel) return null;
        if (selectedModel.startsWith('sa3-')) return selectedModel;
        const model = availableModels.find(m => m.name === selectedModel);
        return model?.base_model || null;
    })();

    // All three user-visible SA3 models are post-trained (distilled to 8
    // steps, CFG baked at 1.0). The backend ignores cfg_scale on these and
    // defaults steps to 8 — the UI just mirrors that so the controls don't
    // show misleading values.
    const isDistilledBase = !!selectedModel && selectedModel.startsWith('sa3-') && !selectedModel.endsWith('-base');

    const getMaxDuration = () => {
        if (!selectedModel) return 30;
        if (resolvedBaseModel === 'sa3-medium' || resolvedBaseModel === 'sa3-medium-base') return 380;
        if (resolvedBaseModel && resolvedBaseModel.startsWith('sa3-')) return 120;
        return 30;
    };

    useEffect(() => {
        const maxDuration = getMaxDuration();
        if (generationDuration > maxDuration) {
            setGenerationDuration(maxDuration);
        }
        // SA3 post-trained models run at 8 steps with CFG=1.0; base variants
        // want ~50 steps with CFG~7. Snap the slider so the UI reflects what
        // will actually run.
        if (isDistilledBase && steps !== 8) {
            setSteps(8);
        } else if (!isDistilledBase && steps < 50) {
            setSteps(50);
        }
    }, [selectedModel, selectedUnwrappedModel, isDistilledBase]);

    const handleTabChange = (event, newValue) => {
        if (newValue === tabValue) return;
        setTabValue(newValue);
    };

    // Sync displayedTab to tabValue with a fade-out delay so content
    // swap happens while the wrapper opacity is at 0. Works for any
    // code path that updates tabValue (Tabs click, togglePerformance,
    // model-warning auto-jump, etc).
    useEffect(() => {
        if (tabValue === displayedTab) return;
        const t = window.setTimeout(() => setDisplayedTab(tabValue), TAB_FADE_MS);
        return () => window.clearTimeout(t);
    }, [tabValue, displayedTab]);

    useEffect(() => {
        const onScroll = () => setIsScrolled(window.scrollY > 8);
        onScroll();
        window.addEventListener('scroll', onScroll, { passive: true });
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    // Re-measure header bottom edge on mount, resize, and content
    // reflows. Nav rail's `top` = headerBottom + headerRow.mb +
    // tabPanelStyles.pt so it lines up with the first card.
    useEffect(() => {
        if (!headerRef.current) return undefined;
        const el = headerRef.current;
        const measure = () => {
            // Header is sticky at top: 0, so rect.bottom is already the
            // viewport y of the header's bottom edge.
            const rect = el.getBoundingClientRect();
            const w = window.innerWidth;
            const offset = w >= 900 ? 18 : w >= 600 ? 14 : 12;
            setNavTopPx(rect.bottom + offset);
        };
        measure();
        // Re-measure only when the header's actual size changes (e.g.
        // GPU card transitions detected ↔ not on first load) or the
        // window resizes — never on scroll, never on poll churn.
        const ro = new ResizeObserver(measure);
        ro.observe(el);
        window.addEventListener('resize', measure);
        return () => {
            ro.disconnect();
            window.removeEventListener('resize', measure);
        };
    }, []);

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

    const fetchAvailableLoras = async () => {
        try {
            const response = await api.get('/api/loras');
            setAvailableLoras(response.data.loras || []);
        } catch (error) {
            console.error('Error fetching available LoRAs:', error);
        }
    };

    const fetchBaseModelsStatus = async () => {
        try {
            const response = await api.get('/api/checkpoints');
            const byId = Object.fromEntries(
                (response.data.checkpoints || []).map(c => [c.id, c])
            );
            setBaseModels(prevModels =>
                prevModels.map(model => ({
                    ...model,
                    downloaded: byId[model.name]?.downloaded || false,
                }))
            );
        } catch (error) {
            console.error('Error fetching checkpoint status:', error);
        }
    };

    // Delete a fine-tuned model OR a LoRA — both live under models/fine_tuned/<name>
    // so the same endpoint (rmtree on the directory) handles either. After
    // success, clear any selection that pointed at it and refresh both lists.
    const handleDeleteFineTunedOrLora = async (name, { isLora } = {}) => {
        const kind = isLora ? 'LoRA' : 'fine-tuned model';
        const confirmed = window.confirm(
            `Delete ${kind} "${name}"? This removes the directory and all its checkpoints. This cannot be undone.`
        );
        if (!confirmed) return;
        try {
            await api.delete(`/api/models/fine-tuned/${encodeURIComponent(name)}`);
            if (isLora) {
                // If the deleted LoRA was selected anywhere, clear it.
                const deletedLora = availableLoras.find(l => l.name === name);
                const paths = deletedLora ? (deletedLora.all_checkpoints || [deletedLora.path]) : [];
                if (paths.includes(selectedLora)) setSelectedLora('');
            } else {
                if (selectedModel === name) {
                    setSelectedModel('');
                    setSelectedUnwrappedModel('');
                }
            }
            refreshAllModels();
        } catch (err) {
            const msg = err?.response?.data?.error || err.message || 'Delete failed';
            setProcessingStatus(`Failed to delete "${name}": ${msg}`);
        }
    };

    const refreshAllModels = async () => {
        await Promise.all([
            fetchAvailableModels(),
            fetchBaseModelsStatus(),
            fetchAvailableLoras()
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
        fetchAvailableLoras();
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
        trainingConfig.steps,
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
                            // refreshAllModels picks up the new LoRA too if
                            // this was a LoRA run — without it, the LoRA
                            // picker stays empty until the user manually hits
                            // refresh.
                            refreshAllModels();
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

    const fetchHyperparamSuggestion = async () => {
        setShowRationale(false);
        setSuggestionDialog({ open: true, data: null, loading: true });
        try {
            const resp = await api.get(`/api/training/suggest-hyperparams?mode=${trainingConfig.mode}`);
            setSuggestionDialog({ open: true, data: resp.data, loading: false });
        } catch (e) {
            setSuggestionDialog({
                open: true,
                data: { ok: false, error: e?.response?.data?.error || e.message },
                loading: false,
            });
        }
    };

    const applyHyperparamSuggestion = () => {
        const cfg = suggestionDialog.data?.config;
        if (!cfg) return;
        setTrainingConfig({ ...trainingConfig, ...cfg });
        setSuggestionDialog({ open: false, data: null, loading: false });
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
            steps: steps,
        };
        const negTrim = negativePrompt.trim();
        if (negTrim) {
            baseRequestData.negative_prompt = negTrim;
        }

        // LoRA stack — only attach slots that have a path picked.
        const activeLoras = (loraStack || []).filter(s => s.path);
        if (activeLoras.length) {
            baseRequestData.loras = activeLoras.map(s => ({
                path: s.path,
                strength: s.strength,
            }));
        }
        // SA3 post-trained models bake CFG at 1.0 — only the *-base variants
        // honour cfg_scale. Sending it on a post-trained model is harmless
        // (backend forces 1.0), but we only attach it for base variants so
        // the UI matches what the backend will use.
        if (!isDistilledBase) {
            baseRequestData.cfg_scale = cfgScale;
        }

        const baseModel = baseModels.find(m => m.name === selectedModel);
        if (baseModel) {
            if (!baseModel.downloaded) {
                showModelWarning({
                    title: 'Model Not Downloaded',
                    message: `"${baseModel.displayName}" hasn't been downloaded yet. Open the Checkpoint Manager to fetch it.`,
                    canOpenModels: true,
                });
                return;
            }
            baseRequestData.model_id = selectedModel;
        } else if (selectedModel && selectedModel.startsWith('sa3-')) {
            // Hidden SA3 variant (base or AE) reachable via /api/checkpoints?include=all.
            baseRequestData.model_id = selectedModel;
        } else {
            setProcessingStatus(
                selectedModel
                    ? `'${selectedModel}' is an SA2 fine-tune; SA3 cannot load it. Pick a Stable Audio 3 model.`
                    : 'Please select a model'
            );
            return;
        }

        // LoRA stacking is Phase 4; ignore selectedLora for now.

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
        setCheckpointMgrOpen(true);
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
                    <Box ref={headerRef} sx={[appStyles.headerRow, isScrolled && appStyles.headerRowScrolled]}>
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
                            <Paper sx={appStyles.gpuCard(isCompactLayout)}>
                                {gpuMemoryStatus && gpuMemoryStatus.cuda ? (
                                    <>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', mb: 0.75 }}>
                                            <Typography variant="overline" color="text.secondary" sx={{ fontSize: '0.65rem', lineHeight: 1 }}>
                                                GPU
                                            </Typography>
                                            <Typography variant="caption" sx={{ color: 'primary.main', fontWeight: 600, fontSize: '0.72rem' }}>
                                                {gpuMemoryStatus.cuda.free.toFixed(1)} / {gpuMemoryStatus.cuda.total.toFixed(0)} GB free
                                            </Typography>
                                        </Box>
                                        <Box sx={{ height: 4, borderRadius: 999, bgcolor: 'rgba(255, 255, 255, 0.08)', overflow: 'hidden' }}>
                                            <Box
                                                sx={{
                                                    height: '100%',
                                                    width: `${Math.min(Math.max(((gpuMemoryStatus.cuda.total - gpuMemoryStatus.cuda.free) / gpuMemoryStatus.cuda.total) * 100, 0), 100)}%`,
                                                    bgcolor: 'primary.main',
                                                    transition: 'width 0.3s ease',
                                                }}
                                            />
                                        </Box>
                                    </>
                                ) : (
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                                        <Typography variant="overline" color="text.secondary" sx={{ fontSize: '0.65rem', lineHeight: 1 }}>
                                            GPU
                                        </Typography>
                                        <Typography variant="caption" color="warning.main" sx={{ fontSize: '0.72rem' }}>
                                            Not detected · CPU mode
                                        </Typography>
                                    </Box>
                                )}
                            </Paper>
                        </Box>
                    </Box>

                    {/* Main Content with Sidebar Layout */}
                    <Box sx={appStyles.mainLayout(isCompactLayout, isIconOnlySidebar)}>
                        {/* Left Sidebar with Vertical Tabs */}
                        <Paper sx={[appStyles.navPaper(isCompactLayout, isIconOnlySidebar), !isCompactLayout && { top: `${navTopPx}px` }]}>
                            <Tabs
                                value={tabValue}
                                onChange={handleTabChange}
                                orientation={isCompactLayout ? 'horizontal' : 'vertical'}
                                aria-label="main navigation tabs"
                                sx={appStyles.navigationTabs(isCompactLayout, isIconOnlySidebar)}
                            >
                                <Tab icon={<UploadIcon size={20} />} iconPosition={isIconOnlySidebar ? 'top' : 'start'} label={(isIconOnlySidebar || isMobileLayout) ? undefined : 'Data Processing'} />
                                <Tab icon={<ActivityIcon size={20} />} iconPosition={isIconOnlySidebar ? 'top' : 'start'} label={(isIconOnlySidebar || isMobileLayout) ? undefined : 'Training'} />
                                <Tab icon={<SparklesIcon size={20} />} iconPosition={isIconOnlySidebar ? 'top' : 'start'} label={(isIconOnlySidebar || isMobileLayout) ? undefined : 'Generation'} />
                                <Tab
                                    icon={<PerformanceIcon size={20} />}
                                    iconPosition={isIconOnlySidebar ? 'top' : 'start'}
                                    label={(isIconOnlySidebar || isMobileLayout) ? undefined : (
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
                        <Box sx={appStyles.mainContentBox}>
                            <Box
                                sx={{
                                    flex: 1,
                                    display: 'flex',
                                    flexDirection: 'column',
                                    minHeight: 0,
                                    opacity: tabValue === displayedTab ? 1 : 0,
                                    transition: `opacity ${TAB_FADE_MS}ms ease`,
                                }}
                            >

                            {/* Data Processing Tab */}
                            <TabPanel value={displayedTab} index={0}>
                                <Grid container spacing={{ xs: 2, sm: 2.5, md: 3 }} sx={appStyles.dataProcessingGrid}>
                                    <Grid item xs={12} md={8} sx={appStyles.primaryPaneItem}>
                                        <Box sx={appStyles.primaryPaneContent}>
                                            <CsvImportPanel key={`csv-${uploadKey}`} onCommitted={fetchSystemStatus} />

                                            <BulkAnnotatePanel key={`bulk-${uploadKey}`} onCommitted={fetchSystemStatus} />

                                            <Paper sx={{ p: { xs: 2.25, sm: 3 }, borderRadius: 2.5 }} variant="outlined">
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

                                                <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', mt: 1 }}>
                                                    <Button
                                                        variant="outlined"
                                                        startIcon={<AddIcon />}
                                                        onClick={addUploadRow}
                                                    >
                                                        Add Another Row
                                                    </Button>
                                                    <Box sx={{ flex: 1 }} />
                                                    <Button
                                                        variant="contained"
                                                        onClick={processFiles}
                                                        disabled={isProcessing || uploadRows.every((r) => !r.file)}
                                                        startIcon={isProcessing ? <CircularProgress size={20} /> : <UploadIcon />}
                                                    >
                                                        {isProcessing ? 'Saving…' : 'Save to dataset'}
                                                    </Button>
                                                </Box>
                                            </Paper>
                                        </Box>
                                    </Grid>

                                    <Grid item xs={12} md={4}>

                                        {!systemStatus && (
                                            <Paper sx={[appStyles.elevatedInfoCard, appStyles.datasetStatusSticky(navTopPx)]}>
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
                                            <Paper sx={[appStyles.elevatedInfoCard, appStyles.datasetStatusSticky(navTopPx)]}>
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
                            <TabPanel value={displayedTab} index={1}>
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

                                                <Box sx={{ mb: 2 }}>
                                                    <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mb: 0.5 }}>
                                                        Training mode
                                                    </Typography>
                                                    <ToggleButtonGroup
                                                        value={trainingConfig.mode}
                                                        exclusive
                                                        size="small"
                                                        onChange={(e, newMode) => {
                                                            if (newMode !== null) {
                                                                setTrainingConfig({ ...trainingConfig, mode: newMode });
                                                            }
                                                        }}
                                                        fullWidth
                                                    >
                                                        <Tooltip title="LoRA adapter — small (~50 MB) trainable layer attached to the frozen base model. Works on 16 GB cards. Recommended for most use cases.">
                                                            <ToggleButton value="lora">
                                                                LoRA Adapter
                                                            </ToggleButton>
                                                        </Tooltip>
                                                        <Tooltip title="Full fine-tune — rewrites the entire base model. Produces a ~5 GB checkpoint. Requires ≥24 GB VRAM for the large base.">
                                                            <ToggleButton value="full">
                                                                Full Fine-tune
                                                            </ToggleButton>
                                                        </Tooltip>
                                                    </ToggleButtonGroup>
                                                </Box>

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

                                                <Box sx={appStyles.fieldMarginBottom}>
                                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                                                        Base model to fine-tune
                                                    </Typography>
                                                    <Select
                                                        fullWidth
                                                        size="small"
                                                        value={trainingConfig.baseModel}
                                                        onChange={(e) => setTrainingConfig({
                                                            ...trainingConfig,
                                                            baseModel: e.target.value,
                                                        })}
                                                    >
                                                        {/* LoRA training requires CFG-aware *-base checkpoints —
                                                            post-trained models have CFG distilled out and
                                                            can't be trained against. */}
                                                        {baseModels
                                                            .filter(m => m.name.endsWith('-base'))
                                                            .map(m => (
                                                                <MenuItem key={m.name} value={m.name}>
                                                                    <Box>
                                                                        <Typography variant="body2">
                                                                            {m.displayName}
                                                                        </Typography>
                                                                        <Typography variant="caption" color="text.secondary">
                                                                            {m.description}
                                                                            {!m.downloaded && ' · not yet downloaded (will fetch on training start)'}
                                                                        </Typography>
                                                                    </Box>
                                                                </MenuItem>
                                                            ))}
                                                    </Select>
                                                </Box>

                                                <Accordion>
                                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                                        <Typography variant="subtitle1">Advanced Settings</Typography>
                                                    </AccordionSummary>
                                                    <AccordionDetails sx={appStyles.advancedSettingsDetails}>
                                                        <Grid container spacing={{ xs: 2, sm: 2.5, md: 3 }}>
                                                            <Grid item xs={12}>
                                                                <Typography gutterBottom>Training Steps</Typography>
                                                                <Box sx={appStyles.sliderRow}>
                                                                    <Slider
                                                                        value={trainingConfig.steps}
                                                                        onChange={(e, value) => setTrainingConfig({
                                                                            ...trainingConfig,
                                                                            steps: value
                                                                        })}
                                                                        min={500}
                                                                        max={20000}
                                                                        step={500}
                                                                        marks={[
                                                                            { value: 1000, label: '1k' },
                                                                            { value: 5000, label: '5k' },
                                                                            { value: 10000, label: '10k' },
                                                                            { value: 20000, label: '20k' },
                                                                        ]}
                                                                        valueLabelDisplay="auto"
                                                                        sx={appStyles.sliderFlexGrow}
                                                                    />
                                                                    <TextField
                                                                        type="number"
                                                                        value={trainingConfig.steps}
                                                                        onChange={(e) => {
                                                                            const val = parseInt(e.target.value) || 500;
                                                                            setTrainingConfig({
                                                                                ...trainingConfig,
                                                                                steps: Math.max(100, Math.min(50000, val))
                                                                            });
                                                                        }}
                                                                        inputProps={{ min: 100, max: 50000, step: 100 }}
                                                                        sx={appStyles.sliderInputSmall}
                                                                        size="small"
                                                                    />
                                                                </Box>
                                                                <Typography variant="caption" color="text.secondary">
                                                                    SA3 LoRAs typically converge in 2 000 – 10 000 steps depending on dataset size.
                                                                </Typography>
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Typography gutterBottom>Adapter Type</Typography>
                                                                <Select
                                                                    fullWidth
                                                                    size="small"
                                                                    value={trainingConfig.adapterType || 'dora-rows'}
                                                                    onChange={(e) => setTrainingConfig({
                                                                        ...trainingConfig,
                                                                        adapterType: e.target.value,
                                                                    })}
                                                                >
                                                                    <MenuItem value="dora-rows">DoRA-rows (recommended)</MenuItem>
                                                                    <MenuItem value="dora-cols">DoRA-cols</MenuItem>
                                                                    <MenuItem value="lora">LoRA (classic)</MenuItem>
                                                                    <MenuItem value="bora">BoRA</MenuItem>
                                                                    <MenuItem value="lora-xs">LoRA-XS (compact)</MenuItem>
                                                                    <MenuItem value="dora-rows-xs">DoRA-rows-XS (compact)</MenuItem>
                                                                    <MenuItem value="dora-cols-xs">DoRA-cols-XS (compact)</MenuItem>
                                                                    <MenuItem value="bora-xs">BoRA-XS (compact)</MenuItem>
                                                                </Select>
                                                                <Typography variant="caption" color="text.secondary">
                                                                    DoRA-rows is upstream's default and works best for most stylistic LoRAs.
                                                                </Typography>
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

                                                            {trainingConfig.mode === 'lora' && (
                                                                <Grid item xs={12}>
                                                                    <Typography variant="subtitle2" color="textSecondary" sx={{ mt: 1, mb: 1 }}>
                                                                        LoRA settings
                                                                    </Typography>

                                                                    <Typography gutterBottom>Rank</Typography>
                                                                    <Box sx={appStyles.sliderRow}>
                                                                        <Slider
                                                                            value={trainingConfig.loraRank}
                                                                            onChange={(e, value) => setTrainingConfig({
                                                                                ...trainingConfig,
                                                                                loraRank: value,
                                                                                // Keep alpha == rank by default (common LoRA practice)
                                                                                ...(trainingConfig.loraAlpha === trainingConfig.loraRank
                                                                                    ? { loraAlpha: value } : {}),
                                                                            })}
                                                                            min={4}
                                                                            max={128}
                                                                            step={4}
                                                                            marks
                                                                            valueLabelDisplay="auto"
                                                                            sx={appStyles.sliderFlexGrow}
                                                                        />
                                                                        <TextField
                                                                            type="number"
                                                                            value={trainingConfig.loraRank}
                                                                            onChange={(e) => {
                                                                                const v = Math.max(4, Math.min(128, parseInt(e.target.value, 10) || 16));
                                                                                setTrainingConfig({ ...trainingConfig, loraRank: v });
                                                                            }}
                                                                            inputProps={{ min: 4, max: 128, step: 4 }}
                                                                            sx={appStyles.sliderInputSmall}
                                                                            size="small"
                                                                        />
                                                                    </Box>
                                                                    <Typography variant="caption" color="textSecondary">
                                                                        Higher rank = more capacity but more VRAM. r=16 fits comfortably on 16 GB.
                                                                    </Typography>

                                                                    <Typography gutterBottom sx={{ mt: 2 }}>Alpha</Typography>
                                                                    <Box sx={appStyles.sliderRow}>
                                                                        <Slider
                                                                            value={trainingConfig.loraAlpha}
                                                                            onChange={(e, value) => setTrainingConfig({
                                                                                ...trainingConfig,
                                                                                loraAlpha: value,
                                                                            })}
                                                                            min={4}
                                                                            max={256}
                                                                            step={4}
                                                                            valueLabelDisplay="auto"
                                                                            sx={appStyles.sliderFlexGrow}
                                                                        />
                                                                        <TextField
                                                                            type="number"
                                                                            value={trainingConfig.loraAlpha}
                                                                            onChange={(e) => {
                                                                                const v = Math.max(4, Math.min(256, parseInt(e.target.value, 10) || 16));
                                                                                setTrainingConfig({ ...trainingConfig, loraAlpha: v });
                                                                            }}
                                                                            inputProps={{ min: 4, max: 256, step: 4 }}
                                                                            sx={appStyles.sliderInputSmall}
                                                                            size="small"
                                                                        />
                                                                    </Box>
                                                                    <Typography variant="caption" color="textSecondary">
                                                                        Scaling factor for the LoRA update. Conventional choice: alpha = rank.
                                                                    </Typography>

                                                                    <Typography gutterBottom sx={{ mt: 2 }}>Dropout</Typography>
                                                                    <Box sx={appStyles.sliderRow}>
                                                                        <Slider
                                                                            value={trainingConfig.loraDropout}
                                                                            onChange={(e, value) => setTrainingConfig({
                                                                                ...trainingConfig,
                                                                                loraDropout: value,
                                                                            })}
                                                                            min={0}
                                                                            max={0.5}
                                                                            step={0.05}
                                                                            valueLabelDisplay="auto"
                                                                            sx={appStyles.sliderFlexGrow}
                                                                        />
                                                                        <TextField
                                                                            type="number"
                                                                            value={trainingConfig.loraDropout}
                                                                            onChange={(e) => {
                                                                                const v = Math.max(0, Math.min(0.5, parseFloat(e.target.value) || 0));
                                                                                setTrainingConfig({ ...trainingConfig, loraDropout: v });
                                                                            }}
                                                                            inputProps={{ min: 0, max: 0.5, step: 0.05 }}
                                                                            sx={appStyles.sliderInputSmall}
                                                                            size="small"
                                                                        />
                                                                    </Box>
                                                                    <Typography variant="caption" color="textSecondary">
                                                                        Regularization for the LoRA layers. 0 is fine for most cases; raise if overfitting on small datasets.
                                                                    </Typography>
                                                                </Grid>
                                                            )}

                                                        </Grid>
                                                    </AccordionDetails>
                                                </Accordion>



                                                <Box sx={{ mt: 1.5, mb: 1.5 }}>
                                                    <Button
                                                        variant="contained"
                                                        color="warm"
                                                        fullWidth
                                                        onClick={fetchHyperparamSuggestion}
                                                        disabled={isTraining}
                                                        startIcon={<WandIcon size={16} />}
                                                    >
                                                        Suggest hyperparameters for my dataset
                                                    </Button>
                                                </Box>

                                                <Box sx={appStyles.trainingActionRow}>
                                                    <Button
                                                        variant="contained"
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
                                            <Box sx={[appStyles.trainingMonitorWrap, appStyles.datasetStatusSticky(navTopPx)]}>
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
                            <TabPanel value={displayedTab} index={2}>
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
                                                            open={generationModelSelectOpen}
                                                            onOpen={() => setGenerationModelSelectOpen(true)}
                                                            onClose={() => setGenerationModelSelectOpen(false)}
                                                            onChange={(event) => {
                                                                console.log('Model dropdown selected:', event.target.value, typeof event.target.value);
                                                                handleModelChange(event);
                                                            }}
                                                            displayEmpty
                                                        >
                                                            <MenuItem value="" disabled>
                                                                <em>Select a model</em>
                                                            </MenuItem>
                                                            {[
                                                                { kind: 'post-trained', label: '── Distilled · fixed cfg + steps (fast) ──' },
                                                                { kind: 'base',         label: '── Base · cfg + steps live ──' },
                                                            ].flatMap(group => {
                                                                const rows = baseModels.filter(m => m.kind === group.kind);
                                                                if (!rows.length) return [];
                                                                return [
                                                                    <MenuItem key={`hdr-${group.kind}`} disabled>
                                                                        <Typography variant="subtitle2" color="textSecondary">
                                                                            {group.label}
                                                                        </Typography>
                                                                    </MenuItem>,
                                                                    ...rows.map(model => (
                                                                        <MenuItem
                                                                            key={model.name}
                                                                            value={String(model.name)}
                                                                            disabled={!model.downloaded}
                                                                            sx={{
                                                                                display: 'flex',
                                                                                alignItems: 'center',
                                                                                gap: 1,
                                                                                // Disabled MenuItems get opacity from MUI; the
                                                                                // download IconButton needs to stay clickable
                                                                                // (and look it), so re-enable pointer events on
                                                                                // the action slot and lift its opacity.
                                                                                '&.Mui-disabled': { pointerEvents: 'auto' },
                                                                            }}
                                                                        >
                                                                            <Box sx={{ flex: 1, minWidth: 0 }}>
                                                                                <Typography variant="body1">{model.displayName}</Typography>
                                                                                <Typography variant="caption" color="textSecondary">
                                                                                    {model.description}
                                                                                </Typography>
                                                                            </Box>
                                                                            {!model.downloaded && (
                                                                                <Tooltip title="Download this model">
                                                                                    <IconButton
                                                                                        size="small"
                                                                                        onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); }}
                                                                                        onClick={(e) => {
                                                                                            e.stopPropagation();
                                                                                            e.preventDefault();
                                                                                            setGenerationModelSelectOpen(false);
                                                                                            setCheckpointMgrOpen(true);
                                                                                        }}
                                                                                        sx={{ opacity: 1, color: 'primary.main' }}
                                                                                    >
                                                                                        <CloudDownloadIcon size={16} />
                                                                                    </IconButton>
                                                                                </Tooltip>
                                                                            )}
                                                                        </MenuItem>
                                                                    )),
                                                                ];
                                                            })}
                                                            {/* Fine-tuned Models Section */}
                                                            {availableModels.length > 0 && (
                                                                <MenuItem disabled>
                                                                    <Typography variant="subtitle2" color="textSecondary">
                                                                        ── Fine-tuned Models ──
                                                                    </Typography>
                                                                </MenuItem>
                                                            )}
                                                            {availableModels.map((model) => (
                                                                <MenuItem
                                                                    key={model.name}
                                                                    value={String(model.name)}
                                                                    disabled={false}
                                                                    sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, pr: 0.5 }}
                                                                >
                                                                    <Box sx={{ flex: 1, minWidth: 0 }}>
                                                                        <Typography variant="body1">{model.name}</Typography>
                                                                        <Typography variant="caption" color="textSecondary">
                                                                            {model.has_checkpoint ? 'Checkpoint' : 'No Checkpoint'} |
                                                                            {model.unwrapped_models && model.unwrapped_models.length > 0
                                                                                ? ` ${model.unwrapped_models.length} unwrapped models`
                                                                                : ' No unwrapped models'}
                                                                        </Typography>
                                                                    </Box>
                                                                    <Tooltip title="Delete fine-tuned model">
                                                                        <IconButton
                                                                            size="small"
                                                                            onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); }}
                                                                            onClick={(e) => {
                                                                                e.stopPropagation();
                                                                                e.preventDefault();
                                                                                handleDeleteFineTunedOrLora(model.name);
                                                                            }}
                                                                            sx={{
                                                                                color: 'text.disabled',
                                                                                '&:hover': { color: 'error.main', bgcolor: 'action.hover' },
                                                                            }}
                                                                        >
                                                                            <DeleteIcon size={14} />
                                                                        </IconButton>
                                                                    </Tooltip>
                                                                </MenuItem>
                                                            ))}
                                                        </Select>
                                                    </FormControl>
                                                    <IconButton
                                                        onClick={refreshAllModels}
                                                        title="Refresh models & LoRAs"
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

                                                {/* LoRA picker — only meaningful on a base model. Filters to
                                                    LoRAs trained against the currently-selected base.
                                                    Two-step: pick LoRA name, then pick which saved checkpoint
                                                    of that LoRA to load (defaults to latest). */}
                                                {baseModels.find(m => m.name === selectedModel) && (() => {
                                                    const compatibleLoras = availableLoras.filter(
                                                        l => l.base_model === selectedModel
                                                    );
                                                    if (compatibleLoras.length === 0) return null;
                                                    // Derive which LoRA the current checkpoint path belongs to,
                                                    // so the dropdown's `value` stays in sync after the second
                                                    // picker mutates `selectedLora`.
                                                    const currentLora = compatibleLoras.find(
                                                        l => l.path === selectedLora ||
                                                             (l.all_checkpoints || []).includes(selectedLora)
                                                    );
                                                    const currentLoraName = currentLora?.name || '';
                                                    return (
                                                        <>
                                                            <FormControl fullWidth sx={appStyles.formControlMarginBottom} variant="outlined">
                                                                <Select
                                                                    labelId="lora-select-label"
                                                                    id="lora-select"
                                                                    value={currentLoraName}
                                                                    label="Select LoRA (optional)"
                                                                    onChange={(e) => {
                                                                        const name = e.target.value;
                                                                        if (!name) { setSelectedLora(''); return; }
                                                                        const lora = compatibleLoras.find(l => l.name === name);
                                                                        // Default to latest checkpoint when picking a LoRA.
                                                                        setSelectedLora(lora?.path || '');
                                                                    }}
                                                                    displayEmpty
                                                                >
                                                                    <MenuItem value="">
                                                                        <em>No LoRA (base model only)</em>
                                                                    </MenuItem>
                                                                    {compatibleLoras.map((lora) => (
                                                                        <MenuItem
                                                                            key={lora.name}
                                                                            value={lora.name}
                                                                            sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, pr: 0.5 }}
                                                                        >
                                                                            <Box sx={{ flex: 1, minWidth: 0 }}>
                                                                                <Typography variant="body1">{lora.name}</Typography>
                                                                                <Typography variant="caption" color="textSecondary">
                                                                                    rank={lora.rank}, alpha={lora.alpha}
                                                                                    {lora.all_checkpoints?.length > 1
                                                                                        ? ` · ${lora.all_checkpoints.length} checkpoints`
                                                                                        : ''}
                                                                                </Typography>
                                                                            </Box>
                                                                            <Tooltip title="Delete LoRA">
                                                                                <IconButton
                                                                                    size="small"
                                                                                    onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); }}
                                                                                    onClick={(e) => {
                                                                                        e.stopPropagation();
                                                                                        e.preventDefault();
                                                                                        handleDeleteFineTunedOrLora(lora.name, { isLora: true });
                                                                                    }}
                                                                                    sx={{
                                                                                        color: 'text.disabled',
                                                                                        '&:hover': { color: 'error.main', bgcolor: 'action.hover' },
                                                                                    }}
                                                                                >
                                                                                    <DeleteIcon size={14} />
                                                                                </IconButton>
                                                                            </Tooltip>
                                                                        </MenuItem>
                                                                    ))}
                                                                </Select>
                                                            </FormControl>

                                                            {/* Second picker: which checkpoint of the chosen LoRA */}
                                                            {currentLora && currentLora.all_checkpoints?.length > 1 && (
                                                                <FormControl fullWidth sx={appStyles.formControlMarginBottom} variant="outlined">
                                                                    <Select
                                                                        labelId="lora-checkpoint-select-label"
                                                                        id="lora-checkpoint-select"
                                                                        value={selectedLora || currentLora.path}
                                                                        label="Checkpoint"
                                                                        onChange={(e) => setSelectedLora(String(e.target.value))}
                                                                    >
                                                                        {currentLora.all_checkpoints.map((ckpt, i, arr) => (
                                                                            <MenuItem key={ckpt} value={ckpt}>
                                                                                <Typography variant="body2">
                                                                                    {parseCheckpointLabel(ckpt)}
                                                                                    {i === arr.length - 1 ? ' (latest)' : ''}
                                                                                </Typography>
                                                                            </MenuItem>
                                                                        ))}
                                                                    </Select>
                                                                </FormControl>
                                                            )}
                                                        </>
                                                    );
                                                })()}

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

                                                <Accordion>
                                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                                        <Typography variant="subtitle1">Advanced Settings</Typography>
                                                    </AccordionSummary>
                                                    <AccordionDetails sx={appStyles.advancedSettingsDetails}>
                                                        <Grid container spacing={{ xs: 2, sm: 2.5, md: 3 }}>
                                                            <Grid item xs={12}>
                                                                <TextField
                                                                    fullWidth
                                                                    multiline
                                                                    minRows={1}
                                                                    maxRows={3}
                                                                    label="Negative Prompt (optional)"
                                                                    placeholder="What to avoid: vocals, distortion, silence..."
                                                                    value={negativePrompt}
                                                                    onChange={(e) => setNegativePrompt(e.target.value)}
                                                                />
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <LoraStack
                                                                    selectedModel={selectedModel}
                                                                    value={loraStack}
                                                                    onChange={setLoraStack}
                                                                />
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Accordion>
                                                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                                                        <Typography variant="subtitle1">Edit existing audio</Typography>
                                                                        <Typography variant="caption" color="textSecondary" sx={{ ml: 1, alignSelf: 'center' }}>
                                                                            Style transfer · Inpaint · Extend
                                                                        </Typography>
                                                                    </AccordionSummary>
                                                                    <AccordionDetails sx={{ p: 0 }}>
                                                                        <EditPanel
                                                                            model_id={selectedModel}
                                                                            negativePrompt={negativePrompt}
                                                                            onGenerated={(blob, filename, params) => {
                                                                                const audioUrl = URL.createObjectURL(blob);
                                                                                setGeneratedFragments(prev => [
                                                                                    ...prev,
                                                                                    {
                                                                                        id: Date.now(),
                                                                                        prompt: params.prompt,
                                                                                        duration: params.duration,
                                                                                        cfgScale: params.cfg_scale,
                                                                                        steps: params.steps,
                                                                                        seed: params.seed,
                                                                                        batchIndex: 1,
                                                                                        batchTotal: 1,
                                                                                        audioUrl,
                                                                                        audioBlob: blob,
                                                                                        filename,
                                                                                        timestamp: new Date().toLocaleString(),
                                                                                        editMode: params.init_audio_path ? 'style' : params.inpaint_audio_path ? 'inpaint/extend' : null,
                                                                                    },
                                                                                ]);
                                                                            }}
                                                                        />
                                                                    </AccordionDetails>
                                                                </Accordion>
                                                            </Grid>

                                                            {/* CFG + Steps are only meaningful on *-base checkpoints.
                                                                Distilled post-trained models bake cfg=1.0 / steps=8 and
                                                                ignore overrides, so we hide the controls entirely. */}
                                                            {!isDistilledBase && (
                                                                <>
                                                                    <Grid item xs={12}>
                                                                        <Typography gutterBottom>CFG Scale</Typography>
                                                                        <Box sx={appStyles.sliderRow}>
                                                                            <Slider
                                                                                value={cfgScale}
                                                                                onChange={(e, value) => setCfgScale(value)}
                                                                                min={0.1}
                                                                                max={20}
                                                                                step={0.1}
                                                                                valueLabelDisplay="auto"
                                                                                sx={appStyles.sliderFlexGrow}
                                                                            />
                                                                            <TextField
                                                                                type="number"
                                                                                value={cfgScale}
                                                                                onChange={(e) => {
                                                                                    const val = parseFloat(e.target.value);
                                                                                    if (Number.isNaN(val)) return;
                                                                                    setCfgScale(Math.max(0.1, Math.min(20, val)));
                                                                                }}
                                                                                inputProps={{ min: 0.1, max: 20, step: 0.1 }}
                                                                                sx={appStyles.sliderInputSmall}
                                                                                size="small"
                                                                            />
                                                                        </Box>
                                                                    </Grid>

                                                                    <Grid item xs={12}>
                                                                        <Typography gutterBottom>Inference Steps</Typography>
                                                                        <Box sx={appStyles.sliderRow}>
                                                                            <Slider
                                                                                value={steps}
                                                                                onChange={(e, value) => setSteps(value)}
                                                                                min={20}
                                                                                max={250}
                                                                                step={null}
                                                                                marks={[
                                                                                    { value: 20, label: '20' },
                                                                                    { value: 50, label: '50' },
                                                                                    { value: 100, label: '100' },
                                                                                    { value: 150, label: '150' },
                                                                                    { value: 200, label: '200' },
                                                                                    { value: 250, label: '250' },
                                                                                ]}
                                                                                valueLabelDisplay="auto"
                                                                                sx={appStyles.sliderFlexGrow}
                                                                            />
                                                                        </Box>
                                                                    </Grid>
                                                                </>
                                                            )}

                                                            {selectedLora && (
                                                                <Grid item xs={12}>
                                                                    <Typography gutterBottom>LoRA Multiplier</Typography>
                                                                    <Box sx={appStyles.sliderRow}>
                                                                        <Slider
                                                                            value={loraMultiplier}
                                                                            onChange={(e, v) => setLoraMultiplier(v)}
                                                                            min={0}
                                                                            max={2}
                                                                            step={0.05}
                                                                            valueLabelDisplay="auto"
                                                                            sx={appStyles.sliderFlexGrow}
                                                                        />
                                                                        <TextField
                                                                            type="number"
                                                                            value={loraMultiplier}
                                                                            onChange={(e) => {
                                                                                const val = parseFloat(e.target.value);
                                                                                if (Number.isNaN(val)) return;
                                                                                setLoraMultiplier(Math.max(0, Math.min(2, val)));
                                                                            }}
                                                                            inputProps={{ min: 0, max: 2, step: 0.05 }}
                                                                            sx={appStyles.sliderInputSmall}
                                                                            size="small"
                                                                        />
                                                                    </Box>
                                                                </Grid>
                                                            )}

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
                                            <Box sx={appStyles.datasetStatusSticky(navTopPx)}>
                                                <GeneratedFragmentsWindow
                                                    fragments={generatedFragments}
                                                    onDownload={downloadFragment}
                                                />
                                            </Box>
                                        </Box>
                                    </Grid>
                                </Grid>
                            </TabPanel>

                            <TabPanel value={displayedTab} index={3} keepMounted>
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
                                            availableLoras={availableLoras}
                                            selectedLora={selectedLora}
                                            loraMultiplier={loraMultiplier}
                                            onSelectModel={setSelectedModel}
                                            onSelectUnwrappedModel={setSelectedUnwrappedModel}
                                            onRefreshModels={refreshAllModels}
                                            onSelectLora={setSelectedLora}
                                            onLoraMultiplierChange={setLoraMultiplier}
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
                                            Performance mode is turned off. Toggle on from the {(isIconOnlySidebar || isMobileLayout) ? 'bottom-left menu' : 'sidebar'} if you wish to enter performance mode.
                                        </Typography>
                                    </Box>
                                )}
                            </TabPanel>
                            </Box>
                        </Box>
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

            <Snackbar
                open={Boolean(processingStatus)}
                autoHideDuration={10000}
                onClose={(_e, reason) => { if (reason !== 'clickaway') setProcessingStatus(''); }}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            >
                <Alert
                    onClose={() => setProcessingStatus('')}
                    severity={
                        /error|failed/i.test(processingStatus) ? 'error'
                        : /completed|success/i.test(processingStatus) ? 'success'
                        : 'info'
                    }
                    variant="filled"
                    sx={{ minWidth: 280, boxShadow: 6 }}
                >
                    {processingStatus}
                </Alert>
            </Snackbar>

            {isDockCollapsed ? (
                <>
                    <IconButton
                        aria-label="Open actions menu"
                        onClick={(e) => setDockMenuAnchor(e.currentTarget)}
                        sx={appStyles.dockHamburger}
                    >
                        <MenuIcon size={18} />
                    </IconButton>
                    <Menu
                        anchorEl={dockMenuAnchor}
                        open={Boolean(dockMenuAnchor)}
                        onClose={() => setDockMenuAnchor(null)}
                        anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
                        transformOrigin={{ vertical: 'bottom', horizontal: 'left' }}
                    >
                        <MenuItem
                            onClick={() => { setDockMenuAnchor(null); setCheckpointMgrOpen(true); }}
                        >
                            <ListItemIcon><CloudDownloadIcon size={18} /></ListItemIcon>
                            <ListItemText>Get Models</ListItemText>
                        </MenuItem>
                        <MenuItem
                            onClick={() => { setDockMenuAnchor(null); handleOpenOutputFolder(); }}
                        >
                            <ListItemIcon><FolderOpenIcon size={18} /></ListItemIcon>
                            <ListItemText>Outputs</ListItemText>
                        </MenuItem>
                        <MenuItem
                            onClick={() => { setDockMenuAnchor(null); setShowFreeGPUDialog(true); }}
                            disabled={isFreeingGPU || !(gpuMemoryStatus && gpuMemoryStatus.cuda)}
                        >
                            <ListItemIcon>
                                {isFreeingGPU ? <CircularProgress size={16} color="inherit" /> : <RefreshIcon size={18} />}
                            </ListItemIcon>
                            <ListItemText>{isFreeingGPU ? 'Freeing…' : 'Free GPU'}</ListItemText>
                        </MenuItem>
                        <MenuItem
                            onClick={() => { setDockMenuAnchor(null); setShowStartFreshDialog(true); }}
                            disabled={isStartingFresh}
                            sx={{ color: 'error.main', '& .MuiListItemIcon-root': { color: 'inherit' } }}
                        >
                            <ListItemIcon>
                                {isStartingFresh ? <CircularProgress size={16} color="inherit" /> : <DeleteIcon size={18} />}
                            </ListItemIcon>
                            <ListItemText>{isStartingFresh ? 'Starting…' : 'Fresh Start'}</ListItemText>
                        </MenuItem>
                        <Divider />
                        {(isIconOnlySidebar || isMobileLayout) && (
                            <MenuItem
                                onClick={() => { setDockMenuAnchor(null); togglePerformance(); }}
                                sx={performanceEnabled
                                    ? { color: 'warm.main', '& .MuiListItemIcon-root': { color: 'inherit' } }
                                    : undefined}
                            >
                                <ListItemIcon><PerformanceIcon size={18} /></ListItemIcon>
                                <ListItemText>{performanceEnabled ? 'Performance: On' : 'Performance: Off'}</ListItemText>
                            </MenuItem>
                        )}
                        <MenuItem
                            onClick={() => { setDockMenuAnchor(null); toggleColorMode(); }}
                        >
                            <ListItemIcon>
                                {colorMode === 'light' ? <MoonIcon size={18} /> : <SunIcon size={18} />}
                            </ListItemIcon>
                            <ListItemText>{colorMode === 'light' ? 'Dark Mode' : 'Light Mode'}</ListItemText>
                        </MenuItem>
                        <MenuItem
                            onClick={() => { setDockMenuAnchor(null); setShowInfoDialog(true); }}
                        >
                            <ListItemIcon><InfoIcon size={18} /></ListItemIcon>
                            <ListItemText>About</ListItemText>
                        </MenuItem>
                    </Menu>
                </>
            ) : (
                <Paper sx={appStyles.bottomDock}>
                    {(isIconOnlySidebar || isMobileLayout) && (
                        <Box sx={appStyles.dockItem}>
                            <IconButton
                                aria-label={performanceEnabled ? 'Disable performance mode' : 'Enable performance mode'}
                                onClick={togglePerformance}
                                sx={performanceEnabled
                                    ? [appStyles.dockIconButton, { color: 'warm.main', '&:hover': { color: 'warm.main' } }]
                                    : appStyles.dockIconButton}
                            >
                                <PerformanceIcon size={18} />
                            </IconButton>
                            <Typography className="dock-label" sx={appStyles.dockLabel}>
                                {performanceEnabled ? 'Performance: On' : 'Performance: Off'}
                            </Typography>
                        </Box>
                    )}

                    <Box sx={appStyles.dockItem}>
                        <IconButton
                            aria-label="Get models"
                            onClick={() => setCheckpointMgrOpen(true)}
                            sx={appStyles.dockIconButton}
                        >
                            <CloudDownloadIcon size={18} />
                        </IconButton>
                        <Typography className="dock-label" sx={appStyles.dockLabel}>
                            Get Models
                        </Typography>
                    </Box>

                    <Box sx={appStyles.dockItem}>
                        <IconButton
                            aria-label="Open outputs folder"
                            onClick={handleOpenOutputFolder}
                            sx={appStyles.dockIconButton}
                        >
                            <FolderOpenIcon size={18} />
                        </IconButton>
                        <Typography className="dock-label" sx={appStyles.dockLabel}>
                            Outputs
                        </Typography>
                    </Box>

                    <Box sx={appStyles.dockItem}>
                        <IconButton
                            aria-label="Free GPU memory"
                            onClick={() => setShowFreeGPUDialog(true)}
                            disabled={isFreeingGPU || !(gpuMemoryStatus && gpuMemoryStatus.cuda)}
                            sx={[appStyles.dockIconButton, appStyles.dockIconButtonAccent]}
                        >
                            {isFreeingGPU ? <CircularProgress size={16} color="inherit" /> : <RefreshIcon size={18} />}
                        </IconButton>
                        <Typography className="dock-label" sx={appStyles.dockLabel}>
                            {isFreeingGPU ? 'Freeing…' : 'Free GPU'}
                        </Typography>
                    </Box>

                    <Box sx={appStyles.dockItem}>
                        <IconButton
                            aria-label="Fresh start"
                            onClick={() => setShowStartFreshDialog(true)}
                            disabled={isStartingFresh}
                            sx={[appStyles.dockIconButton, appStyles.dockIconButtonDanger]}
                        >
                            {isStartingFresh ? <CircularProgress size={16} color="inherit" /> : <DeleteIcon size={18} />}
                        </IconButton>
                        <Typography className="dock-label" sx={appStyles.dockLabel}>
                            {isStartingFresh ? 'Starting…' : 'Fresh Start'}
                        </Typography>
                    </Box>

                    <Box sx={appStyles.dockItem}>
                        <IconButton
                            aria-label={colorMode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
                            onClick={toggleColorMode}
                            sx={[appStyles.dockIconButton, { color: colorMode === 'light' ? 'night.main' : 'warm.main', '&:hover': { color: colorMode === 'light' ? 'night.main' : 'warm.main' } }]}
                        >
                            {colorMode === 'light' ? <MoonIcon size={18} /> : <SunIcon size={18} />}
                        </IconButton>
                        <Typography className="dock-label" sx={appStyles.dockLabel}>
                            {colorMode === 'light' ? 'Dark Mode' : 'Light Mode'}
                        </Typography>
                    </Box>

                    <Box sx={appStyles.dockItem}>
                        <IconButton
                            aria-label="Open about and documentation"
                            onClick={() => setShowInfoDialog(true)}
                            sx={appStyles.dockIconButton}
                        >
                            <InfoIcon size={18} />
                        </IconButton>
                        <Typography className="dock-label" sx={appStyles.dockLabel}>
                            About
                        </Typography>
                    </Box>
                </Paper>
            )}

            <Dialog
                open={showInfoDialog}
                onClose={() => setShowInfoDialog(false)}
                aria-labelledby="about-documentation-dialog-title"
                maxWidth="sm"
                fullWidth
            >
                <DialogTitle id="about-documentation-dialog-title">
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                        <Box sx={{
                            ...appStyles.logo,
                            width: 52, height: 52,
                            border: 'none',
                            boxShadow: 'none',
                            filter: 'none',
                        }} />
                        <Typography variant="h5" component="span" sx={appStyles.title}>
                            Fragmenta
                        </Typography>
                    </Box>
                </DialogTitle>
                <DialogContent>
                    <Typography sx={appStyles.infoDialogIntro}>
                        Fragmenta is an open source, local-first pipeline to fine-tune, LoRA, train, generate and perform with text-to-audio diffusion models.
                        Made by the composer and researcher Misagh Azimi.
                    </Typography>

                    <Box sx={appStyles.infoDialogActionStack}>
                        <Button
                            variant="contained"
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
                            size="small"
                            disabled
                            sx={appStyles.infoDocButton}
                        >
                            Tutorials (Coming soon...)
                        </Button>
                    </Box>

                    <Box sx={{ mt: 3, pt: 1.5, borderTop: '1px solid', borderColor: 'divider', textAlign: 'center' }}>
                        <Typography variant="caption" color="textSecondary" sx={{ display: 'block', lineHeight: 1.5, fontSize: '0.68rem' }}>
                            <strong>Powered by Stability AI</strong> —{' '}
                            <Typography
                                component="a"
                                variant="caption"
                                href="https://huggingface.co/stabilityai/stable-audio-open-1.0"
                                target="_blank"
                                rel="noopener noreferrer"
                                sx={{ color: 'primary.main', textDecoration: 'underline', fontSize: '0.68rem' }}
                            >
                                Stable Audio Open
                            </Typography>{' '}
                            models, governed by the{' '}
                            <Typography
                                component="a"
                                variant="caption"
                                href="https://stability.ai/license"
                                target="_blank"
                                rel="noopener noreferrer"
                                sx={{ color: 'primary.main', textDecoration: 'underline', fontSize: '0.68rem' }}
                            >
                                Stability AI Community License
                            </Typography>.
                        </Typography>
                        <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 0.5, fontStyle: 'italic', fontSize: '0.6rem', lineHeight: 1.4 }}>
                            "This Stability AI Model is licensed under the Stability AI Community License,{' '}
                            Copyright © Stability AI Ltd. All Rights Reserved"
                        </Typography>
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setShowInfoDialog(false)}>
                        Close
                    </Button>
                </DialogActions>
            </Dialog>

            <Dialog
                open={suggestionDialog.open}
                onClose={() => setSuggestionDialog({ open: false, data: null, loading: false })}
                maxWidth="sm"
                fullWidth
            >
                <DialogTitle>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <WandIcon size={18} />
                        <span>Suggested hyperparameters</span>
                    </Box>
                </DialogTitle>
                <DialogContent>
                    {suggestionDialog.loading && (
                        <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}>
                            <CircularProgress size={28} />
                        </Box>
                    )}
                    {!suggestionDialog.loading && suggestionDialog.data?.ok === false && (
                        <Typography color="error" variant="body2">
                            {suggestionDialog.data.error || 'Could not generate a suggestion.'}
                        </Typography>
                    )}
                    {!suggestionDialog.loading && suggestionDialog.data?.ok && (() => {
                        const { stats, config, rationale } = suggestionDialog.data;
                        return (
                            <Box>
                                <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                                    {stats.file_count} files · {stats.duration_human}
                                    {stats.vram_gb ? ` · GPU ${stats.vram_gb} GB` : ''}
                                </Typography>

                                <Box sx={{
                                    display: 'grid',
                                    gridTemplateColumns: '1fr auto',
                                    rowGap: 0.75,
                                    columnGap: 2,
                                    fontVariantNumeric: 'tabular-nums',
                                    mb: 2,
                                }}>
                                    <Typography variant="body2">Batch size</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{config.batchSize}</Typography>
                                    <Typography variant="body2">Learning rate</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{config.learningRate}</Typography>
                                    <Typography variant="body2">Epochs</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{config.epochs}</Typography>
                                    {trainingConfig.mode === 'lora' && (
                                        <>
                                            <Typography variant="body2">LoRA rank / alpha</Typography>
                                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                                {config.loraRank} / {config.loraAlpha}
                                            </Typography>
                                        </>
                                    )}
                                    <Typography variant="body2" color="textSecondary">Total steps</Typography>
                                    <Typography variant="body2" color="textSecondary">{stats.total_steps}</Typography>
                                </Box>

                                <Button
                                    size="small"
                                    onClick={() => setShowRationale(v => !v)}
                                    endIcon={<ExpandMoreIcon
                                        size={14}
                                        style={{ transform: showRationale ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}
                                    />}
                                    sx={{ textTransform: 'none', mb: 1, px: 0 }}
                                >
                                    Why these values?
                                </Button>
                                {showRationale && (
                                    <Box component="ul" sx={{ pl: 2.5, m: 0 }}>
                                        {rationale.map((r, i) => (
                                            <Typography component="li" variant="body2" color="textSecondary" key={i} sx={{ mb: 0.5 }}>
                                                {r}
                                            </Typography>
                                        ))}
                                    </Box>
                                )}
                            </Box>
                        );
                    })()}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setSuggestionDialog({ open: false, data: null, loading: false })}>
                        Cancel
                    </Button>
                    <Button
                        variant="contained"
                        onClick={applyHyperparamSuggestion}
                        disabled={!suggestionDialog.data?.ok}
                    >
                        Apply
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

            <CheckpointManagerWindow
                open={checkpointMgrOpen}
                onClose={() => {
                    setCheckpointMgrOpen(false);
                    refreshAllModels();
                }}
            />
        </ThemeProvider>
    );
}

export default App; 