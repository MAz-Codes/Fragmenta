import React, { useCallback, useState, useEffect, useMemo, useRef } from 'react';
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
    Moon as MoonIcon,
    Sun as SunIcon,
    Piano as PerformanceIcon,
    AlertCircle as AlertIcon,
    Wand2 as WandIcon,
    Trash2 as DeleteIcon,
    Menu as MenuIcon,
    CheckCircle2 as CheckCircleIcon,
} from 'lucide-react';
import api from './api';
import HfAuthDialog from './components/HfAuthDialog';
import AboutDialog from './components/AboutDialog';
import TabPanel from './components/TabPanel';
import DatasetPrep from './components/DatasetPrep';
import TrainingMonitor from './components/TrainingMonitor';
import CheckpointManagerWindow from './components/CheckpointManagerWindow';
import LoraStack from './components/LoraStack';
import EditPanel from './components/EditPanel';
import GeneratedFragmentsWindow from './components/GeneratedFragmentsWindow';
import WelcomePage from './components/WelcomePage';
import { formatDuration } from './utils/format';
import theme, { appStyles, lightTheme } from './theme';

import PerformancePanel from './components/PerformancePanel';

const COLOR_MODE_STORAGE_KEY = 'fragmenta-color-mode';
const HIDE_WELCOME_PAGE_KEY = 'fragmenta-hide-welcome-v2';

// Persisted across reload so the user lands back where they were.
// Tabs are: 0=Dataset, 1=Training, 2=Generation, 3=Performance.
const TAB_STORAGE_KEY = 'fragmenta.lastTab';
const TAB_COUNT = 4;
const readStoredTab = () => {
    try {
        const raw = window.localStorage.getItem(TAB_STORAGE_KEY);
        const n = Number(raw);
        return Number.isFinite(n) && n >= 0 && n < TAB_COUNT ? n : 0;
    } catch {
        return 0;
    }
};

function App() {
    const [tabValue, setTabValue] = useState(readStoredTab);
    // Lags behind tabValue by ~fadeDuration so content swap happens
    // while the panel is invisible (cross-fade between pages).
    const [displayedTab, setDisplayedTab] = useState(readStoredTab);
    const TAB_FADE_MS = 180;

    // Persist the active tab so a reload returns the user to it.
    useEffect(() => {
        try { window.localStorage.setItem(TAB_STORAGE_KEY, String(tabValue)); } catch {}
    }, [tabValue]);
    // Header sticky chrome only kicks in once the page has scrolled.
    const [isScrolled, setIsScrolled] = useState(false);
    // Measure the header's actual rendered height so the fixed nav
    // rail can be pinned at exactly the first card's top edge.
    const headerRef = useRef(null);
    const [navTopPx, setNavTopPx] = useState(94);
    const [processingStatus, setProcessingStatus] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);

    const [showWelcomePage, setShowWelcomePage] = useState(
        () => window.localStorage.getItem(HIDE_WELCOME_PAGE_KEY) !== 'true'
    );
    const [authDialogOpen, setAuthDialogOpen] = useState(false);
    const [checkpointMgrOpen, setCheckpointMgrOpen] = useState(false);
    const [generationModelSelectOpen, setGenerationModelSelectOpen] = useState(false);
    const [trainingBaseModelSelectOpen, setTrainingBaseModelSelectOpen] = useState(false);
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
        steps: 1000,                          // SA3 quick-start
        checkpointSteps: 250,
        checkpointAuto: true,
        batchSize: 1,                         // SA3 examples all use 1
        learningRate: 1e-4,
        modelName: 'my_lora',
        baseModel: 'sa3-small-music-base',    // only *-base checkpoints are valid targets
        precision: 'bf16',
        duration: 30.0,                       // training window in seconds

        loraRank: 16,
        loraAlpha: 16,
        loraDropout: 0,
        adapterType: 'dora-rows',             // SA3 upstream default
        seed: 42,                              // reproducible by default

        // SA3 docs' "common case" layer filter — prevents conditioner-hijacking
        // on small datasets. Stored as space-separated strings (the format SA3's
        // CLI consumes) so the Advanced TextFields can edit them directly.
        include: 'transformer.layers',
        exclude: 'seconds_total to_local_embed',
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

    // Generation panel top-level mode: 'create' (text → audio) or
    // 'edit' (audio → audio: style transfer, inpaint, extend).
    const [generationMode, setGenerationMode] = useState('create');
    const [generationPrompt, setGenerationPrompt] = useState('');
    const [negativePrompt, setNegativePrompt] = useState('');
    const [loraStack, setLoraStack] = useState([]);   // [{path, strength}]
    const [generationDuration, setGenerationDuration] = useState(10);
    const [generatedAudio, setGeneratedAudio] = useState(null);
    const [generatedAudioBlob, setGeneratedAudioBlob] = useState(null);
    const [isGenerating, setIsGenerating] = useState(false);
    const [generationProgress, setGenerationProgress] = useState(0);
    const [selectedModel, setSelectedModel] = useState('');
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

    const deleteFragment = async (fragment) => {
        if (!fragment?.filename) return;
        try {
            await api.delete(`/api/fragments/${encodeURIComponent(fragment.filename)}`);
            setGeneratedFragments(prev => prev.filter(f => f.id !== fragment.id));
            // Best-effort revoke of blob URLs created during this session so
            // we don't leak object URLs after delete.
            if (fragment.audioUrl?.startsWith('blob:')) {
                try { URL.revokeObjectURL(fragment.audioUrl); } catch { /* ignore */ }
            }
        } catch (err) {
            console.error('Delete fragment failed:', err);
        }
    };

    const clearAllFragments = async () => {
        try {
            await api.delete('/api/fragments');
            // Revoke any in-session blob URLs before clearing state.
            generatedFragments.forEach(f => {
                if (f.audioUrl?.startsWith('blob:')) {
                    try { URL.revokeObjectURL(f.audioUrl); } catch { /* ignore */ }
                }
            });
            setGeneratedFragments([]);
        } catch (err) {
            console.error('Clear all fragments failed:', err);
        }
    };

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

    // Dataset Workbench projects available as training inputs. Refreshed on
    // mount and every time the Training tab becomes visible (in case the user
    // just committed a project on the Dataset tab).
    const [trainingProjects, setTrainingProjects] = useState([]);
    const [trainingProject, setTrainingProject] = useState(() => {
        try { return window.localStorage.getItem('fragmenta.training.lastProject') || ''; }
        catch { return ''; }
    });
    // Phase 6 — pre-encode state for the selected training project.
    // { latents_count, latents_present, job: {state, current, total, ...} | null }
    const [trainingPreEncode, setTrainingPreEncode] = useState({
        latents_count: 0,
        latents_present: false,
        job: null,
    });
    const preEncodePollRef = useRef(null);
    const refreshTrainingProjects = useCallback(async () => {
        try {
            const { data } = await api.get('/api/projects');
            setTrainingProjects(data.projects || []);
        } catch { /* non-fatal */ }
    }, []);
    useEffect(() => { refreshTrainingProjects(); }, [refreshTrainingProjects]);
    useEffect(() => {
        if (tabValue === 1) refreshTrainingProjects();
    }, [tabValue, refreshTrainingProjects]);

    // Hydrate the Generated Fragments panel from disk on mount. Each
    // /api/generate writes a sidecar JSON next to the WAV; this restores
    // the latest 100 across page reloads. Server returns newest-first; we
    // reverse so the in-memory order stays oldest-first (matches the
    // append-at-end pattern used elsewhere — GeneratedFragmentsWindow
    // reverses for display).
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const r = await api.get('/api/fragments?limit=100');
                if (cancelled) return;
                const items = (r.data?.fragments || []).map((f, i) => ({
                    id: f.created_at ? Math.round(f.created_at * 1000) + i : Date.now() - i,
                    prompt: f.prompt || '',
                    duration: f.duration,
                    cfgScale: f.cfg_scale,
                    steps: f.steps,
                    seed: f.seed,
                    modelId: f.model_id || '',
                    batchIndex: 1,
                    batchTotal: f.batch_size || 1,
                    audioUrl: `/api/fragments/${encodeURIComponent(f.filename)}`,
                    audioBlob: null,
                    filename: f.filename,
                    timestamp: f.created_at
                        ? new Date(f.created_at * 1000).toLocaleString()
                        : '',
                    createdAt: f.created_at ? f.created_at * 1000 : null,
                    editMode: f.edit_mode || null,
                }));
                // Server sends newest-first; reverse to keep the in-memory
                // append-at-end convention.
                items.reverse();
                setGeneratedFragments(items);
            } catch (err) {
                // Non-fatal — empty list is fine.
                console.warn('Failed to hydrate fragments from server:', err);
            }
        })();
        return () => { cancelled = true; };
    }, []);
    useEffect(() => {
        try {
            if (trainingProject) window.localStorage.setItem('fragmenta.training.lastProject', trainingProject);
        } catch {}
    }, [trainingProject]);
    // If the persisted project no longer exists, clear it so the picker shows "(none)".
    useEffect(() => {
        if (trainingProject && trainingProjects.length > 0 && !trainingProjects.some(p => p.name === trainingProject)) {
            setTrainingProject('');
        }
    }, [trainingProject, trainingProjects]);

    // Phase 6 — refresh pre-encode state when the user changes which project
    // they're training on, and keep polling while a job is in flight.
    const refreshTrainingPreEncode = useCallback(async (name) => {
        if (!name) {
            setTrainingPreEncode({ latents_count: 0, latents_present: false, job: null });
            return;
        }
        try {
            const [proj, status] = await Promise.all([
                api.get(`/api/projects/${encodeURIComponent(name)}`),
                api.get(`/api/projects/${encodeURIComponent(name)}/pre-encode/status`),
            ]);
            setTrainingPreEncode({
                latents_count: proj.data.latents_count ?? 0,
                latents_present: !!proj.data.latents_present,
                job: status.data.job ?? null,
            });
        } catch { /* non-fatal */ }
    }, []);

    useEffect(() => {
        refreshTrainingPreEncode(trainingProject);
    }, [trainingProject, refreshTrainingPreEncode]);

    // Poll while a job is queued/running. Clean up on project change or unmount.
    useEffect(() => {
        const job = trainingPreEncode.job;
        const inFlight = job && (job.state === 'queued' || job.state === 'running');
        if (!inFlight || !trainingProject) {
            if (preEncodePollRef.current) {
                window.clearTimeout(preEncodePollRef.current);
                preEncodePollRef.current = null;
            }
            return;
        }
        preEncodePollRef.current = window.setTimeout(() => {
            refreshTrainingPreEncode(trainingProject);
        }, 750);
        return () => {
            if (preEncodePollRef.current) {
                window.clearTimeout(preEncodePollRef.current);
                preEncodePollRef.current = null;
            }
        };
    }, [trainingProject, trainingPreEncode.job, refreshTrainingPreEncode]);

    const startTrainingPreEncode = useCallback(async () => {
        if (!trainingProject) return;
        try {
            await api.post(`/api/projects/${encodeURIComponent(trainingProject)}/pre-encode`);
            refreshTrainingPreEncode(trainingProject);
        } catch (e) {
            console.error('Failed to start pre-encode', e);
        }
    }, [trainingProject, refreshTrainingPreEncode]);

    const cancelTrainingPreEncode = useCallback(async () => {
        if (!trainingProject) return;
        try {
            await api.post(`/api/projects/${encodeURIComponent(trainingProject)}/pre-encode/cancel`);
            refreshTrainingPreEncode(trainingProject);
        } catch (e) { /* non-fatal */ }
    }, [trainingProject, refreshTrainingPreEncode]);

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
    }, [selectedModel, isDistilledBase]);

    const handleTabChange = (event, newValue) => {
        if (newValue === tabValue) return;
        setTabValue(newValue);
    };

    // Sync displayedTab to tabValue with a fade-out delay so content
    // swap happens while the wrapper opacity is at 0. Works for any
    // code path that updates tabValue (Tabs click, model-warning
    // auto-jump, etc).
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
                            current_step: currentStatus.current_step ?? currentStatus.step ?? 0,
                            loss: currentStatus.loss,
                            checkpoints_saved: currentStatus.checkpoints_saved
                                ?? (currentStatus.checkpoints?.length || 0),
                            is_training: currentStatus.is_training,
                            message: currentStatus.error ||
                                (currentStatus.progress > 0 ? `Progress: ${currentStatus.progress}%` : 'Starting...')
                        };

                        const lastEntry = prev[prev.length - 1];
                        if (!lastEntry ||
                            lastEntry.progress !== newEntry.progress ||
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
                            // refreshAllModels picks up the new LoRA — without it,
                            // the LoRA picker stays empty until the user manually
                            // hits refresh.
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


    const fetchHyperparamSuggestion = async () => {
        setShowRationale(false);
        if (!trainingProject) {
            setSuggestionDialog({
                open: true,
                data: { ok: false, error: "Pick a dataset project first." },
                loading: false,
            });
            return;
        }
        setSuggestionDialog({ open: true, data: null, loading: true });
        try {
            const url = `/api/training/suggest-hyperparams`
                + `?project_name=${encodeURIComponent(trainingProject)}`
                + `&base_model=${encodeURIComponent(trainingConfig.baseModel || '')}`;
            const resp = await api.get(url);
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
        // Suggester returns include/exclude as arrays; the form edits them as
        // space-separated strings. Backend's sa3_trainer accepts either.
        const normalized = {
            ...cfg,
            include: Array.isArray(cfg.include) ? cfg.include.join(' ') : (cfg.include || ''),
            exclude: Array.isArray(cfg.exclude) ? cfg.exclude.join(' ') : (cfg.exclude || ''),
        };
        setTrainingConfig({ ...trainingConfig, ...normalized });
        setSuggestionDialog({ open: false, data: null, loading: false });
    };

    // Confirm dialog for the same-name LoRA collision case.
    const [overwriteConfirm, setOverwriteConfirm] = useState(null);

    const startTraining = async (overwrite = false) => {
        // Defensive: an `onClick={startTraining}` would pass React's
        // SyntheticEvent in as the first arg; coerce so it can never
        // leak into the JSON payload as a circular DOM reference.
        overwrite = overwrite === true;
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

        if (!trainingProject) {
            showModelWarning({
                title: 'Dataset Required',
                message: 'Pick a dataset project before starting training. '
                       + 'Create one in the Dataset tab if you don\'t have any yet.',
                canOpenModels: false,
            });
            return;
        }

        setIsTraining(true);
        setTrainingProgress(0);
        setTrainingError(null);
        setTrainingStartTime(Date.now());
        setTrainingHistory([]);

        await api.post('/api/clap/unload').catch(() => {});

        try {
            const { checkpointAuto, ...rest } = trainingConfig;
            const payload = {
                ...rest,
                projectName: trainingProject,
                checkpointSteps: checkpointAuto ? null : trainingConfig.checkpointSteps,
                overwrite: overwrite,
            };
            const response = await api.post('/api/start-training', payload);
            setProcessingStatus('Training started successfully!');
        } catch (error) {
            const errorData = error.response?.data;
            const errorMessage = errorData?.error || error.message;

            // Same-name collision (HTTP 409) — surface a confirm dialog so the
            // user can choose to overwrite the previous run rather than
            // co-mingling its checkpoints.
            if (error.response?.status === 409 && errorData?.code === 'run_exists') {
                setIsTraining(false);
                setOverwriteConfirm({
                    runName: errorData.run_name,
                    checkpointCount: errorData.checkpoint_count,
                    message: errorData.message,
                });
                return;
            }

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

        // LoRA stack — LoraStack is the single source of truth for the
        // Generation panel. Empty slots (path === '') are filtered out
        // so an unused slot doesn't break the request.
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

        const parsedSeed = parseInt(seedValue, 10);
        if (!randomSeed && (Number.isNaN(parsedSeed) || parsedSeed < 0)) {
            setProcessingStatus('Please enter a non-negative integer seed, or enable Random Seed');
            return;
        }

        const totalRuns = Math.max(1, Math.min(10, batchCount));

        await api.post('/api/clap/unload').catch(() => {});

        stopGenerationRef.current = false;
        const abortController = new AbortController();
        generationAbortRef.current = abortController;

        setIsGenerating(true);
        setGenerationProgress(0);

        // Real progress polling — the backend exposes /api/generation-progress
        // which reflects the SA3 sampler's per-ODE-step callback. We poll at
        // ~250ms; sampling is N steps total (8 for distilled, ~50 for base)
        // so each step takes hundreds of ms to several seconds — finer polling
        // is unnecessary.
        let progressInterval;
        const startProgressTicker = () => {
            progressInterval = setInterval(async () => {
                try {
                    const r = await api.get('/api/generation-progress');
                    const d = r.data || {};
                    // Don't drop to 0 just because backend briefly reports
                    // idle between batch elements; clamp monotonic until
                    // we hand off to setGenerationProgress(100) on response.
                    const pct = Number(d.progress) || 0;
                    setGenerationProgress(prev => Math.max(prev, Math.min(95, pct)));
                } catch {
                    /* poll failure is non-fatal — bar just freezes briefly */
                }
            }, 250);
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
                    modelId: selectedModel,
                    batchIndex,
                    batchTotal: totalRuns,
                    audioUrl,
                    audioBlob: response.data,
                    filename: fragmentFilename,
                    timestamp: new Date().toLocaleString(),
                    createdAt: Date.now(),
                };

                setGeneratedFragments(prev => {
                    const next = [...prev, newFragment];
                    return next.length > 100 ? next.slice(next.length - 100) : next;
                });
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
        if (!selectedModel) return '';
        const baseModel = baseModels.find(m => m.name === selectedModel);
        if (baseModel) return baseModel.displayName;
        return selectedModel;
    };

    const allAvailableModels = [
        ...baseModels,
        ...availableModels
    ];

    const handleModelChange = (event) => {
        const newSelectedModel = event.target.value;
        setSelectedModel(newSelectedModel);

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
                                <Tab icon={<UploadIcon size={20} />} iconPosition={isIconOnlySidebar ? 'top' : 'start'} label={(isIconOnlySidebar || isMobileLayout) ? undefined : 'Dataset'} />
                                <Tab icon={<ActivityIcon size={20} />} iconPosition={isIconOnlySidebar ? 'top' : 'start'} label={(isIconOnlySidebar || isMobileLayout) ? undefined : 'Training'} />
                                <Tab icon={<SparklesIcon size={20} />} iconPosition={isIconOnlySidebar ? 'top' : 'start'} label={(isIconOnlySidebar || isMobileLayout) ? undefined : 'Generation'} />
                                <Tab
                                    icon={<PerformanceIcon size={20} />}
                                    iconPosition={isIconOnlySidebar ? 'top' : 'start'}
                                    label={(isIconOnlySidebar || isMobileLayout) ? undefined : 'Performance'}
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

                            {/* Dataset Tab */}
                            <TabPanel value={displayedTab} index={0}>
                                <DatasetPrep onOpenCheckpointManager={() => setCheckpointMgrOpen(true)} />
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
                                                        Dataset
                                                    </Typography>
                                                    {trainingProjects.length === 0 ? (
                                                        <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                                                            No projects yet — create one in the Dataset tab.
                                                        </Typography>
                                                    ) : (
                                                        <Select
                                                            fullWidth
                                                            size="small"
                                                            displayEmpty
                                                            value={trainingProject}
                                                            onChange={(e) => setTrainingProject(e.target.value)}
                                                            renderValue={(val) => {
                                                                if (!val) return <Typography variant="body2" color="text.secondary">Pick a project…</Typography>;
                                                                const p = trainingProjects.find(x => x.name === val);
                                                                return p
                                                                    ? `${p.name} (${p.clip_count ?? 0} clips)`
                                                                    : val;
                                                            }}
                                                        >
                                                            {trainingProjects.map(p => (
                                                                <MenuItem key={p.name} value={p.name}>
                                                                    <Box sx={{ flex: 1, minWidth: 0 }}>
                                                                        <Typography variant="body2">{p.name}</Typography>
                                                                        <Typography variant="caption" color="text.secondary">
                                                                            {(p.clip_count ?? 0)} clip{p.clip_count === 1 ? '' : 's'}
                                                                            {p.has_draft ? ' · draft pending' : ''}
                                                                        </Typography>
                                                                    </Box>
                                                                </MenuItem>
                                                            ))}
                                                        </Select>
                                                    )}

                                                    {/* Phase 6 — pre-encode latents button. State machine:
                                                        no latents → "Pre-encode latents · N clips" (clickable, outlined)
                                                        running   → "Encoding… X / Y" (disabled, with Stop button)
                                                        present   → "✓ Pre-encoded · N latents" (disabled, outlined, green tint). */}
                                                    {trainingProject && (() => {
                                                        const job = trainingPreEncode.job;
                                                        const inFlight = job && (job.state === 'queued' || job.state === 'running');
                                                        const ready = trainingPreEncode.latents_present && !inFlight;
                                                        const project = trainingProjects.find(p => p.name === trainingProject);
                                                        const clipCount = project?.clip_count ?? 0;
                                                        let label = `Pre-encode latents · ${clipCount} clip${clipCount === 1 ? '' : 's'}`;
                                                        if (inFlight) {
                                                            label = job.total > 0
                                                                ? `Encoding… ${job.current} / ${job.total}`
                                                                : 'Encoding…';
                                                        } else if (ready) {
                                                            label = `Pre-encoded · ${trainingPreEncode.latents_count} latent${trainingPreEncode.latents_count === 1 ? '' : 's'}`;
                                                        }
                                                        return (
                                                            <Box sx={{ mt: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                <Button
                                                                    fullWidth
                                                                    size="small"
                                                                    variant="outlined"
                                                                    onClick={startTrainingPreEncode}
                                                                    disabled={inFlight || ready || clipCount === 0}
                                                                    color={ready ? 'success' : 'primary'}
                                                                    startIcon={ready ? <CheckCircleIcon size={14} /> : null}
                                                                    sx={{
                                                                        justifyContent: 'center',
                                                                        textTransform: 'none',
                                                                        // Make the "done" state visibly disabled (gray border /
                                                                        // muted text) while still showing the success-green
                                                                        // checkmark so the user can read the status at a glance.
                                                                        ...(ready ? {
                                                                            '&.Mui-disabled': {
                                                                                color: 'text.disabled',
                                                                                borderColor: 'divider',
                                                                                '& .MuiButton-startIcon': {
                                                                                    color: 'success.main',
                                                                                    opacity: 0.8,
                                                                                },
                                                                            },
                                                                        } : {}),
                                                                    }}
                                                                >
                                                                    {label}
                                                                </Button>
                                                                {inFlight && (
                                                                    <Button
                                                                        size="small"
                                                                        variant="text"
                                                                        color="error"
                                                                        onClick={cancelTrainingPreEncode}
                                                                        sx={{ minWidth: 0, textTransform: 'none' }}
                                                                    >
                                                                        Stop
                                                                    </Button>
                                                                )}
                                                            </Box>
                                                        );
                                                    })()}
                                                </Box>

                                                <Box sx={appStyles.fieldMarginBottom}>
                                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                                                        Base model to fine-tune
                                                    </Typography>
                                                    <Select
                                                        fullWidth
                                                        size="small"
                                                        value={trainingConfig.baseModel}
                                                        open={trainingBaseModelSelectOpen}
                                                        onOpen={() => setTrainingBaseModelSelectOpen(true)}
                                                        onClose={() => setTrainingBaseModelSelectOpen(false)}
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
                                                                <MenuItem
                                                                    key={m.name}
                                                                    value={m.name}
                                                                    disabled={!m.downloaded}
                                                                    sx={{
                                                                        display: 'flex',
                                                                        alignItems: 'center',
                                                                        gap: 1,
                                                                        '&.Mui-disabled': { pointerEvents: 'auto' },
                                                                    }}
                                                                >
                                                                    <Box sx={{ flex: 1, minWidth: 0 }}>
                                                                        <Typography variant="body2">
                                                                            {m.displayName}
                                                                        </Typography>
                                                                        <Typography variant="caption" color="text.secondary">
                                                                            {m.description}
                                                                        </Typography>
                                                                    </Box>
                                                                    {!m.downloaded && (
                                                                        <Tooltip title="Download this model">
                                                                            <IconButton
                                                                                size="small"
                                                                                onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); }}
                                                                                onClick={(e) => {
                                                                                    e.stopPropagation();
                                                                                    e.preventDefault();
                                                                                    setTrainingBaseModelSelectOpen(false);
                                                                                    setCheckpointMgrOpen(true);
                                                                                }}
                                                                                sx={{ opacity: 1, color: 'primary.main' }}
                                                                            >
                                                                                <CloudDownloadIcon size={16} />
                                                                            </IconButton>
                                                                        </Tooltip>
                                                                    )}
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
                                                                <Box sx={appStyles.fieldLabelRow}>
                                                                    <Typography>Training Steps</Typography>
                                                                    <Tooltip arrow placement="top" title="SA3's documented quick-start is 1 000 steps. LoRAs typically overfit well before that; watch the loss curve.">
                                                                        <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                    </Tooltip>
                                                                </Box>
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
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Box sx={appStyles.fieldLabelRow}>
                                                                    <Typography>Adapter Type</Typography>
                                                                    <Tooltip arrow placement="top" title="DoRA-rows is SA3's upstream default and works best for most stylistic LoRAs. The -xs variants freeze SVD bases and only train a tiny core matrix — far fewer parameters, useful when VRAM is tight. BoRA scales both rows and columns independently (more expressive, more parameters).">
                                                                        <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                    </Tooltip>
                                                                </Box>
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
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                                        <Typography>Checkpoint Interval (steps)</Typography>
                                                                        <Tooltip arrow placement="top" title="How often a LoRA .safetensors snapshot gets written. Auto picks ~10 checkpoints per run (capped 250–1 000 steps). Lower = more granular but more disk; higher = fewer files to compare.">
                                                                            <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                        </Tooltip>
                                                                    </Box>
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
                                                                <Box sx={appStyles.fieldLabelRow}>
                                                                    <Typography>Batch Size</Typography>
                                                                    <Tooltip arrow placement="top" title="SA3 examples use 1. Each extra sample adds ~1–2 GB of activations. Raise only on roomy GPUs (≥24 GB); medium-base activations are heavy. Lower if you hit CUDA OOM.">
                                                                        <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                    </Tooltip>
                                                                </Box>
                                                                <Box sx={appStyles.sliderRow}>
                                                                    <Slider
                                                                        value={trainingConfig.batchSize}
                                                                        onChange={(e, value) => setTrainingConfig({
                                                                            ...trainingConfig,
                                                                            batchSize: value
                                                                        })}
                                                                        min={1}
                                                                        max={8}
                                                                        step={1}
                                                                        marks
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
                                                                                batchSize: Math.max(1, Math.min(8, val))
                                                                            });
                                                                        }}
                                                                        inputProps={{ min: 1, max: 8, step: 1 }}
                                                                        sx={appStyles.sliderInputSmall}
                                                                        size="small"
                                                                    />
                                                                </Box>
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Box sx={appStyles.fieldLabelRow}>
                                                                    <Typography>Base-model Precision</Typography>
                                                                    <Tooltip arrow placement="top" title="Cast applied to the frozen base weights only; LoRA parameters stay in fp32 for the optimizer. bf16 halves the VRAM used by the base with negligible quality cost on Ampere and newer cards.">
                                                                        <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                    </Tooltip>
                                                                </Box>
                                                                <FormControl fullWidth size="small">
                                                                    <Select
                                                                        value={trainingConfig.precision}
                                                                        onChange={(e) => setTrainingConfig({
                                                                            ...trainingConfig,
                                                                            precision: e.target.value
                                                                        })}
                                                                    >
                                                                        <MenuItem value="bf16">bf16 (recommended — halves VRAM, negligible quality cost)</MenuItem>
                                                                        <MenuItem value="fp16">fp16 (legacy — only if your GPU lacks bf16 support)</MenuItem>
                                                                    </Select>
                                                                </FormControl>
                                                            </Grid>

                                                            <Grid item xs={12}>
                                                                <Typography variant="subtitle2" color="textSecondary" sx={{ mt: 1, mb: 1 }}>
                                                                    LoRA settings
                                                                </Typography>

                                                                    <Box sx={appStyles.fieldLabelRow}>
                                                                        <Typography>Rank</Typography>
                                                                        <Tooltip arrow placement="top" title="Capacity of the LoRA update — rank-k matrices A (k×in) and B (out×k) are trained. Higher rank = more expressive but larger file and more VRAM. r=16 fits comfortably on 16 GB and is SA3's default.">
                                                                            <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                        </Tooltip>
                                                                    </Box>
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
                                                                    <Box sx={{ ...appStyles.fieldLabelRow, mt: 2 }}>
                                                                        <Typography>Alpha</Typography>
                                                                        <Tooltip arrow placement="top" title="Scaling factor for the LoRA update. Effective scaling is alpha / rank — setting alpha = rank gives a scaling of 1.0. Conventional choice: alpha = rank.">
                                                                            <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                        </Tooltip>
                                                                    </Box>
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
                                                                    <Box sx={{ ...appStyles.fieldLabelRow, mt: 2 }}>
                                                                        <Typography>Dropout</Typography>
                                                                        <Tooltip arrow placement="top" title="Regularization probability applied to LoRA inputs during training. 0 is fine for most cases — raise to ~0.05 if you see overfitting on small datasets.">
                                                                            <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                        </Tooltip>
                                                                    </Box>
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

                                                                    <Box sx={{ ...appStyles.fieldLabelRow, mt: 2 }}>
                                                                        <Typography>Seed</Typography>
                                                                        <Tooltip arrow placement="top" title="Random seed for reproducibility — same dataset + same hyperparameters + same seed produces the same LoRA. Change it to re-roll with different sampling behaviour.">
                                                                            <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                        </Tooltip>
                                                                    </Box>
                                                                    <TextField
                                                                        type="number"
                                                                        size="small"
                                                                        fullWidth
                                                                        value={trainingConfig.seed}
                                                                        onChange={(e) => {
                                                                            const v = parseInt(e.target.value, 10);
                                                                            setTrainingConfig({
                                                                                ...trainingConfig,
                                                                                seed: Number.isFinite(v) ? v : 42,
                                                                            });
                                                                        }}
                                                                        inputProps={{ min: 0, step: 1 }}
                                                                    />

                                                                    <Box sx={{ ...appStyles.fieldLabelRow, mt: 2 }}>
                                                                        <Typography>Training Window (seconds)</Typography>
                                                                        <Tooltip arrow placement="top" title="Audio fed to the model per training step. Long clips get random-cropped to this length each step; short clips get silence-padded. Raise beyond 30s only after pre-encoded latents land (Phase 6).">
                                                                            <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                        </Tooltip>
                                                                    </Box>
                                                                    <Box sx={appStyles.sliderRow}>
                                                                        <Slider
                                                                            value={trainingConfig.duration}
                                                                            onChange={(e, value) => setTrainingConfig({
                                                                                ...trainingConfig,
                                                                                duration: value,
                                                                            })}
                                                                            min={5}
                                                                            max={30}
                                                                            step={1}
                                                                            valueLabelDisplay="auto"
                                                                            sx={appStyles.sliderFlexGrow}
                                                                        />
                                                                        <TextField
                                                                            type="number"
                                                                            value={trainingConfig.duration}
                                                                            onChange={(e) => {
                                                                                const v = Math.max(5, Math.min(30, parseFloat(e.target.value) || 30));
                                                                                setTrainingConfig({ ...trainingConfig, duration: v });
                                                                            }}
                                                                            inputProps={{ min: 5, max: 30, step: 1 }}
                                                                            sx={appStyles.sliderInputSmall}
                                                                            size="small"
                                                                        />
                                                                    </Box>

                                                                    <Box sx={{ ...appStyles.fieldLabelRow, mt: 2 }}>
                                                                        <Typography>Include layers</Typography>
                                                                        <Tooltip arrow placement="top" title="Space-separated substrings — only layers whose fully-qualified name contains one of these get LoRA. Empty = all matching Linear/Conv1d layers. Example: transformer.layers.">
                                                                            <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                        </Tooltip>
                                                                    </Box>
                                                                    <TextField
                                                                        fullWidth
                                                                        size="small"
                                                                        value={trainingConfig.include}
                                                                        onChange={(e) => setTrainingConfig({ ...trainingConfig, include: e.target.value })}
                                                                    />

                                                                    <Box sx={{ ...appStyles.fieldLabelRow, mt: 2 }}>
                                                                        <Typography>Exclude layers</Typography>
                                                                        <Tooltip arrow placement="top" title="Space-separated substrings — matching layers are skipped, even if they also match Include. SA3-docs default (seconds_total to_local_embed) prevents conditioner-hijacking on small datasets.">
                                                                            <Box component="span" sx={appStyles.fieldHelpIcon}><InfoIcon size={14} /></Box>
                                                                        </Tooltip>
                                                                    </Box>
                                                                    <TextField
                                                                        fullWidth
                                                                        size="small"
                                                                        value={trainingConfig.exclude}
                                                                        onChange={(e) => setTrainingConfig({ ...trainingConfig, exclude: e.target.value })}
                                                                    />
                                                            </Grid>

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
                                                        onClick={() => startTraining(false)}
                                                        disabled={isTraining || !trainingProject || !trainingConfig.baseModel || (() => {
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
                                                    trainingError={trainingError}
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
                                                                            {model.has_checkpoint ? 'Checkpoint' : 'No Checkpoint'}
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


                                                {/* Phase 4: LoraStack is the single LoRA picker for the
                                                    Generation panel. Always rendered between model picker
                                                    and mode toggle so it's visible in both Create + Edit
                                                    modes without expanding Advanced Settings. */}
                                                <Box sx={{ mb: 2 }}>
                                                    <LoraStack
                                                        selectedModel={selectedModel}
                                                        value={loraStack}
                                                        onChange={setLoraStack}
                                                    />
                                                </Box>

                                                {/* Phase 8: top-level mode switch. Create = text→audio,
                                                    Edit = audio→audio (style / inpaint / extend). The
                                                    model picker and LoRA picker above stay visible in
                                                    both modes. */}
                                                <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
                                                    <ToggleButtonGroup
                                                        value={generationMode}
                                                        exclusive
                                                        size="small"
                                                        onChange={(_, v) => v && setGenerationMode(v)}
                                                    >
                                                        <ToggleButton value="create">Create</ToggleButton>
                                                        <ToggleButton value="edit">Edit existing</ToggleButton>
                                                    </ToggleButtonGroup>
                                                </Box>

                                                {generationMode === 'create' && (<>
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
                                                </>)}

                                                {generationMode === 'edit' && (
                                                    <EditPanel
                                                        model_id={selectedModel}
                                                        negativePrompt={negativePrompt}
                                                        onGenerated={(blob, filename, params) => {
                                                            const audioUrl = URL.createObjectURL(blob);
                                                            const newFrag = {
                                                                id: Date.now(),
                                                                prompt: params.prompt,
                                                                duration: params.duration,
                                                                cfgScale: params.cfg_scale,
                                                                steps: params.steps,
                                                                seed: params.seed,
                                                                modelId: params.model_id,
                                                                batchIndex: 1,
                                                                batchTotal: 1,
                                                                audioUrl,
                                                                audioBlob: blob,
                                                                filename,
                                                                timestamp: new Date().toLocaleString(),
                                                                createdAt: Date.now(),
                                                                editMode: params.init_audio_path ? 'style' : params.inpaint_audio_path ? 'inpaint/extend' : null,
                                                            };
                                                            setGeneratedFragments(prev => {
                                                                const next = [...prev, newFrag];
                                                                return next.length > 100 ? next.slice(next.length - 100) : next;
                                                            });
                                                        }}
                                                    />
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
                                                    onDelete={deleteFragment}
                                                    onClearAll={clearAllFragments}
                                                />
                                            </Box>
                                        </Box>
                                    </Grid>
                                </Grid>
                            </TabPanel>

                            <TabPanel value={displayedTab} index={3} keepMounted>
                                <PerformancePanel
                                    selectedModel={selectedModel}
                                    availableModels={availableModels}
                                    baseModels={baseModels}
                                    availableLoras={availableLoras}
                                    selectedLora={selectedLora}
                                    loraMultiplier={loraMultiplier}
                                    onSelectModel={setSelectedModel}
                                    onRefreshModels={refreshAllModels}
                                    onSelectLora={setSelectedLora}
                                    onLoraMultiplierChange={setLoraMultiplier}
                                    steps={steps}
                                    onStepsChange={setSteps}
                                    randomSeed={randomSeed}
                                    seedValue={seedValue}
                                    onRandomSeedChange={setRandomSeed}
                                    onSeedValueChange={setSeedValue}
                                    onOpenCheckpointManager={() => setCheckpointMgrOpen(true)}
                                />
                            </TabPanel>
                            </Box>
                        </Box>
                    </Box>

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
                        <Divider />
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

            <AboutDialog
                open={showInfoDialog}
                onClose={() => setShowInfoDialog(false)}
                onOpenDocumentation={handleOpenDocumentation}
                isOpeningDocumentation={isOpeningDocumentation}
            />

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
                        const { stats, config, rationale, warnings } = suggestionDialog.data;
                        const includeStr = (config.include || []).join(', ') || '(all layers)';
                        const excludeStr = (config.exclude || []).join(', ') || '(none)';
                        return (
                            <Box>
                                <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                                    {stats.file_count} files · {stats.duration_human}
                                    {stats.median_clip_sec ? ` · median ${stats.median_clip_sec.toFixed(1)}s` : ''}
                                    {stats.vram_gb ? ` · GPU ${stats.vram_gb} GB` : ' · no GPU'}
                                </Typography>

                                {(warnings || []).map((w, i) => (
                                    <Alert key={i} severity="warning" sx={{ mb: 1 }} variant="outlined">
                                        {w}
                                    </Alert>
                                ))}

                                <Box sx={{
                                    display: 'grid',
                                    gridTemplateColumns: '1fr auto',
                                    rowGap: 0.75,
                                    columnGap: 2,
                                    fontVariantNumeric: 'tabular-nums',
                                    mt: warnings && warnings.length ? 2 : 0,
                                    mb: 2,
                                }}>
                                    <Typography variant="body2">Steps</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{config.steps.toLocaleString()}</Typography>
                                    <Typography variant="body2">Batch size</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{config.batchSize}</Typography>
                                    <Typography variant="body2">Learning rate</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{config.learningRate}</Typography>
                                    <Typography variant="body2">Training window</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{config.duration.toFixed(0)}s</Typography>
                                    <Typography variant="body2">Adapter · rank / α</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                        {config.adapterType} · {config.loraRank} / {config.loraAlpha}
                                    </Typography>
                                    <Typography variant="body2">Dropout · precision</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                        {config.loraDropout} · {config.precision}
                                    </Typography>
                                    <Typography variant="body2" color="textSecondary">Include layers</Typography>
                                    <Typography variant="body2" color="textSecondary">{includeStr}</Typography>
                                    <Typography variant="body2" color="textSecondary">Exclude layers</Typography>
                                    <Typography variant="body2" color="textSecondary">{excludeStr}</Typography>
                                    <Typography variant="body2" color="textSecondary">Checkpoint every</Typography>
                                    <Typography variant="body2" color="textSecondary">{config.checkpointSteps.toLocaleString()} steps</Typography>
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

            <Dialog
                open={Boolean(overwriteConfirm)}
                onClose={() => setOverwriteConfirm(null)}
                maxWidth="xs"
                fullWidth
            >
                <DialogTitle>Overwrite existing run?</DialogTitle>
                <DialogContent>
                    <Typography variant="body2">
                        {overwriteConfirm?.message}
                    </Typography>
                    <Alert severity="warning" sx={{ mt: 2 }} variant="outlined">
                        The previous run dir for <strong>{overwriteConfirm?.runName}</strong> will
                        be deleted, including <strong>{overwriteConfirm?.checkpointCount} checkpoint(s)</strong>,
                        training.log, metrics.csv and any Lightning logs. This cannot be undone.
                    </Alert>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOverwriteConfirm(null)}>Cancel</Button>
                    <Button
                        color="error"
                        variant="contained"
                        onClick={() => {
                            setOverwriteConfirm(null);
                            startTraining(true);
                        }}
                    >
                        Overwrite and train
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