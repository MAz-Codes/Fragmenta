import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Box, Typography, Paper, Button, IconButton, Select, MenuItem, TextField,
    Slider, FormControl, InputLabel, ToggleButton, ToggleButtonGroup, Menu,
    LinearProgress, Accordion, AccordionSummary, AccordionDetails, Chip,
    FormControlLabel, Checkbox, Alert, useTheme,
} from '@mui/material';
import {
    CircuitBoard as CircuitBoardIcon, Dices as DicesIcon,
    Save as SaveIcon, ChevronDown as ExpandMoreIcon, Square as StopIcon,
    RotateCcw as UnbendIcon, NotebookPen as LogIcon,
} from 'lucide-react';
import Tooltip from '../Tooltip';
import api from '../../api';
import { TIPS } from '../../tooltips';
import { appStyles } from '../../theme';
import SignalPath from './SignalPath';
import BendModuleCard from './BendModuleCard';
import BendingLogPanel from './BendingLogPanel';
import BreakPanel from './BreakPanel';
import BendTakes from './BendTakes';
import { buildPatch, chancePatch, newModule, nextId } from './bendUtils';

/**
 * Bend — network bending & model bending as an instrument.
 *
 * Two sub-modes: Bend (inference-time network bending — the signal path +
 * rack) and Break (training-time model bending — BreakPanel). See
 * BEND_PLAN.md; grounded in Broad et al., Kotowski & Font, Gillespie &
 * Schachter.
 *
 * Everything here is declarative and per-request: the rack describes the
 * patch, the patch rides each /api/generate call, and the backend
 * guarantees the model comes back pristine afterwards. UNBEND is
 * UI-state reset + reassurance, not cleanup.
 */

const STORE_KEY = 'fragmenta.bend.v1';

// /api/generate errors arrive as APIResponse.error — {error: {message, …}} —
// while the bend routes return {error: "…"}. Always hand React a string.
const errorText = (err, fallback) => {
    const e = err?.response?.data?.error;
    if (typeof e === 'string') return e;
    if (e && typeof e.message === 'string') return e.message;
    return fallback;
};

const readStore = () => {
    try {
        const raw = window.localStorage.getItem(STORE_KEY);
        return raw ? JSON.parse(raw) : {};
    } catch { return {}; }
};

export default function BendPanel({ models, active = true, isDocker = false }) {
    const theme = useTheme();
    const accent = theme.palette.bend?.main || '#AEB9C4';
    const stored = useMemo(readStore, []);

    const [mode, setMode] = useState(stored.mode || 'bend');
    const [modelId, setModelId] = useState(stored.modelId || '');
    const [registry, setRegistry] = useState(null);
    const [modules, setModules] = useState(stored.modules || []);
    const [prompt, setPrompt] = useState(stored.prompt || '');
    const [duration, setDuration] = useState(stored.duration || 8);
    const [steps, setSteps] = useState(stored.steps || null);
    const [seed, setSeed] = useState(stored.seed ?? 0);
    const [randomSeed, setRandomSeed] = useState(stored.randomSeed !== false);

    const [busy, setBusy] = useState(false);          // 'bent' | 'clean' | false
    const [progress, setProgress] = useState(0);
    const [statusMsg, setStatusMsg] = useState('');
    const [bendWarnings, setBendWarnings] = useState([]);
    const [bentAudio, setBentAudio] = useState(null);  // {url, seed, filename}
    const [cleanAudio, setCleanAudio] = useState(null);
    // The last bent request minus its patch: Clean A/B replays exactly this
    // (model, prompt, duration, steps, seed), so editing the prompt after a
    // bent take can't make the comparison apples-to-oranges.
    const [lastRunBody, setLastRunBody] = useState(null);

    const [presets, setPresets] = useState([]);
    const [presetAnchor, setPresetAnchor] = useState(null);
    const [chanceAnchor, setChanceAnchor] = useState(null);
    const [logEntries, setLogEntries] = useState([]);
    const [logOpen, setLogOpen] = useState(false);

    const abortRef = useRef(null);
    const tickerRef = useRef(null);

    const downloadedModels = useMemo(
        () => (models || []).filter(m => m.downloaded), [models]);

    // Default model: first downloaded, preferring distilled small (fast
    // audition loop — the modify→listen cycle must stay tight).
    // Also replaces a remembered model that is no longer downloaded.
    useEffect(() => {
        if (!downloadedModels.length) return;
        if (modelId && downloadedModels.some(m => m.name === modelId)) return;
        const preferred = downloadedModels.find(m => m.name === 'sa3-small-music')
            || downloadedModels[0];
        setModelId(preferred.name);
    }, [downloadedModels, modelId]);

    // Registry per model. A slow response for a model the user has already
    // switched away from must not overwrite the current one.
    useEffect(() => {
        let stale = false;
        const id = modelId || 'sa3-small-music-base';
        api.get(`/api/bend/targets?model_id=${encodeURIComponent(id)}`)
            .then(({ data }) => { if (!stale) setRegistry(data); })
            .catch(() => {});
        return () => { stale = true; };
    }, [modelId]);

    const refreshLog = useCallback(() => {
        const q = modelId ? `?model_id=${encodeURIComponent(modelId)}` : '';
        api.get(`/api/bend/log${q}`)
            .then(({ data }) => setLogEntries(data.entries || []))
            .catch(() => {});
    }, [modelId]);

    const refreshPresets = useCallback(() => {
        api.get('/api/bend/presets')
            .then(({ data }) => setPresets(data.presets || []))
            .catch(() => {});
    }, []);
    // The panel stays mounted across tab switches; re-read on return, since
    // Performance channels generating through a preset append to the log.
    useEffect(() => {
        if (!active) return;
        refreshLog();
        refreshPresets();
    }, [active, refreshLog, refreshPresets]);

    // Persist the session (per-viewer convenience only).
    useEffect(() => {
        try {
            window.localStorage.setItem(STORE_KEY, JSON.stringify({
                mode, modelId, modules, prompt, duration, steps, seed, randomSeed,
            }));
        } catch { /* ignore */ }
    }, [mode, modelId, modules, prompt, duration, steps, seed, randomSeed]);

    // --- rack operations ---------------------------------------------------
    const addModule = (stage) => {
        setModules(m => [...m, newModule(stage, registry)]);
    };
    const updateModule = (id, next) => {
        setModules(m => m.map(x => (x.id === id ? next : x)));
    };
    const removeModule = (id) => setModules(m => m.filter(x => x.id !== id));
    const moveModule = (id, dir) => {
        setModules(m => {
            const i = m.findIndex(x => x.id === id);
            const j = i + dir;
            if (i < 0 || j < 0 || j >= m.length) return m;
            const next = [...m];
            [next[i], next[j]] = [next[j], next[i]];
            return next;
        });
    };
    const randomizeModule = (id) => {
        setModules(m => m.map(x => {
            if (x.id !== id) return x;
            const [replacement] = chancePatch(registry, 'bend', { stage: x.target?.stage });
            return replacement ? { ...replacement, id: x.id, enabled: x.enabled } : x;
        }));
    };
    const doChance = (level) => {
        setChanceAnchor(null);
        if (!registry) return;
        setModules(chancePatch(registry, level).map(m => ({ ...m, id: nextId() })));
    };
    const unbend = () => {
        setModules([]);
        setStatusMsg('Rack cleared — the model is never left bent between generations anyway.');
    };

    // --- generation ---------------------------------------------------------
    const stopTicker = () => {
        if (tickerRef.current) { clearInterval(tickerRef.current); tickerRef.current = null; }
    };
    useEffect(() => () => { stopTicker(); abortRef.current?.abort?.(); }, []);

    const generate = async (withBend) => {
        if (!withBend && !lastRunBody) return;
        if (withBend && !modelId) { setStatusMsg('Pick a model first.'); return; }
        if (withBend && !prompt.trim()) { setStatusMsg('Write a prompt.'); return; }
        const activeModules = modules.filter(m => m.enabled !== false);
        if (withBend && !activeModules.length) {
            setStatusMsg('The rack is empty — attach a module on the signal path, or hit Chance.');
            return;
        }
        // A/B contract: "clean" replays the last bent request verbatim
        // (same seed, prompt, model, duration, steps) with the patch
        // removed; a fresh bent run resolves its seed here.
        let body;
        let patch = null;
        if (withBend) {
            const freshSeed = randomSeed
                ? Math.floor(Math.random() * 0xffffffff) : (parseInt(seed, 10) || 0);
            body = {
                model_id: modelId,
                prompt: prompt.trim(),
                duration: Number(duration),
                seed: freshSeed,
                batch_size: 1,
            };
            // Steps is only shown (and meaningful) for base models; a value
            // left over from one must not push a distilled model to 50 steps.
            if (steps && modelId.endsWith('-base')) body.steps = Number(steps);
            patch = buildPatch(activeModules, modelId, freshSeed);
        } else {
            body = { ...lastRunBody };
        }
        const runSeed = body.seed;

        const controller = new AbortController();
        abortRef.current = controller;
        setBusy(withBend ? 'bent' : 'clean');
        setBendWarnings([]);
        setProgress(0);
        setStatusMsg(withBend ? 'Generating bent…' : 'Generating clean (same seed)…');
        tickerRef.current = setInterval(async () => {
            try {
                const r = await api.get('/api/generation-progress');
                const pct = Number(r.data?.progress) || 0;
                setProgress(prev => Math.max(prev, Math.min(95, pct)));
            } catch { /* non-fatal */ }
        }, 250);

        try {
            const response = await api.post('/api/generate',
                patch ? { ...body, bend_patch: patch } : body, {
                    responseType: 'blob', signal: controller.signal,
                });
            stopTicker();
            setProgress(100);
            const url = URL.createObjectURL(response.data);
            const filename = response.headers?.['x-fragment-filename'] || '';
            try {
                const w = JSON.parse(response.headers?.['x-bend-warnings'] || '[]');
                setBendWarnings(Array.isArray(w) ? w : []);
            } catch { setBendWarnings([]); }
            const result = {
                url, blob: response.data, seed: runSeed, filename, body, key: JSON.stringify(body),
            };
            if (withBend) {
                if (bentAudio?.url?.startsWith('blob:')) URL.revokeObjectURL(bentAudio.url);
                setBentAudio(result);
                setCleanAudio(prev => {
                    // A clean take of a different request (seed, prompt,
                    // model, duration, steps) is no longer comparable.
                    if (prev && prev.key !== result.key) {
                        if (prev.url?.startsWith('blob:')) URL.revokeObjectURL(prev.url);
                        return null;
                    }
                    return prev;
                });
                setLastRunBody(body);
                refreshLog();
                setStatusMsg(`Bent fragment ready (seed ${runSeed}). Note what it did in the log below.`);
            } else {
                if (cleanAudio?.url?.startsWith('blob:')) URL.revokeObjectURL(cleanAudio.url);
                setCleanAudio(result);
                setStatusMsg(`Clean reference ready (seed ${runSeed}) — A/B against the bent take.`);
            }
            setTimeout(() => setProgress(0), 1500);
        } catch (err) {
            stopTicker();
            setProgress(0);
            if (err?.name === 'AbortError') {
                setStatusMsg('Stopped.');
            } else {
                setStatusMsg(errorText(err, `Generation failed: ${err.message}`));
            }
        } finally {
            stopTicker();
            setBusy(false);
            abortRef.current = null;
        }
    };

    const stopGeneration = () => {
        api.post('/api/stop-generation').catch(() => {});
        abortRef.current?.abort?.();
    };

    // --- presets -------------------------------------------------------------
    const savePreset = async () => {
        setPresetAnchor(null);
        if (!modules.length) {
            setStatusMsg('The rack is empty — nothing to save yet.');
            return;
        }
        const name = window.prompt('Preset name:')?.trim();
        if (!name) return;
        if (presets.some(p => p.name === name)
            && !window.confirm(`A preset named “${name}” exists. Replace it?`)) return;
        // Record the seed of the take the user just heard, not a stale
        // manual-seed field that random-seed mode never used.
        const presetSeed = bentAudio?.seed ?? (parseInt(seed, 10) || 0);
        try {
            await api.post('/api/bend/presets', {
                name,
                patch: buildPatch(modules, modelId, presetSeed, name,
                                  { keepDisabled: true }),
            });
            refreshPresets();
            setStatusMsg(`Preset “${name}” saved.`);
        } catch (err) {
            setStatusMsg(errorText(err, 'Preset save failed.'));
        }
    };
    const loadPreset = (p) => {
        setPresetAnchor(null);
        setModules((p.patch?.modules || []).map(m => ({ ...m, id: nextId() })));
        setStatusMsg(`Preset “${p.name}” loaded${
            p.model_id && p.model_id !== modelId ? ` (saved against ${p.model_id} — unresolvable targets are skipped)` : ''}.`);
    };
    const deletePreset = async (p, e) => {
        e.stopPropagation();
        try {
            await api.delete(`/api/bend/presets/${encodeURIComponent(p.name)}`);
            refreshPresets();
        } catch { /* ignore */ }
    };

    const recallLogEntry = (entry) => {
        setModules((entry.patch?.modules || []).map(m => ({ ...m, id: nextId() })));
        if (entry.prompt) setPrompt(entry.prompt);
        // Duration and steps shape the noise and the schedule: without them
        // the same patch + seed is a different sound.
        // Kept exact even past the slider's range (Performance entries).
        if (entry.duration) setDuration(entry.duration);
        setSteps(entry.steps || null);
        setRandomSeed(false);
        setSeed(entry.seed);
        setMode('bend');
        setStatusMsg(`Recalled bend from the log (seed ${entry.seed}).`);
    };

    const activeCount = modules.filter(m => m.enabled !== false).length;

    return (
        <Paper sx={appStyles.elevatedInfoCard}>
            <Box sx={appStyles.sectionCardHeader}>
                <Box component="span" sx={appStyles.sectionCardIcon}>
                    <CircuitBoardIcon size={20} />
                </Box>
                <Typography variant="h6" sx={appStyles.sectionCardTitle}>Bend</Typography>
                <Box sx={{ flex: 1 }} />
                <ToggleButtonGroup
                    size="small" exclusive value={mode}
                    onChange={(_, v) => v && setMode(v)}
                    sx={{ '& .MuiToggleButton-root': { px: 2, py: 0.4 } }}
                >
                    <Tooltip title={TIPS.bend.modeBend}>
                        <ToggleButton value="bend">Bend</ToggleButton>
                    </Tooltip>
                    <Tooltip title={TIPS.bend.modeBreak}>
                        <ToggleButton value="break">Break</ToggleButton>
                    </Tooltip>
                </ToggleButtonGroup>
            </Box>

            {mode === 'break' ? (
                <BreakPanel defaultBase={
                    modelId && modelId.endsWith('-base') ? modelId : 'sa3-small-music-base'} />
            ) : (
            <>
                {/* toolbar */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                    <Tooltip title={TIPS.bend.model}>
                        <FormControl size="small" sx={{ minWidth: 180, flex: 1 }}>
                            <InputLabel>Model</InputLabel>
                            <Select value={modelId} label="Model"
                                    onChange={(e) => setModelId(e.target.value)}>
                                {downloadedModels.map(m => (
                                    <MenuItem key={m.name} value={m.name}>{m.displayName || m.name}</MenuItem>
                                ))}
                                {!downloadedModels.length && (
                                    <MenuItem value="" disabled>
                                        No models downloaded — use Get models first
                                    </MenuItem>
                                )}
                            </Select>
                        </FormControl>
                    </Tooltip>
                    <Tooltip title={TIPS.bend.presets}>
                        <Button size="small" variant="outlined" startIcon={<SaveIcon size={14} />}
                                onClick={(e) => setPresetAnchor(e.currentTarget)}>
                            Presets
                        </Button>
                    </Tooltip>
                    <Menu anchorEl={presetAnchor} open={!!presetAnchor}
                          onClose={() => setPresetAnchor(null)}>
                        <MenuItem onClick={savePreset}>Save current rack…</MenuItem>
                        {presets.length > 0 && <MenuItem disabled>— load —</MenuItem>}
                        {presets.map(p => (
                            <MenuItem key={p.name} onClick={() => loadPreset(p)}
                                      sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}>
                                <span>{p.name} <Typography component="span" variant="caption" color="textSecondary">
                                    ({p.modules})</Typography></span>
                                <Typography variant="caption" color="error"
                                            onClick={(e) => deletePreset(p, e)}
                                            sx={{ cursor: 'pointer' }}>delete</Typography>
                            </MenuItem>
                        ))}
                    </Menu>
                    <Tooltip title={TIPS.bend.chance}>
                        <span>
                            <Button size="small" variant="contained" color="bend"
                                    startIcon={<DicesIcon size={14} />}
                                    disabled={!registry}
                                    onClick={(e) => setChanceAnchor(e.currentTarget)}>
                                Chance
                            </Button>
                        </span>
                    </Tooltip>
                    <Menu anchorEl={chanceAnchor} open={!!chanceAnchor}
                          onClose={() => setChanceAnchor(null)}>
                        <MenuItem onClick={() => doChance('nudge')}>Nudge — one gentle bend</MenuItem>
                        <MenuItem onClick={() => doChance('bend')}>Bend — a couple, committed</MenuItem>
                        <MenuItem onClick={() => doChance('break')}>Break — no hypothesis, all in</MenuItem>
                    </Menu>
                    <Tooltip title={TIPS.bend.unbend}>
                        <Button size="small" variant="outlined" startIcon={<UnbendIcon size={14} />}
                                onClick={unbend} disabled={!modules.length}>
                            Unbend
                        </Button>
                    </Tooltip>
                </Box>

                {/* signal path */}
                <SignalPath registry={registry} modules={modules} onAddModule={addModule} />

                {/* rack */}
                {modules.length === 0 ? (
                    <Box sx={{
                        textAlign: 'center', py: 3, px: 2, mb: 1.5,
                        border: '1px dashed', borderColor: 'divider', borderRadius: 2.5,
                    }}>
                        <Typography variant="body2" color="textSecondary">
                            The rack is empty. Click a stage on the signal path to
                            attach a bend module — or press <b>Chance</b> and just listen.
                            No theory required; that's the point.
                        </Typography>
                    </Box>
                ) : (
                    <Box sx={{ mb: 1.5 }}>
                        {modules.map((m, i) => (
                            <BendModuleCard
                                key={m.id}
                                module={m}
                                registry={registry}
                                onChange={(next) => updateModule(m.id, next)}
                                onRemove={() => removeModule(m.id)}
                                onRandomize={() => randomizeModule(m.id)}
                                onMoveUp={() => moveModule(m.id, -1)}
                                onMoveDown={() => moveModule(m.id, +1)}
                                isFirst={i === 0}
                                isLast={i === modules.length - 1}
                            />
                        ))}
                    </Box>
                )}

                {/* generate strip */}
                <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center', mb: 1 }}>
                    <TextField
                        fullWidth multiline minRows={1} maxRows={3}
                        label="Prompt" value={prompt}
                        placeholder="Describe the sound — then bend what the model does with it…"
                        onChange={(e) => setPrompt(e.target.value)}
                    />
                </Box>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center', mb: 1.5 }}>
                    <Tooltip title={TIPS.bend.duration}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 170 }}>
                            <Typography variant="caption" color="textSecondary">dur</Typography>
                            <Slider size="small" value={duration} min={1} max={30} step={1}
                                    onChange={(_, v) => setDuration(v)} valueLabelDisplay="auto"
                                    sx={{ width: 100 }} />
                            <Typography variant="caption">{duration}s</Typography>
                        </Box>
                    </Tooltip>
                    {modelId.endsWith('-base') && (
                        <TextField size="small" label="Steps" type="number"
                                   value={steps ?? ''} placeholder="50"
                                   onChange={(e) => setSteps(e.target.value ? parseInt(e.target.value, 10) : null)}
                                   sx={{ width: 90 }} />
                    )}
                    <FormControlLabel
                        control={<Checkbox size="small" checked={randomSeed}
                                           onChange={(e) => setRandomSeed(e.target.checked)} />}
                        label={<Typography variant="caption">Random seed</Typography>}
                    />
                    {!randomSeed && (
                        <TextField size="small" label="Seed" type="number" value={seed}
                                   onChange={(e) => setSeed(e.target.value)} sx={{ width: 120 }} />
                    )}
                    <Box sx={{ flex: 1 }} />
                    {busy ? (
                        <Button variant="outlined" color="error" startIcon={<StopIcon size={14} />}
                                onClick={stopGeneration}>
                            Stop
                        </Button>
                    ) : (
                        <>
                            <Tooltip title={TIPS.bend.abCompare}>
                                <span>
                                    <Button variant="outlined" disabled={!bentAudio}
                                            onClick={() => generate(false)}>
                                        Clean A/B
                                    </Button>
                                </span>
                            </Tooltip>
                            <Tooltip title={TIPS.bend.generate}>
                                <span>
                                    <Button variant="contained" color="bend"
                                            disabled={!activeCount || !modelId}
                                            onClick={() => generate(true)}>
                                        Generate bent
                                    </Button>
                                </span>
                            </Tooltip>
                        </>
                    )}
                </Box>

                {busy && <LinearProgress variant="determinate" value={progress} sx={{ mb: 1.5 }} />}
                {bendWarnings.length > 0 && (
                    <Alert severity="warning" onClose={() => setBendWarnings([])} sx={{ mb: 1.5 }}>
                        {bendWarnings.map((w, i) => (
                            <Typography key={i} variant="body2">{w}</Typography>
                        ))}
                    </Alert>
                )}
                {statusMsg && (
                    <Typography variant="caption" color="textSecondary"
                                sx={{ display: 'block', mb: 1.5 }}>
                        {statusMsg}
                    </Typography>
                )}

                {/* results / A-B — the app's fragment-player idiom */}
                {(bentAudio || cleanAudio) && (
                    <BendTakes
                        isDocker={isDocker}
                        onMessage={setStatusMsg}
                        takes={[
                            bentAudio && {
                                ...bentAudio, key: 'bent', title: 'Bent',
                                color: accent, titleColor: accent,
                            },
                            cleanAudio && { ...cleanAudio, key: 'clean', title: 'Clean' },
                        ].filter(Boolean)}
                    />
                )}

                {/* Bending Log */}
                <Accordion disableGutters expanded={logOpen}
                           onChange={(_, v) => { setLogOpen(v); if (v) refreshLog(); }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon size={18} />}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <LogIcon size={16} />
                            <Typography variant="subtitle1">Bending Log</Typography>
                            {logEntries.length > 0 && (
                                <Chip size="small" label={logEntries.length}
                                      sx={{ height: 18, fontSize: '0.65rem' }} />
                            )}
                        </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                        <BendingLogPanel
                            entries={logEntries}
                            registry={registry}
                            onRecall={recallLogEntry}
                            onChanged={refreshLog}
                        />
                    </AccordionDetails>
                </Accordion>
            </>
            )}
        </Paper>
    );
}
