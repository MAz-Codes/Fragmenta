import React from 'react';
import {
    Box,
    Button,
    Checkbox,
    FormControlLabel,
    Typography,
} from '@mui/material';
import {
    Info as InfoIcon,
    BookOpen as BookOpenIcon,
    PlayCircle as PlayIcon,
} from 'lucide-react';
import { appStyles } from '../theme';
import Tooltip from './Tooltip';
import { APP_VERSION } from '../version';

const LICENSES_URL = 'https://github.com/MAz-Codes/Fragmenta/tree/main/licenses';
const NOTICE_SX = { display: 'block', fontSize: '0.65rem', lineHeight: 1.5, mt: 0.5 };
const LINK_SX = { color: 'primary.dark', textDecoration: 'underline', fontSize: '0.65rem' };

/**
 * "About Fragmenta" — logo + title, short intro, three doc buttons
 * (About / Documentation / Tutorials), updates, and the model-license
 * attribution footer (Stability AI + Gemma notices).
 *
 * Props:
 *   onOpenDocumentation:        ('about' | 'documentation' | 'tutorials') => void
 *   isOpeningDocumentation:     bool — disables the doc buttons while a
 *                               native open-file call is in flight
 *   update:                     useUpdateStatus() result (optional)
 *   onShowUpdate:               () => void — open the update dialog
 */
function updateLine(s) {
    if (s.disabled_reason === 'env') return 'Update checks are turned off (FRAGMENTA_NO_UPDATE_CHECK).';
    if (s.available) return `Update available: ${s.latest}`;
    if (s.checked_at) return 'Up to date';
    if (s.last_error) return "Couldn't check for updates (offline?)";
    return 'Not checked for updates yet';
}

// The About page: the first page of the Settings dialog (gear in the top
// toolbar), shown exactly as the old About dialog.
export function AboutPanel({
    onOpenDocumentation,
    isOpeningDocumentation,
    update,
    onShowUpdate,
}) {
    const us = update && update.status;
    const showUpdates = Boolean(us && us.disabled_reason !== 'hf_space');
    return (
        <Box>
            <Box sx={{ pb: 1 }}>
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
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem', letterSpacing: '0.04em' }}>
                        v{APP_VERSION}
                    </Typography>
                </Box>
            </Box>
                <Typography sx={appStyles.infoDialogIntro}>
                    Fragmenta is an open source, local-first suit to prepare datasets, train, generate and perform with text-to-audio diffusion models.
                    Made by the composer and researcher{' '}
                    <Typography component="a" href="https://www.misaghazimi.com" target="_blank" rel="noopener noreferrer"
                        sx={{ font: 'inherit', color: 'primary.dark', textDecoration: 'underline' }}>
                        Misagh Azimi
                    </Typography>.
                </Typography>

                <Box sx={appStyles.infoDialogActionStack}>
                    <Button
                        variant="contained"
                        size="small"
                        startIcon={<InfoIcon size={16} />}
                        onClick={() => onOpenDocumentation('about')}
                        disabled={isOpeningDocumentation}
                        sx={appStyles.infoDocButton}
                    >
                        About
                    </Button>
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={<BookOpenIcon size={16} />}
                        onClick={() => onOpenDocumentation('documentation')}
                        disabled={isOpeningDocumentation}
                        sx={appStyles.infoDocButton}
                    >
                        Documentation
                    </Button>
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={<PlayIcon size={16} />}
                        onClick={() => onOpenDocumentation('tutorials')}
                        disabled={isOpeningDocumentation}
                        sx={appStyles.infoDocButton}
                    >
                        Tutorials
                    </Button>
                </Box>

                {/* Updates: status on the left; the automatic-check box and
                    Check now on the right. Unticking only stops the daily
                    check — Check now always works. */}
                {showUpdates && (
                    <Box sx={{
                        mt: 3, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1,
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, flexWrap: 'wrap',
                    }}>
                        <Box sx={{ minWidth: 0 }}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {updateLine(us)}
                                {us.available && (
                                    <Button size="small" onClick={onShowUpdate}
                                        sx={{ ml: 1, minWidth: 0, py: 0, px: 0.5, verticalAlign: 'baseline' }}>
                                        Details
                                    </Button>
                                )}
                            </Typography>
                            {us.checked_at && (
                                <Typography variant="caption" color="text.secondary">
                                    Last checked {new Date(us.checked_at * 1000).toLocaleString(undefined,
                                        { dateStyle: 'medium', timeStyle: 'short' })}
                                </Typography>
                            )}
                        </Box>
                        {us.disabled_reason !== 'env' && (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexShrink: 0 }}>
                                <Tooltip title="Asks GitHub once a day. No personal data is sent.">
                                    <FormControlLabel
                                        sx={{ mr: 0 }}
                                        control={<Checkbox size="small" checked={Boolean(us.enabled)}
                                            onChange={(e) => update.setSettings({ enabled: e.target.checked })} />}
                                        label={<Typography variant="caption">Check automatically</Typography>}
                                    />
                                </Tooltip>
                                <Button size="small" variant="outlined"
                                    onClick={update.checkNow} disabled={update.checking}>
                                    {update.checking ? 'Checking…' : 'Check now'}
                                </Button>
                            </Box>
                        )}
                    </Box>
                )}

                {/* Attribution the model licenses require: "Powered by Stability AI"
                    verbatim and visible (Community License §IV(a)), its notice
                    sentence, and Google's Gemma notice for the bundled T5Gemma
                    text encoder. Full texts: licenses/ in the repo. */}
                <Box sx={{ mt: 3, pt: 1.5, borderTop: '1px solid', borderColor: 'divider', textAlign: 'center' }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        Powered by Stability AI
                    </Typography>
                    <Typography variant="caption" color="textSecondary" sx={NOTICE_SX}>
                        "This Stability AI Model is licensed under the{' '}
                        <Typography component="a" variant="caption" href="https://stability.ai/community-license-agreement"
                            target="_blank" rel="noopener noreferrer" sx={LINK_SX}>
                            Stability AI Community License
                        </Typography>, Copyright © Stability AI Ltd. All Rights Reserved"
                    </Typography>
                    <Typography variant="caption" color="textSecondary" sx={NOTICE_SX}>
                        Includes Google's T5Gemma text encoder. Gemma is provided under and subject to the
                        Gemma Terms of Use found at{' '}
                        <Typography component="a" variant="caption" href="https://ai.google.dev/gemma/terms"
                            target="_blank" rel="noopener noreferrer" sx={LINK_SX}>
                            ai.google.dev/gemma/terms
                        </Typography>
                    </Typography>
                    <Typography variant="caption" sx={NOTICE_SX}>
                        <Typography component="a" variant="caption" href={LICENSES_URL}
                            target="_blank" rel="noopener noreferrer" sx={LINK_SX}>
                            Licenses and use policies
                        </Typography>
                    </Typography>
                </Box>
        </Box>
    );
}
