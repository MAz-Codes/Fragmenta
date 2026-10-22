import React from 'react';
import {
    Box,
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Typography,
} from '@mui/material';
import {
    Info as InfoIcon,
    BookOpen as BookOpenIcon,
} from 'lucide-react';
import { appStyles } from '../theme';
import { APP_VERSION } from '../version';

const LICENSES_URL = 'https://github.com/MAz-Codes/Fragmenta/tree/main/licenses';
const NOTICE_SX = { display: 'block', fontSize: '0.65rem', lineHeight: 1.5, mt: 0.5 };
const LINK_SX = { color: 'primary.main', textDecoration: 'underline', fontSize: '0.65rem' };

/**
 * "About Fragmenta" dialog — logo + title, short intro, three doc buttons
 * (About / Documentation / Tutorials), and the model-license attribution
 * footer (Stability AI + Gemma notices).
 *
 * Props:
 *   open:                       bool
 *   onClose:                    () => void
 *   onOpenDocumentation:        ('about' | 'documentation') => void
 *   isOpeningDocumentation:     bool — disables the doc buttons while a
 *                               native open-file call is in flight
 */
export default function AboutDialog({
    open,
    onClose,
    onOpenDocumentation,
    isOpeningDocumentation,
}) {
    return (
        <Dialog
            open={open}
            onClose={onClose}
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
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem', letterSpacing: '0.04em' }}>
                        v{APP_VERSION}
                    </Typography>
                </Box>
            </DialogTitle>
            <DialogContent>
                <Typography sx={appStyles.infoDialogIntro}>
                    Fragmenta is an open source, local-first suit to prepare datasets, train, generate and perform with text-to-audio diffusion models.
                    Made by the composer and researcher Misagh Azimi.
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
                        disabled
                        sx={appStyles.infoDocButton}
                    >
                        Tutorials (Coming soon...)
                    </Button>
                </Box>

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
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Close</Button>
            </DialogActions>
        </Dialog>
    );
}
