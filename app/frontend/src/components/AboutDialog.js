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

/**
 * "About Fragmenta" dialog — logo + title, short intro, three doc buttons
 * (About / Documentation / Tutorials), and the Stability AI Community
 * License attribution footer.
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

                <Box sx={{ mt: 3, pt: 1.5, borderTop: '1px solid', borderColor: 'divider', textAlign: 'center' }}>
                    <Typography variant="caption" color="textSecondary" sx={{ display: 'block', fontStyle: 'italic', fontSize: '0.6rem', lineHeight: 1.5 }}>
                        Powered by{' '}
                        <Typography
                            component="a"
                            variant="caption"
                            href="https://github.com/Stability-AI/stable-audio-3"
                            target="_blank"
                            rel="noopener noreferrer"
                            sx={{ color: 'primary.main', textDecoration: 'underline', fontStyle: 'italic', fontSize: '0.6rem' }}
                        >
                            Stable Audio 3
                        </Typography>{' '}by Stability AI. "This Stability AI Model is licensed under the{' '}
                        <Typography
                            component="a"
                            variant="caption"
                            href="https://stability.ai/license"
                            target="_blank"
                            rel="noopener noreferrer"
                            sx={{ color: 'primary.main', textDecoration: 'underline', fontStyle: 'italic', fontSize: '0.6rem' }}
                        >
                            Stability AI Community License
                        </Typography>,{' '}
                        Copyright © Stability AI Ltd. All Rights Reserved"
                    </Typography>
                </Box>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Close</Button>
            </DialogActions>
        </Dialog>
    );
}
