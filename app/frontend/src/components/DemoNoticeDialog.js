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
    ExternalLink as ExternalLinkIcon,
} from 'lucide-react';
import { appStyles, demoNoticeStyles } from '../theme';

export const FRAGMENTA_SITE_URL = 'https://www.misaghazimi.com/fragmenta';
const LICENSES_URL = 'https://github.com/MAz-Codes/Fragmenta/blob/main/licenses';

/**
 * Demo notice — shown once per session on the Hugging Face Spaces deployment
 * only (GET /api/environment → hf_space), right after the welcome page is
 * dismissed. Makes it unmistakable that the Space is a demonstration and
 * points at the real install.
 *
 * Never rendered on desktop or the regular Docker images: App.js gates it on
 * the hf_space flag.
 *
 * Props:
 *   open:    bool
 *   onClose: () => void
 */
export default function DemoNoticeDialog({ open, onClose }) {
    return (
        <Dialog
            open={open}
            // Deliberately no onClose handler: this is a notice, so it is
            // dismissed through the button rather than by clicking away.
            disableEscapeKeyDown
            aria-labelledby="demo-notice-dialog-title"
            maxWidth="xs"
            fullWidth
        >
            <DialogTitle id="demo-notice-dialog-title">
                <Box sx={demoNoticeStyles.titleStack}>
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
                    <Box sx={demoNoticeStyles.badge}>Demo Space</Box>
                </Box>
            </DialogTitle>

            <DialogContent>
                <Typography sx={demoNoticeStyles.body}>
                    This space is for demonstration purposes only and is not a fully
                    functional app.
                </Typography>

                <Typography sx={demoNoticeStyles.subBody}>
                    To install and use the app locally, please visit:
                </Typography>

                <Box sx={appStyles.infoDialogActionStack}>
                    <Button
                        variant="contained"
                        size="small"
                        component="a"
                        href={FRAGMENTA_SITE_URL}
                        target="_blank"
                        rel="noopener noreferrer"
                        endIcon={<ExternalLinkIcon size={14} />}
                        sx={demoNoticeStyles.linkButton}
                    >
                        misaghazimi.com/fragmenta
                    </Button>
                </Box>

                {/* The Space is a hosted service of the models, so the Gemma
                    Terms (§3.1) want the use restrictions agreed to by its
                    users; continuing is that agreement. */}
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2.5, lineHeight: 1.6, textAlign: 'center', '& a': { color: 'primary.dark' } }}>
                    By continuing you agree to the{' '}
                    <a href={`${LICENSES_URL}/STABILITY_AI_COMMUNITY_LICENSE.md`} target="_blank" rel="noopener noreferrer">Stability AI Community License</a>,
                    the <a href={`${LICENSES_URL}/GEMMA_TERMS_OF_USE.md`} target="_blank" rel="noopener noreferrer">Gemma Terms of Use</a> and
                    their use policies. Audio made here is AI-generated and marked as such. Powered by Stability AI.
                </Typography>
            </DialogContent>

            <DialogActions sx={demoNoticeStyles.actions}>
                <Button variant="outlined" size="small" onClick={onClose} sx={appStyles.infoDocButton}>
                    Continue to the demo
                </Button>
            </DialogActions>
        </Dialog>
    );
}
