import React, { useState, useEffect } from 'react';
import { Backdrop, Box, Fade, Typography, Button, Checkbox, FormControlLabel } from '@mui/material';
import { welcomePageStyles } from '../theme';

export default function WelcomePage({ open, onClose }) {
    const [titleVisible, setTitleVisible] = useState(false);
    const [textVisible, setTextVisible] = useState(false);
    const [dontShowAgain, setDontShowAgain] = useState(false);

    useEffect(() => {
        if (open) {
            const titleTimer = setTimeout(() => setTitleVisible(true), 500);
            const textTimer = setTimeout(() => setTextVisible(true), 1500);
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
            onClick={() => onClose(false)}
            sx={welcomePageStyles.backdrop}
        >
            <Box
                sx={welcomePageStyles.panel}
                onClick={(e) => e.stopPropagation()}
            >
                <Fade in={titleVisible} timeout={800}>
                    <Box sx={welcomePageStyles.logo} />
                </Fade>

                <Fade in={titleVisible} timeout={1000}>
                    <Typography
                        variant="h2"
                        component="h1"
                        sx={welcomePageStyles.title}
                    >
                        Welcome to Fragmenta
                    </Typography>
                </Fade>

                <Fade in={textVisible} timeout={1000}>
                    <Box>
                        <Typography
                            variant="overline"
                            sx={welcomePageStyles.overline}
                        >
                            An End-to-End Pipeline to Fine-Tune and Use Text-to-Audio Models.
                        </Typography>


                        <Typography
                            variant="body2"
                            sx={welcomePageStyles.footer}
                        >
                            @2025-2026 Misagh Azimi
                        </Typography>
                        <Typography
                            variant="body2"
                            sx={welcomePageStyles.version}
                        >
                            Version 0.1.1
                        </Typography>
                        <Button
                            variant="contained"
                            onClick={() => onClose(dontShowAgain)}
                            sx={welcomePageStyles.ctaButton}
                        >
                            Get Started
                        </Button>
                        <Box sx={{ mt: 1.5 }}>
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={dontShowAgain}
                                        onChange={(e) => setDontShowAgain(e.target.checked)}
                                        size="small"
                                        sx={{ color: 'text.secondary' }}
                                    />
                                }
                                label={
                                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                        Don't show this again
                                    </Typography>
                                }
                            />
                        </Box>

                    </Box>
                </Fade>
            </Box>
        </Backdrop>
    );
}
