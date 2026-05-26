import React, { useState, useEffect } from 'react';
import { Backdrop, Box, Fade, Typography, Button, Checkbox, FormControlLabel, Stack } from '@mui/material';
import { welcomePageStyles } from '../theme';
import { APP_VERSION } from '../version';

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
                    <Stack alignItems="center">
                        <Stack alignItems="center">
                            <Typography variant="body3" color="text.secondary">
                                ©2025-2026 Misagh Azimi
                            </Typography>
                            <Typography variant="body3" color="text.secondary">
                                Version {APP_VERSION}
                            </Typography>
                        </Stack>
                        <Box mt={5}>
                            <Button
                                variant="contained"
                                onClick={() => onClose(dontShowAgain)}
                            >
                                Get Started
                            </Button>
                        </Box>
                        <Box mt={6}>
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={dontShowAgain}
                                        onChange={(e) => setDontShowAgain(e.target.checked)}
                                        size="small"
                                    />
                                }
                                label={
                                    <Typography variant="caption" color="text.secondary">
                                        Don't show this again
                                    </Typography>
                                }
                            />
                        </Box>
                    </Stack>
                </Fade>
            </Box>
        </Backdrop>
    );
}
