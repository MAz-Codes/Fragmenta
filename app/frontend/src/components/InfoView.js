import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';
import { Box, Typography } from '@mui/material';
import { Info as InfoIcon } from 'lucide-react';

/**
 * Ableton-style "Info View".
 *
 * A toggleable strip pinned to the bottom of the window that shows the help
 * text for whatever control the pointer (or keyboard focus) is over, instead
 * of popping a tooltip on the control itself. The shared <Tooltip> feeds this
 * panel when the view is enabled (see components/Tooltip.js).
 *
 * State design: `enabled` is owned by App (changes rarely, persisted). The
 * *hint* — which changes on every hover — lives inside the provider and is
 * read only by the bar, so updating it never re-renders the app tree (the app
 * is passed as `children`, whose element identity is stable across the
 * provider's internal state changes).
 */
export const InfoViewContext = createContext({ enabled: false, setHint: () => {} });

export const useInfoView = () => useContext(InfoViewContext);

export function InfoViewProvider({ enabled, children }) {
    const [hint, setHint] = useState(null);
    // Stable setter so the context value only changes when `enabled` flips —
    // hover-driven hint updates don't churn every tooltip consumer.
    const update = useCallback((value) => setHint(value ?? null), []);
    const value = useMemo(() => ({ enabled, setHint: update }), [enabled, update]);

    return (
        <InfoViewContext.Provider value={value}>
            {children}
            {enabled && <InfoViewBar hint={hint} />}
        </InfoViewContext.Provider>
    );
}

function InfoViewBar({ hint }) {
    // Only present when there's something to say — no placeholder.
    if (!hint) return null;

    return (
        // Full-width fixed row that centers the pill at the bottom of the page.
        <Box
            sx={{
                position: 'fixed',
                left: 0,
                right: 0,
                bottom: { xs: 16, md: 24 },
                zIndex: 1340,           // under the bottom dock (1350)
                px: 2,
                display: 'flex',
                justifyContent: 'center',
                pointerEvents: 'none',  // pure overlay — never intercepts clicks
            }}
        >
            <Box
                role="status"
                aria-live="polite"
                sx={(theme) => ({
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 1,
                    maxWidth: 'min(680px, 90vw)',
                    px: 1.75,
                    py: 0.9,
                    borderRadius: 999,
                    // Blurred translucent pill — just enough backing for the
                    // text to stay readable over any content behind it.
                    backgroundColor: theme.palette.mode === 'dark'
                        ? 'rgba(20, 22, 24, 0.55)'
                        : 'rgba(248, 243, 234, 0.62)',
                    backdropFilter: 'blur(16px) saturate(160%)',
                    WebkitBackdropFilter: 'blur(16px) saturate(160%)',
                    border: `1px solid ${theme.palette.divider}`,
                    boxShadow: theme.palette.mode === 'dark'
                        ? '0 8px 28px rgba(0,0,0,0.5)'
                        : '0 8px 28px rgba(43,31,18,0.16)',
                    animation: 'fragmenta-fade-up 240ms cubic-bezier(0.16, 1, 0.3, 1)',
                })}
            >
                <Box component="span" sx={{ flexShrink: 0, display: 'inline-flex', color: 'primary.main' }}>
                    <InfoIcon size={15} />
                </Box>
                <Typography variant="body2" sx={{ color: 'text.primary', lineHeight: 1.3 }}>
                    {hint}
                </Typography>
            </Box>
        </Box>
    );
}
