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
    return (
        <Box
            role="status"
            aria-live="polite"
            sx={(theme) => ({
                position: 'fixed',
                left: 0,
                right: 0,
                bottom: 0,
                zIndex: 1340,           // under the bottom dock (1350) so the dock floats above its left edge
                minHeight: 44,
                display: 'flex',
                alignItems: 'center',
                gap: 1.25,
                // Clear the bottom-left floating dock column on the left.
                pl: { xs: '64px', sm: '76px', md: '88px' },
                pr: { xs: 2, sm: 3 },
                py: 1,
                borderTop: `1px solid ${theme.palette.divider}`,
                backgroundColor: theme.palette.mode === 'dark'
                    ? 'rgba(24, 26, 27, 0.82)'
                    : 'rgba(242, 237, 227, 0.88)',
                backdropFilter: 'blur(18px) saturate(160%)',
                WebkitBackdropFilter: 'blur(18px) saturate(160%)',
                boxShadow: theme.palette.mode === 'dark'
                    ? '0 -8px 24px rgba(0,0,0,0.45)'
                    : '0 -8px 24px rgba(43,31,18,0.12)',
            })}
        >
            <Box
                component="span"
                sx={{
                    flexShrink: 0,
                    display: 'inline-flex',
                    color: hint ? 'primary.main' : 'text.disabled',
                }}
            >
                <InfoIcon size={16} />
            </Box>
            <Typography
                variant="body2"
                sx={{
                    color: hint ? 'text.primary' : 'text.disabled',
                    fontStyle: hint ? 'normal' : 'italic',
                    lineHeight: 1.35,
                }}
            >
                {hint || 'Info View — hover any control to see what it does.'}
            </Typography>
        </Box>
    );
}
