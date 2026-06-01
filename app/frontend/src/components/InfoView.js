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
    // Only present when there's something to say — no card, no placeholder.
    if (!hint) return null;

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
                display: 'flex',
                alignItems: 'center',
                gap: 1.25,
                pointerEvents: 'none',  // pure overlay — never intercepts clicks
                // Clear the bottom-left floating dock column on the left.
                pl: { xs: '64px', sm: '76px', md: '88px' },
                pr: { xs: 2, sm: 3 },
                py: 1,
                // No background card — text reads directly over the page. A
                // soft shadow keeps it legible over arbitrary content.
                textShadow: theme.palette.mode === 'dark'
                    ? '0 1px 3px rgba(0,0,0,0.85), 0 0 10px rgba(0,0,0,0.6)'
                    : '0 1px 3px rgba(242,237,227,0.95), 0 0 10px rgba(242,237,227,0.8)',
            })}
        >
            <Box component="span" sx={{ flexShrink: 0, display: 'inline-flex', color: 'primary.main' }}>
                <InfoIcon size={16} />
            </Box>
            <Typography variant="body2" sx={{ color: 'text.primary', lineHeight: 1.35 }}>
                {hint}
            </Typography>
        </Box>
    );
}
