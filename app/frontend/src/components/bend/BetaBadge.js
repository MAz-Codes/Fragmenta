import React from 'react';
import { Box } from '@mui/material';
import { alpha } from '@mui/material/styles';

/**
 * Small blue BETA pill for the Bend tab. Uses the secondary blue so it
 * reads as a label, not an action (cyan) or an automatic function (warm).
 * `compact` is the icon-only sidebar variant, overlaid on the tab icon.
 */
export default function BetaBadge({ compact = false, sx }) {
    return (
        <Box
            component="span"
            aria-label="beta"
            sx={[
                (theme) => ({
                    display: 'inline-flex',
                    alignItems: 'center',
                    fontFamily: theme.typography.overline.fontFamily,
                    fontSize: compact ? '0.45rem' : '0.55rem',
                    fontWeight: 600,
                    letterSpacing: '0.1em',
                    lineHeight: 1,
                    px: compact ? 0.4 : 0.65,
                    py: compact ? 0.2 : 0.3,
                    borderRadius: 999,
                    color: theme.palette.mode === 'dark'
                        ? theme.palette.secondary.light
                        : theme.palette.secondary.main,
                    backgroundColor: alpha(theme.palette.secondary.main, 0.14),
                    border: `1px solid ${alpha(theme.palette.secondary.main, 0.5)}`,
                    pointerEvents: 'none',
                    userSelect: 'none',
                }),
                ...(Array.isArray(sx) ? sx : [sx]),
            ]}
        >
            BETA
        </Box>
    );
}
