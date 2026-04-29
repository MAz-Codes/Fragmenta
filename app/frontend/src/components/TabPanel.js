import React from 'react';
import { Box } from '@mui/material';
import { tabPanelStyles } from '../theme';

export default function TabPanel({ children, value, index, keepMounted = false, ...other }) {
    const isActive = value === index;
    // keepMounted: render children unconditionally and toggle visibility via CSS,
    // so component state, audio nodes, and decoded buffers survive tab switches.
    // Use sparingly — by default we still mount/unmount so inactive tabs cost
    // nothing at idle.
    if (keepMounted) {
        return (
            <div
                role="tabpanel"
                hidden={!isActive}
                id={`simple-tabpanel-${index}`}
                aria-labelledby={`simple-tab-${index}`}
                {...other}
            >
                <Box sx={{ ...tabPanelStyles.root, display: isActive ? undefined : 'none' }}>
                    {children}
                </Box>
            </div>
        );
    }

    return (
        <div
            role="tabpanel"
            hidden={!isActive}
            id={`simple-tabpanel-${index}`}
            aria-labelledby={`simple-tab-${index}`}
            {...other}
        >
            {isActive && (
                <Box sx={tabPanelStyles.root}>
                    {children}
                </Box>
            )}
        </div>
    );
}
