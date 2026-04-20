import React from 'react';
import { Box } from '@mui/material';
import { tabPanelStyles } from '../theme';

export default function TabPanel({ children, value, index, ...other }) {
    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`simple-tabpanel-${index}`}
            aria-labelledby={`simple-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={tabPanelStyles.root}>
                    {children}
                </Box>
            )}
        </div>
    );
}
