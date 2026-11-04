import React, { useState } from 'react';
import { Box, ButtonBase, Collapse, Stack, Typography } from '@mui/material';
import { ChevronDown as ChevronDownIcon } from 'lucide-react';
import { datasetStyles } from '../theme';

/**
 * One collapsible band in the inspector rail. Kept deliberately lighter than
 * MUI's Accordion — the rail holds several of these and the Accordion's own
 * padding and dividers stack up into exactly the bulk the rail exists to
 * avoid.
 */
export default function RailSection({ title, icon, defaultOpen = true, badge, children }) {
    const [open, setOpen] = useState(defaultOpen);
    return (
        <Box sx={datasetStyles.railCard}>
            <ButtonBase
                onClick={() => setOpen((o) => !o)}
                aria-expanded={open}
                sx={datasetStyles.railHeader}
            >
                <Typography component="h3" variant="subtitle1" sx={datasetStyles.railTitle}>
                    {title}
                </Typography>
                {badge}
                <Box
                    component="span"
                    sx={{
                        display: 'flex',
                        color: 'text.disabled',
                        transform: open ? 'rotate(180deg)' : 'none',
                        transition: 'transform .18s',
                    }}
                >
                    <ChevronDownIcon size={15} />
                </Box>
            </ButtonBase>
            <Collapse in={open} unmountOnExit>
                <Stack spacing={1.5} sx={{ px: 2.25, pb: 2.25, pt: 0.25 }}>
                    {children}
                </Stack>
            </Collapse>
        </Box>
    );
}

