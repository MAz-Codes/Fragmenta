import React from 'react';
import { Box, Typography, Badge, ButtonBase, useTheme } from '@mui/material';
import { Plus as PlusIcon } from 'lucide-react';
import Tooltip from '../Tooltip';
import { STAGE_ORDER, STAGE_SHORT } from './bendUtils';

/**
 * SignalPath — the model drawn as a horizontal pipeline the user patches
 * into. Always visible, so the user always knows WHERE they are in the
 * network (Brave's user study: people got lost without a location
 * indicator). Stages with attached modules glow with the prism accent —
 * the app-wide "something is armed/active" color.
 *
 *   Prompt → [Text enc] → [Cond ●] → [DiT ▦] → [Sampler] → [VAE] → Audio
 */
export default function SignalPath({ registry, modules, onAddModule }) {
    const theme = useTheme();
    const prism = theme.palette.prism?.main || '#C27CF2';

    const countFor = (stage) =>
        modules.filter(m => m.target?.stage === stage && m.enabled !== false).length;

    const stages = STAGE_ORDER
        .map(s => (registry?.stages || []).find(x => x.stage === s))
        .filter(Boolean);

    const Arrow = () => (
        <Typography component="span" sx={{
            color: 'text.disabled', fontSize: '0.9rem', px: { xs: 0.25, sm: 0.75 },
            userSelect: 'none', flexShrink: 0,
        }}>→</Typography>
    );

    const EndCap = ({ children }) => (
        <Typography variant="caption" sx={{
            color: 'text.secondary', whiteSpace: 'nowrap', flexShrink: 0,
            display: { xs: 'none', sm: 'block' },
        }}>{children}</Typography>
    );

    return (
        <Box sx={{
            display: 'flex', alignItems: 'center', flexWrap: 'wrap',
            gap: 0.25, rowGap: 1, py: 1.5, px: { xs: 0.5, sm: 1 },
        }}>
            <EndCap>Prompt</EndCap>
            <Arrow />
            {stages.map((stageInfo, i) => {
                const n = countFor(stageInfo.stage);
                const active = n > 0;
                const label = STAGE_SHORT[stageInfo.stage] || stageInfo.label;
                const sub = stageInfo.kind === 'blocks'
                    ? `${stageInfo.count} block${stageInfo.count === 1 ? '' : 's'}` : stageInfo.label;
                return (
                    <React.Fragment key={stageInfo.stage}>
                        {i > 0 && <Arrow />}
                        <Tooltip title={`${stageInfo.hint} Click to attach a bend module.`}>
                            <Badge
                                badgeContent={n || null}
                                sx={{ '& .MuiBadge-badge': {
                                    backgroundColor: prism, color: '#1A0F00',
                                    fontWeight: 600,
                                } }}
                            >
                                <ButtonBase
                                    onClick={() => onAddModule(stageInfo.stage)}
                                    sx={{
                                        flexDirection: 'column', px: 1.25, py: 0.75,
                                        borderRadius: 2.5, minWidth: 64,
                                        border: '1px solid',
                                        borderColor: active ? prism : 'divider',
                                        boxShadow: active ? `0 0 12px ${prism}55, inset 0 0 6px ${prism}22` : 'none',
                                        transition: 'border-color 220ms ease, box-shadow 220ms ease',
                                        '&:hover': {
                                            borderColor: active ? prism : 'primary.main',
                                            '& .bend-add-icon': { opacity: 1 },
                                        },
                                    }}
                                >
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <Typography variant="subtitle2" sx={{
                                            color: active ? prism : 'text.primary',
                                            lineHeight: 1.3, textTransform: 'none',
                                            fontSize: '0.85rem',
                                        }}>
                                            {label}
                                        </Typography>
                                        <Box className="bend-add-icon" sx={{
                                            opacity: 0, transition: 'opacity 160ms ease',
                                            display: 'flex', color: 'primary.main',
                                        }}>
                                            <PlusIcon size={12} />
                                        </Box>
                                    </Box>
                                    <Typography variant="caption" sx={{
                                        color: 'text.disabled', fontSize: '0.62rem', lineHeight: 1.2,
                                    }}>
                                        {sub}
                                    </Typography>
                                </ButtonBase>
                            </Badge>
                        </Tooltip>
                    </React.Fragment>
                );
            })}
            <Arrow />
            <EndCap>Audio</EndCap>
        </Box>
    );
}
