import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Box, Typography } from '@mui/material';
import api from '../api';
import CheckpointRow from './CheckpointRow';

/**
 * Shown under the Training tab's base-model picker while that model isn't on
 * disk: the Checkpoint Manager's own row, so getting it works the same as there.
 */
export default function TrainingModelOffer({ modelId, onInstalled }) {
    const [offer, setOffer] = useState(null);
    const onInstalledRef = useRef(onInstalled);
    onInstalledRef.current = onInstalled;

    useEffect(() => {
        let cancelled = false;
        Promise.all([
            api.get(`/api/checkpoints/${modelId}`),
            api.get('/api/environment'),
            api.get('/api/starter'),
        ]).then(([checkpoint, env, terms]) => {
            if (!cancelled) setOffer({ checkpoint: checkpoint.data, env: env.data, terms: terms.data });
        }).catch(() => { /* the picker's download icon still works */ });
        return () => { cancelled = true; };
    }, [modelId]);

    const refresh = useCallback(async () => {
        const { data } = await api.get(`/api/checkpoints/${modelId}`);
        setOffer((current) => ({ ...current, checkpoint: data }));
        if (data.downloaded) onInstalledRef.current();
    }, [modelId]);

    if (!offer || offer.checkpoint.id !== modelId || offer.checkpoint.downloaded) return null;
    return (
        <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary">Training needs this model.</Typography>
            <CheckpointRow checkpoint={offer.checkpoint} env={offer.env} terms={offer.terms} onChanged={refresh} />
        </Box>
    );
}
