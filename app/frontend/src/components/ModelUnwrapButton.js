import React, { useState } from 'react';
import { Button, Box, Link, Typography } from '@mui/material';
import { CloudDownload as CloudDownloadIcon } from 'lucide-react';
import api from '../api';
import { modelUnwrapButtonStyles } from '../theme';

export default function ModelUnwrapButton({ model, onUnwrap, onRefresh }) {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleUnwrap = async () => {
        setLoading(true);
        setResult(null);
        setError(null);

        try {
            const response = await api.post('/api/unwrap-model', {
                model_config: model.configPath,
                ckpt_path: model.ckptPath,
                name: model.name + '_unwrapped'
            });
            setResult(response.data);
            if (onUnwrap) onUnwrap(response.data);
            if (onRefresh) onRefresh();
        } catch (err) {
            console.error('Unwrap error:', err);
            setError(err.response?.data?.error || err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box sx={modelUnwrapButtonStyles.root}>
            <Button
                variant="outlined"
                color="primary"
                size="small"
                startIcon={<CloudDownloadIcon />}
                onClick={handleUnwrap}
                disabled={loading}
            >
                {loading ? 'Unwrapping...' : 'Unwrap for Inference'}
            </Button>
            {result && result.unwrapped_path && (
                <Box sx={modelUnwrapButtonStyles.result}>
                    <Link href={`file://${result.unwrapped_path}`} target="_blank" rel="noopener noreferrer">
                        Download Unwrapped Model
                    </Link>
                </Box>
            )}
            {error && (
                <Typography sx={modelUnwrapButtonStyles.error}>{error}</Typography>
            )}
        </Box>
    );
}
