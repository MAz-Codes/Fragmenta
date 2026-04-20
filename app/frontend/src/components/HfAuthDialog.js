import React, { useState, useEffect } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    Typography,
    TextField,
    Box,
    CircularProgress,
    Stepper,
    Step,
    StepLabel,
    Link,
    Alert,
    LinearProgress
} from '@mui/material';
import api from '../api';
import { hfAuthDialogStyles } from '../theme';

const HfAuthDialog = ({ open, onClose, onModelsDownloaded }) => {
    const [activeStep, setActiveStep] = useState(0);
    const [missingModels, setMissingModels] = useState([]);
    const [checkingStatus, setCheckingStatus] = useState(true);
    const [token, setToken] = useState('');
    const [error, setError] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [downloadingModel, setDownloadingModel] = useState(null);

    const steps = ['Check required models', 'Authenticate', 'Download models'];

    useEffect(() => {
        if (open) {
            checkModelStatus();
        } else {
            setActiveStep(0);
            setError(null);
            setToken('');
            setMissingModels([]);
        }
    }, [open]);

    const checkModelStatus = async () => {
        setCheckingStatus(true);
        setError(null);
        try {
            const response = await api.get('/api/base-models/status');
            const models = response.data.base_models;
            const missing = Object.entries(models)
                .filter(([_, info]) => !info.downloaded)
                .map(([id, info]) => ({ id, ...info }));
            
            setMissingModels(missing);
            
            if (missing.length === 0) {
                setActiveStep(3);
            } else {
                setActiveStep(1);
            }
        } catch (err) {
            setError(err.response?.data?.error || err.message || 'Failed to check model status.');
        } finally {
            setCheckingStatus(false);
        }
    };

    const handleLogin = async () => {
        if (!token.trim()) {
            setError('Please enter a Hugging Face token.');
            return;
        }

        setIsProcessing(true);
        setError(null);
        try {
            await api.post('/api/hf-login', { token: token.trim() });

            setActiveStep(2);
            startDownloads();
        } catch (err) {
            setError(err.response?.data?.error || err.message || 'Authentication failed. Please check your token.');
            setIsProcessing(false);
        }
    };

    const startDownloads = async () => {
        try {
            for (const model of missingModels) {
                setDownloadingModel(model.name);
                await api.post(`/api/models/${model.id}/accept-terms`);
                await api.post(`/api/models/${model.id}/download`);
            }

            setActiveStep(3);
            if (onModelsDownloaded) {
                onModelsDownloaded();
            }
        } catch (err) {
            setError(err.response?.data?.error || err.message || 'Failed to download models.');
        } finally {
            setIsProcessing(false);
            setDownloadingModel(null);
        }
    };

    const handleClose = () => {
        if (isProcessing && activeStep === 2) {
            return;
        }
        onClose(activeStep === 3);
    };

    const getStepContent = (stepIndex) => {
        if (checkingStatus) {
            return (
                <Box sx={hfAuthDialogStyles.checkingBox}>
                    <CircularProgress sx={hfAuthDialogStyles.checkingProgress} />
                    <Typography>Checking model availability...</Typography>
                </Box>
            );
        }

        switch (stepIndex) {
            case 1:
                return (
                    <Box sx={hfAuthDialogStyles.authStepBox}>
                        <Typography variant="body1" paragraph>
                            Some required base models are missing. You need a Hugging Face access token to download them.
                        </Typography>
                        
                        <Typography variant="body2" color="textSecondary" paragraph>
                            Before proceeding, please ensure you have visited huggingface.co and accepted the terms of use for the required models (e.g., stabilityai/stable-audio-open-1.0).
                        </Typography>
                        
                        <TextField
                            fullWidth
                            label="Hugging Face Access Token"
                            type="password"
                            value={token}
                            onChange={(e) => setToken(e.target.value)}
                            placeholder="hf_xxxxxxxxxxxxxxxxxxx..."
                            margin="normal"
                            variant="outlined"
                        />
                        <Typography variant="caption" color="textSecondary">
                            You can get a token from{' '}
                            <Link href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">
                                your Hugging Face settings
                            </Link>. "Read" access to public gated repos is needed.
                        </Typography>
                    </Box>
                );
            case 2:
                return (
                    <Box sx={hfAuthDialogStyles.downloadStepBox}>
                        <Typography variant="h6" paragraph>
                            Downloading {downloadingModel}...
                        </Typography>
                        <LinearProgress sx={hfAuthDialogStyles.downloadProgress} />
                        <Typography variant="body2" color="textSecondary">
                            This may take several minutes depending on your connection speed. Do not close the application.
                        </Typography>
                    </Box>
                );
            case 3:
                return (
                    <Box sx={hfAuthDialogStyles.successStepBox}>
                        <Typography variant="h6" color="success.main" paragraph>
                            All models are ready!
                        </Typography>
                        <Typography variant="body1">
                            You can now close this dialog and begin using Fragmenta.
                        </Typography>
                    </Box>
                );
            default:
                return "Unknown step";
        }
    };

    return (
        <Dialog 
            open={open} 
            onClose={handleClose}
            maxWidth="sm"
            fullWidth
            disableEscapeKeyDown={isProcessing && activeStep === 2}
        >
            <DialogTitle>
                Hugging Face Authentication
            </DialogTitle>
            <DialogContent dividers>
                <Stepper activeStep={activeStep} alternativeLabel sx={hfAuthDialogStyles.stepper}>
                    {steps.map((label) => (
                        <Step key={label}>
                            <StepLabel>{label}</StepLabel>
                        </Step>
                    ))}
                </Stepper>
                
                {error && (
                    <Alert severity="error" sx={hfAuthDialogStyles.errorAlert}>{error}</Alert>
                )}
                
                {getStepContent(activeStep)}
            </DialogContent>
            
            <DialogActions>
                {activeStep !== 3 && activeStep !== 2 && (
                    <Button onClick={handleClose} disabled={isProcessing}>
                        Cancel
                    </Button>
                )}
                
                {activeStep === 1 && (
                    <Button 
                        variant="contained" 
                        color="primary" 
                        onClick={handleLogin}
                        disabled={isProcessing || !token}
                    >
                        {isProcessing ? <CircularProgress size={hfAuthDialogStyles.loginSpinnerSize} /> : 'Login & Download'}
                    </Button>
                )}
                
                {activeStep === 3 && (
                    <Button variant="contained" color="primary" onClick={handleClose}>
                        Close
                    </Button>
                )}
            </DialogActions>
        </Dialog>
    );
};

export default HfAuthDialog;
