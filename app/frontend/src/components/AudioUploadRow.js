import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, Grid, Box, Typography, TextField, IconButton } from '@mui/material';
import { Upload as UploadIcon, Trash2 as DeleteIcon } from 'lucide-react';
import { audioUploadRowStyles } from '../theme';

export default function AudioUploadRow({ index, data, onChange, onRemove }) {
    const [audioFile, setAudioFile] = useState(null);
    const [audioUrl, setAudioUrl] = useState('');
    const [isDragActive, setIsDragActive] = useState(false);
    const inputRef = useRef(null);

    useEffect(() => {
        if (!data.file && !data.audioUrl) {
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
            setAudioFile(null);
            setAudioUrl('');
        }
    }, [data.file, data.audioUrl, audioUrl]);

    const acceptFile = (file) => {
        if (!file || !file.type.startsWith('audio/')) return;
        const url = URL.createObjectURL(file);
        setAudioFile(file);
        setAudioUrl(url);
        onChange(index, { ...data, file, audioUrl: url });
    };

    return (
        <Card sx={audioUploadRowStyles.card}>
            <CardContent sx={audioUploadRowStyles.cardContent}>
                <Grid container spacing={audioUploadRowStyles.gridSpacing} alignItems="center">
                    <Grid item xs={12} sm={4}>
                        <Box
                            onClick={() => inputRef.current?.click()}
                            onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
                            onDragLeave={() => setIsDragActive(false)}
                            onDrop={(e) => {
                                e.preventDefault();
                                setIsDragActive(false);
                                acceptFile(e.dataTransfer.files?.[0]);
                            }}
                            sx={audioUploadRowStyles.uploadDropZone(isDragActive)}
                        >
                            <input
                                ref={inputRef}
                                type="file"
                                accept="audio/*,.mp3,.wav,.flac,.m4a,.aac"
                                style={audioUploadRowStyles.hiddenInput}
                                onChange={(e) => acceptFile(e.target.files?.[0])}
                            />
                            {audioFile ? (
                                <Box>
                                    <Typography variant="body2" color="textSecondary">
                                        {audioFile.name}
                                    </Typography>
                                    {audioUrl && (
                                        <audio
                                            controls
                                            src={audioUrl}
                                            style={audioUploadRowStyles.audioPreview}
                                        />
                                    )}
                                </Box>
                            ) : (
                                <Box>
                                    <UploadIcon size={20} color="#9198A1" />
                                    <Typography variant="body2" color="textSecondary ">
                                        {isDragActive ? 'Drop audio here' : ""}
                                    </Typography>
                                </Box>
                            )}
                        </Box>
                    </Grid>

                    <Grid item xs={12} sm={7}>
                        <TextField
                            fullWidth
                            multiline
                            minRows={1}
                            maxRows={3}
                            label={`Prompt ${index + 1}`}
                            placeholder="Describe this audio file..."
                            value={data.prompt || ''}
                            onChange={(e) => onChange(index, { ...data, prompt: e.target.value })}
                            variant="outlined"
                        />
                    </Grid>

                    <Grid item xs={12} sm={1} sx={audioUploadRowStyles.deleteGridItem}>
                        <IconButton
                            color="error"
                            onClick={() => onRemove(index)}
                            sx={audioUploadRowStyles.deleteIconButton}
                        >
                            <DeleteIcon />
                        </IconButton>
                    </Grid>
                </Grid>
            </CardContent>
        </Card>
    );
}
