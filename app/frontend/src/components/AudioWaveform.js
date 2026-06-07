import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Box, Typography } from '@mui/material';

/**
 * Canvas waveform with a single draggable region (for SA3 inpaint UX).
 *
 * Decodes the supplied File via the Web Audio API (no network round-trip),
 * computes per-pixel min/max peaks once per (file, width) pair, and renders
 * a region overlay + two draggable handles. Region drag in three modes:
 *   - drag the left handle  → adjust start
 *   - drag the right handle → adjust end
 *   - drag the body         → shift the whole region in place
 *
 * Region is controlled: parent owns `start` / `end` in seconds.
 *
 * Props:
 *   file:            File | null    — source audio
 *   duration:        number          — clip length in seconds (must be passed; we
 *                                       don't infer it from decoded length so the
 *                                       caller can drive a probe before decode
 *                                       finishes)
 *   start, end:      number          — region in seconds
 *   onRegionChange:  (start, end) => void
 *   minRegionSec:    number          — default 0.1
 *   height:          number          — canvas height in px (default 96)
 *   color:           CSS color       — waveform peak color (default theme accent)
 *   regionColor:     CSS color       — fill for the region rect
 */
export default function AudioWaveform({
    file,
    duration,
    start,
    end,
    onRegionChange,
    minRegionSec = 0.1,
    height = 96,
    color = '#279FBB',
    regionColor = 'rgba(253, 162, 43, 0.28)',
}) {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [width, setWidth] = useState(0);
    const [peaks, setPeaks] = useState(null);
    const [decoding, setDecoding] = useState(false);
    const [decodeError, setDecodeError] = useState(null);
    // Drag state lives in a ref to avoid re-renders during pointer move.
    const dragRef = useRef(null);

    // --- responsive width via ResizeObserver -----------------------------
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const ro = new ResizeObserver((entries) => {
            const w = Math.max(1, Math.floor(entries[0].contentRect.width));
            setWidth(w);
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    // --- decode + peak computation ---------------------------------------
    useEffect(() => {
        if (!file || !width) return;
        let cancelled = false;
        setDecoding(true);
        setDecodeError(null);

        (async () => {
            try {
                const buf = await file.arrayBuffer();
                if (cancelled) return;
                // Reuse one AudioContext where possible. Safari and Chrome both
                // permit creating an offline one for pure decode without user
                // gesture, which is what we want.
                const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
                const tmpCtx = Ctx
                    ? new Ctx(1, 44100, 44100)
                    : new (window.AudioContext || window.webkitAudioContext)();
                const audio = await tmpCtx.decodeAudioData(buf.slice(0));
                if (cancelled) return;

                // Average across channels into mono peaks, then bucket into
                // `width` columns. Each column gets (min, max) in [-1, 1].
                const ch0 = audio.getChannelData(0);
                const ch1 = audio.numberOfChannels > 1 ? audio.getChannelData(1) : null;
                const totalSamples = ch0.length;
                const bucketSize = Math.max(1, Math.floor(totalSamples / width));
                const out = new Float32Array(width * 2);
                for (let i = 0; i < width; i++) {
                    const s = i * bucketSize;
                    const e = Math.min(totalSamples, s + bucketSize);
                    let mn = 0, mx = 0;
                    for (let j = s; j < e; j++) {
                        const v = ch1 ? (ch0[j] + ch1[j]) * 0.5 : ch0[j];
                        if (v < mn) mn = v;
                        if (v > mx) mx = v;
                    }
                    out[i * 2] = mn;
                    out[i * 2 + 1] = mx;
                }
                setPeaks(out);
            } catch (err) {
                setDecodeError(err.message || 'Failed to decode audio');
            } finally {
                if (!cancelled) setDecoding(false);
            }
        })();

        return () => { cancelled = true; };
    }, [file, width]);

    // --- canvas drawing --------------------------------------------------
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas || !width || !height) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, height);

        // Background: faint center line so empty audio still shows scale.
        ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.fillRect(0, height / 2 - 0.5, width, 1);

        // Peaks
        if (peaks) {
            ctx.fillStyle = color;
            const mid = height / 2;
            const scale = (height - 4) / 2;
            for (let i = 0; i < width; i++) {
                const mn = peaks[i * 2];
                const mx = peaks[i * 2 + 1];
                const y0 = mid - mx * scale;
                const y1 = mid - mn * scale;
                ctx.fillRect(i, y0, 1, Math.max(1, y1 - y0));
            }
        }

        // Region overlay
        if (duration > 0 && Number.isFinite(start) && Number.isFinite(end)) {
            const sPx = Math.max(0, Math.min(width, (start / duration) * width));
            const ePx = Math.max(0, Math.min(width, (end / duration) * width));
            const rectW = Math.max(1, ePx - sPx);
            ctx.fillStyle = regionColor;
            ctx.fillRect(sPx, 0, rectW, height);
            // Handles
            ctx.fillStyle = '#FDA22B';
            ctx.fillRect(sPx - 1, 0, 2, height);
            ctx.fillRect(ePx - 1, 0, 2, height);
        }
    }, [width, height, peaks, color, regionColor, start, end, duration]);

    useEffect(() => { draw(); }, [draw]);

    // --- pointer interaction --------------------------------------------
    const HIT_PX = 8;
    const pxToSec = useCallback((px) => {
        return Math.max(0, Math.min(duration, (px / width) * duration));
    }, [width, duration]);

    const onPointerDown = (e) => {
        if (!duration || !width) return;
        const rect = canvasRef.current.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const sPx = (start / duration) * width;
        const ePx = (end / duration) * width;
        let mode;
        if (Math.abs(px - sPx) <= HIT_PX) mode = 'start';
        else if (Math.abs(px - ePx) <= HIT_PX) mode = 'end';
        else if (px > sPx && px < ePx) mode = 'body';
        else mode = 'new'; // start a new region by drag
        dragRef.current = {
            mode,
            startPx: px,
            origStart: start,
            origEnd: end,
        };
        canvasRef.current.setPointerCapture(e.pointerId);
        if (mode === 'new') {
            const t = pxToSec(px);
            onRegionChange?.(t, Math.min(duration, t + minRegionSec));
            dragRef.current.mode = 'end';
            dragRef.current.origStart = t;
            dragRef.current.origEnd = t + minRegionSec;
        }
    };

    const onPointerMove = (e) => {
        const d = dragRef.current;
        if (!d) return;
        const rect = canvasRef.current.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const delta = pxToSec(px) - pxToSec(d.startPx);
        let s = d.origStart;
        let en = d.origEnd;
        if (d.mode === 'start') {
            s = Math.max(0, Math.min(d.origEnd - minRegionSec, d.origStart + delta));
        } else if (d.mode === 'end') {
            en = Math.max(d.origStart + minRegionSec, Math.min(duration, d.origEnd + delta));
        } else if (d.mode === 'body') {
            const span = d.origEnd - d.origStart;
            s = Math.max(0, Math.min(duration - span, d.origStart + delta));
            en = s + span;
        }
        onRegionChange?.(s, en);
    };

    const onPointerUp = (e) => {
        if (dragRef.current) {
            canvasRef.current.releasePointerCapture(e.pointerId);
            dragRef.current = null;
        }
    };

    // --- render ----------------------------------------------------------
    return (
        <Box ref={containerRef} sx={{ width: '100%', position: 'relative' }}>
            <canvas
                ref={canvasRef}
                style={{
                    width: '100%',
                    height,
                    display: 'block',
                    cursor: dragRef.current ? 'grabbing' : 'crosshair',
                    touchAction: 'none',
                    borderRadius: 4,
                    background: 'rgba(255,255,255,0.02)',
                }}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
            />
            {(decoding || decodeError || !file) && (
                <Box
                    sx={{
                        position: 'absolute',
                        inset: 0,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        pointerEvents: 'none',
                    }}
                >
                    <Typography variant="caption" color="text.secondary">
                        {decodeError
                            ? `decode failed: ${decodeError}`
                            : !file
                                ? 'no source loaded'
                                : 'decoding…'}
                    </Typography>
                </Box>
            )}
        </Box>
    );
}
