import React, { useEffect, useLayoutEffect, useRef, useState, useCallback } from 'react';
import { Box } from '@mui/material';

const DEFAULT_COLOR = '#279FBB';

/**
 * Compact waveform indicator for a single generated fragment.
 *
 * Decodes `blob` once per width and renders min/max peaks on a canvas.
 * Played portion is rendered in `color`; unplayed in a dim version of it,
 * with a thin playhead line at the current position. The whole element is
 * draggable: dragstart sets a DownloadURL on the dataTransfer so the user
 * can drag the fragment onto their desktop or into a DAW as a .wav file.
 *
 * Props:
 *   blob:        Blob | null     — audio source (Blob is required for the
 *                                   native drag-to-OS file write).
 *   audioUrl:    string          — blob: URL for the same audio. Used in
 *                                   the dataTransfer; we fall back to
 *                                   createObjectURL(blob) if it's missing.
 *   filename:    string          — file name the OS sees when the drag
 *                                   resolves.
 *   currentTime: number          — playback head position in seconds.
 *   duration:    number          — total length in seconds.
 *   height:      number          — canvas height in px (default 28).
 *   color:       string          — accent color (default theme amber).
 */
export default function GenerationWaveform({
    blob,
    audioUrl,
    filename = 'fragment.wav',
    currentTime = 0,
    duration = 0,
    height = 28,
    color = DEFAULT_COLOR,
}) {
    const containerRef = useRef(null);
    const canvasRef = useRef(null);
    // Start at a sensible non-zero width so the decode useEffect (gated on
    // width > 0) runs on first mount instead of waiting for the async
    // ResizeObserver callback — which is what was leaving the canvas blank.
    const [width, setWidth] = useState(200);
    const [peaks, setPeaks] = useState(null);

    // Measure synchronously on mount via useLayoutEffect so we never paint
    // at the placeholder width; ResizeObserver then keeps it in sync with
    // sidebar collapses / window resizes.
    useLayoutEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        if (rect.width > 0) {
            setWidth(Math.max(1, Math.floor(rect.width)));
        }
        const ro = new ResizeObserver((entries) => {
            const w = Math.max(1, Math.floor(entries[0].contentRect.width));
            setWidth(w);
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    // Decode into mono peaks bucketed to one pair per output pixel.
    //
    // Audio source can be either a Blob (in-memory, fresh generations) or
    // an HTTP audioUrl (fragments hydrated from disk on app load have
    // audioBlob=null and audioUrl=/api/fragments/...). The blob path is
    // preferred when available; otherwise fetch the URL.
    //
    // Runs once per (blob, audioUrl, width) — re-renders during playback
    // don't touch this effect because currentTime isn't in the deps.
    useEffect(() => {
        if (!width) return;
        if (!blob && !audioUrl) return;
        let cancelled = false;
        (async () => {
            try {
                let buf;
                if (blob) {
                    buf = await blob.arrayBuffer();
                } else {
                    const r = await fetch(audioUrl);
                    if (!r.ok) {
                        console.warn(`GenerationWaveform fetch failed (${r.status}): ${audioUrl}`);
                        return;
                    }
                    buf = await r.arrayBuffer();
                }
                if (cancelled) return;
                if (!buf || buf.byteLength === 0) {
                    console.warn('GenerationWaveform: empty audio source');
                    return;
                }
                const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
                const tmpCtx = Ctx
                    ? new Ctx(1, 44100, 44100)
                    : new (window.AudioContext || window.webkitAudioContext)();
                const audio = await tmpCtx.decodeAudioData(buf.slice(0));
                if (cancelled) return;
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
                if (!cancelled) setPeaks(out);
            } catch (err) {
                console.warn('GenerationWaveform decode failed:', err);
            }
        })();
        return () => { cancelled = true; };
    }, [blob, audioUrl, width]);

    // Draw — re-runs on every currentTime tick so the playhead moves.
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas || !width || !height) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, height);

        // Always draw a faint center line so the row has a visible "this is
        // a waveform area" cue even while decode is in flight or has failed.
        ctx.fillStyle = `${color}33`;
        ctx.fillRect(0, height / 2 - 0.5, width, 1);

        if (!peaks) return;

        const mid = height / 2;
        const scale = (height - 2) / 2;
        const progressPx = duration > 0
            ? Math.max(0, Math.min(width, (currentTime / duration) * width))
            : 0;

        // Played portion: full color.
        ctx.fillStyle = color;
        const splitPx = Math.floor(progressPx);
        for (let i = 0; i < splitPx; i++) {
            const mn = peaks[i * 2];
            const mx = peaks[i * 2 + 1];
            const y0 = mid - mx * scale;
            const y1 = mid - mn * scale;
            ctx.fillRect(i, y0, 1, Math.max(1, y1 - y0));
        }
        // Unplayed portion: dimmed (35% alpha of accent).
        ctx.fillStyle = `${color}59`;
        for (let i = splitPx; i < width; i++) {
            const mn = peaks[i * 2];
            const mx = peaks[i * 2 + 1];
            const y0 = mid - mx * scale;
            const y1 = mid - mn * scale;
            ctx.fillRect(i, y0, 1, Math.max(1, y1 - y0));
        }
        // Thin playhead at the split.
        if (progressPx > 0 && progressPx < width) {
            ctx.fillStyle = color;
            ctx.fillRect(progressPx - 0.5, 0, 1, height);
        }
    }, [width, height, peaks, color, currentTime, duration]);

    useEffect(() => { draw(); }, [draw]);

    // Native drag-to-OS as a file. The DownloadURL mime type is a Chromium
    // extension the OS interprets as "this drag is a file the browser can
    // serve from URL X with mime/name Y". Source is whichever URL we have:
    // a blob: URL for in-memory fragments, or the backend /api/fragments/
    // path for disk-hydrated ones. The OS needs an ABSOLUTE URL, so we
    // resolve relative paths against window.location.origin.
    const canDrag = !!(audioUrl || blob);
    const handleDragStart = (e) => {
        if (!canDrag) return;
        const raw = audioUrl || URL.createObjectURL(blob);
        const absolute = (raw.startsWith('http') || raw.startsWith('blob:'))
            ? raw
            : `${window.location.origin}${raw.startsWith('/') ? '' : '/'}${raw}`;
        e.dataTransfer.setData('DownloadURL', `audio/wav:${filename}:${absolute}`);
        e.dataTransfer.effectAllowed = 'copy';
    };

    return (
        <Box
            ref={containerRef}
            draggable={canDrag}
            onDragStart={handleDragStart}
            title={canDrag ? 'Drag to save or drop into a DAW' : undefined}
            sx={{
                // Floor the width so the container is never zero — without
                // this, a tight flex row could collapse it before
                // ResizeObserver fires, leaving the canvas un-sized.
                flex: 1,
                minWidth: 120,
                height,
                cursor: canDrag ? 'grab' : 'default',
                '&:active': { cursor: canDrag ? 'grabbing' : 'default' },
            }}
        >
            <canvas
                ref={canvasRef}
                style={{ display: 'block', width: '100%', height }}
            />
        </Box>
    );
}
