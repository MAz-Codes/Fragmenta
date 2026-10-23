import React, { useEffect, useLayoutEffect, useRef, useState, useCallback } from 'react';
import { Box } from '@mui/material';

const DEFAULT_COLOR = '#279FBB';
// Fixed, low waveform resolution — matches the dataset-page waveforms
// (/peaks?n=80). Decoding to a constant bucket count (instead of one pair per
// pixel) means the decode runs once per clip rather than re-running on every
// resize, so fragments render faster.
const PEAK_COUNT = 80;

/**
 * Compact waveform indicator for a single generated fragment.
 *
 * Decodes `blob` once per width and renders min/max peaks on a canvas.
 * Played portion is rendered in `color`; unplayed in a dim version of it,
 * with a thin playhead line at the current position. Click or drag on it to
 * scrub (same gesture as the dataset-page waveforms); the row's prompt text
 * is the drag handle, not this.
 *
 * Props:
 *   blob:        Blob | null     — audio source (preferred; no fetch).
 *   audioUrl:    string          — URL for the same audio, fetched when
 *                                   there's no blob.
 *   currentTime: number          — playback head position in seconds.
 *   duration:    number          — total length in seconds.
 *   height:      number          — canvas height in px (default 28).
 *   color:       string          — accent color (default theme amber).
 *   onSeek:      (fraction) => void — called with 0..1 on click / drag.
 */
export default function GenerationWaveform({
    blob,
    audioUrl,
    currentTime = 0,
    duration = 0,
    height = 28,
    color = DEFAULT_COLOR,
    onSeek,
}) {
    const containerRef = useRef(null);
    const canvasRef = useRef(null);
    const scrubbingRef = useRef(false);
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

    // Decode into PEAK_COUNT mono min/max pairs — a fixed low resolution,
    // independent of pixel width, so the (expensive) decode runs once per clip
    // and not again on every resize.
    //
    // Audio source can be either a Blob (in-memory, fresh generations) or
    // an HTTP audioUrl (fragments hydrated from disk on app load have
    // audioBlob=null and audioUrl=/api/fragments/...). The blob path is
    // preferred when available; otherwise fetch the URL.
    useEffect(() => {
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
                const bucketSize = Math.max(1, Math.floor(totalSamples / PEAK_COUNT));
                const out = new Float32Array(PEAK_COUNT * 2);
                for (let i = 0; i < PEAK_COUNT; i++) {
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
    }, [blob, audioUrl]);

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
        // Stretch the fixed PEAK_COUNT buckets across the canvas width as bars
        // (with a 1px gap), matching the dataset-page waveform look.
        const n = peaks.length / 2;
        const step = width / n;
        const barW = Math.max(1, step - 1);
        const progressFrac = duration > 0
            ? Math.max(0, Math.min(1, currentTime / duration))
            : 0;
        const progressPx = progressFrac * width;
        const splitIdx = Math.floor(progressFrac * n);

        for (let i = 0; i < n; i++) {
            const mn = peaks[i * 2];
            const mx = peaks[i * 2 + 1];
            const y0 = mid - mx * scale;
            const y1 = mid - mn * scale;
            // Played bars: full color; unplayed: dimmed (35% alpha of accent).
            ctx.fillStyle = i < splitIdx ? color : `${color}59`;
            ctx.fillRect(i * step, y0, barW, Math.max(1, y1 - y0));
        }
        // Thin playhead at the split.
        if (progressPx > 0 && progressPx < width) {
            ctx.fillStyle = color;
            ctx.fillRect(progressPx - 0.5, 0, 1, height);
        }
    }, [width, height, peaks, color, currentTime, duration]);

    useEffect(() => { draw(); }, [draw]);

    // Translate a pointer position into a 0..1 offset in the clip.
    const seekTo = (clientX) => {
        const el = containerRef.current;
        if (!el || !onSeek) return;
        const rect = el.getBoundingClientRect();
        if (!rect.width) return;
        onSeek(Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)));
    };

    const endScrub = (e) => {
        scrubbingRef.current = false;
        // Capture is already gone if the pointer was cancelled by the browser.
        try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* noop */ }
    };

    return (
        <Box
            ref={containerRef}
            title={onSeek ? 'Click or drag to scrub' : undefined}
            onPointerDown={(e) => {
                if (!onSeek || e.button !== 0) return;
                // Stops text selection from kicking in mid-drag.
                e.preventDefault();
                // Pointer capture keeps the scrub alive past the canvas edges.
                try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* noop */ }
                scrubbingRef.current = true;
                seekTo(e.clientX);
            }}
            onPointerMove={(e) => { if (scrubbingRef.current) seekTo(e.clientX); }}
            onPointerUp={endScrub}
            onPointerCancel={endScrub}
            sx={{
                // Floor the width so the container is never zero — without
                // this, a tight flex row could collapse it before
                // ResizeObserver fires, leaving the canvas un-sized.
                flex: 1,
                minWidth: 120,
                height,
                cursor: onSeek ? 'pointer' : 'default',
                touchAction: 'none',
            }}
        >
            <canvas
                ref={canvasRef}
                style={{ display: 'block', width: '100%', height }}
            />
        </Box>
    );
}
