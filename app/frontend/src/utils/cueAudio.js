// Cue / audition output. A second AudioContext that can be routed to a
// different output device than the main mix, so the performer can preview
// candidates in headphones while the audience hears the live channels.
//
// Requires AudioContext.setSinkId (Chromium ≥ 110). On unsupported browsers
// isCueSupported() returns false and callers should disable the UI.

let ctx = null;
let currentSource = null;
let currentEndedHandler = null;
let currentSinkId = '';

export function isCueSupported() {
    try {
        const AC = window.AudioContext || window.webkitAudioContext;
        return AC && typeof AC.prototype.setSinkId === 'function';
    } catch {
        return false;
    }
}

function getContext() {
    if (!ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        ctx = new AC();
    }
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
}

// Re-route the cue context to a different output device. Pass '' (or 'default')
// to revert to the system default output. Resolves to the actually-applied
// sinkId so callers can confirm.
export async function setCueDevice(deviceId) {
    if (!isCueSupported()) {
        currentSinkId = '';
        return '';
    }
    const c = getContext();
    const id = deviceId || '';
    try {
        await c.setSinkId(id);
        currentSinkId = id;
    } catch (err) {
        console.warn('[cueAudio] setSinkId failed', err);
    }
    return currentSinkId;
}

export function getCueDevice() {
    return currentSinkId;
}

// Enumerate output devices. Requires a one-time getUserMedia call to unlock
// device labels (otherwise the label is an empty string); we request a
// short-lived mic stream and immediately stop it. Returns an array of
// MediaDeviceInfo for kind === 'audiooutput'.
export async function listOutputDevices() {
    if (!navigator?.mediaDevices?.enumerateDevices) return [];
    // First call without permission gets devices with blank labels — try to
    // get permission so subsequent calls return meaningful labels.
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(t => t.stop());
    } catch {
        // Mic denied is fine; we'll still get devices, just with empty labels.
    }
    const all = await navigator.mediaDevices.enumerateDevices();
    return all.filter(d => d.kind === 'audiooutput');
}

// Play a Blob through the cue context. Returns an async-cancellable handle
// with a stop() method. Any previously-playing cue stops first.
export async function playBlob(blob, { onEnded } = {}) {
    stopCue();
    const c = getContext();
    const arr = await blob.arrayBuffer();
    const buf = await c.decodeAudioData(arr);

    const src = c.createBufferSource();
    src.buffer = buf;
    src.connect(c.destination);

    const handler = () => {
        if (currentSource === src) {
            currentSource = null;
            currentEndedHandler = null;
        }
        onEnded?.();
    };
    src.addEventListener('ended', handler);
    currentSource = src;
    currentEndedHandler = handler;
    src.start();

    return {
        stop: () => {
            if (currentSource === src) stopCue();
        },
    };
}

export function stopCue() {
    if (currentSource) {
        if (currentEndedHandler) {
            currentSource.removeEventListener('ended', currentEndedHandler);
        }
        try { currentSource.stop(); } catch { /* already stopped */ }
        try { currentSource.disconnect(); } catch { /* already disconnected */ }
        currentSource = null;
        currentEndedHandler = null;
    }
}
