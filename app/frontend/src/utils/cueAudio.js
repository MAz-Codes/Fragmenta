// Cue / audition output. A second AudioContext that can be routed to a
// different output device than the main mix, so the performer can preview
// candidates in headphones while the audience hears the live channels.
//
// Stage 2: the cue source now flows through a ChannelSplitter → ChannelMerger
// → destination tail so the user can pick which channel pair of a multichannel
// device the cue goes to (independent from the main mix's pair).
//
// Requires AudioContext.setSinkId (Chromium ≥ 110). On unsupported browsers
// isCueSupported() returns false and callers should disable the UI.

let ctx = null;
let currentSource = null;
let currentEndedHandler = null;
let currentSinkId = '';

let cueSplitter = null;
let cueMerger = null;
let currentCuePair = 0;

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
        buildCueGraph();
    }
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
}

// (Re)build the splitter → merger → destination tail. Called once on first
// context use and again whenever setCueDevice succeeds (since maxChannelCount
// may have changed). Restores currentCuePair after rebuild.
function buildCueGraph() {
    if (!ctx) return;
    try { cueSplitter?.disconnect(); } catch { /* ok */ }
    try { cueMerger?.disconnect(); } catch { /* ok */ }

    const channels = Math.max(2, ctx.destination.maxChannelCount || 2);
    cueSplitter = ctx.createChannelSplitter(2);
    cueMerger = ctx.createChannelMerger(channels);
    cueMerger.connect(ctx.destination);
    wireCuePair(currentCuePair);
}

function wireCuePair(pairIdx) {
    if (!cueSplitter || !cueMerger || !ctx) return;
    const N = ctx.destination.maxChannelCount || 2;
    const maxPair = Math.max(0, Math.floor(N / 2) - 1);
    const pair = Math.min(Math.max(0, pairIdx | 0), maxPair);

    try { cueSplitter.disconnect(); } catch { /* ok */ }
    cueSplitter.connect(cueMerger, 0, pair * 2);
    cueSplitter.connect(cueMerger, 1, pair * 2 + 1);
    currentCuePair = pair;
}

/** Public: pick which channel pair the cue routes to. */
export function setCueOutputPair(pairIdx) {
    wireCuePair(pairIdx);
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
        if (c.state === 'suspended') await c.resume();
        await c.setSinkId(id);
        currentSinkId = id;
        // Same coerce-channels trick as the main engine — try to claim
        // all available channels post-swap. Silently clamped if not.
        try {
            c.destination.channelCount = c.destination.maxChannelCount;
        } catch { /* ok */ }
        try {
            c.destination.channelInterpretation = 'discrete';
        } catch { /* ok */ }
        // maxChannelCount may have changed — rebuild graph.
        buildCueGraph();
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
// with a stop() method. Any previously-playing cue stops first. The source
// is routed through the splitter so it hits the user-selected channel pair.
export async function playBlob(blob, { onEnded } = {}) {
    stopCue();
    const c = getContext();
    const arr = await blob.arrayBuffer();
    const buf = await c.decodeAudioData(arr);

    const src = c.createBufferSource();
    src.buffer = buf;
    // Connect into the splitter, NOT directly to destination — that's how
    // the channel-pair routing applies.
    src.connect(cueSplitter);

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
