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
let currentSourceFade = null;  // per-source gain used by stopCue() to ramp out
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

    // Claim every channel the device exposes and ask for discrete (no
    // speaker-layout up/down-mixing) routing. Done here — not only after
    // setSinkId — so multichannel works on the system-default device too,
    // including browsers without AudioContext.setSinkId.
    try { ctx.destination.channelCount = ctx.destination.maxChannelCount; } catch { /* capped */ }
    try { ctx.destination.channelInterpretation = 'discrete'; } catch { /* older builds */ }

    const channels = Math.max(2, ctx.destination.maxChannelCount || 2);
    cueSplitter = ctx.createChannelSplitter(2);
    cueMerger = ctx.createChannelMerger(channels);
    cueMerger.connect(ctx.destination);
    wireCuePair(currentCuePair);

    // A cue that was already sounding was connected to the old (now
    // disconnected) splitter — re-attach it so a device swap mid-preview
    // doesn't go silent.
    if (currentSourceFade) {
        try { currentSourceFade.disconnect(); } catch { /* ok */ }
        try { currentSourceFade.connect(cueSplitter); } catch { /* ok */ }
    }
}

function wireCuePair(pairIdx) {
    if (!cueSplitter || !cueMerger || !ctx) return;
    const N = ctx.destination.maxChannelCount || 2;
    const maxPair = Math.max(0, Math.floor(N / 2) - 1);
    const pair = Math.min(Math.max(0, pairIdx | 0), maxPair);

    try { cueSplitter.disconnect(); } catch { /* ok */ }
    cueSplitter.connect(cueMerger, 0, pair * 2);
    cueSplitter.connect(cueMerger, 1, pair * 2 + 1);
}

/** Public: pick which channel pair the cue routes to. The unclamped intent is
 *  remembered (even before the context exists) so graph rebuilds — device
 *  swap, or the device exposing more channels once running — restore the
 *  user's selection rather than whatever an early stereo clamp produced. */
export function setCueOutputPair(pairIdx) {
    currentCuePair = Math.max(0, pairIdx | 0);
    wireCuePair(currentCuePair);
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
    // The graph may have been built while the context was suspended (Chromium
    // can report maxChannelCount=2 until the context is running) — if the
    // device exposes more channels than the merger was sized for, rebuild so
    // non-default cue pairs are reachable.
    const wantInputs = Math.max(2, c.destination.maxChannelCount || 2);
    if (!cueMerger || cueMerger.numberOfInputs !== wantInputs) buildCueGraph();
    const arr = await blob.arrayBuffer();
    const buf = await c.decodeAudioData(arr);

    const src = c.createBufferSource();
    src.buffer = buf;
    // Per-source fade gain so stopCue() can ramp out instead of hard-cut
    // (a hard cut at non-zero samples is what produces the click /
    // crackle when switching fragments rapidly). The fade graph is:
    //   source → fadeGain → cueSplitter → cueMerger → destination
    const fadeGain = c.createGain();
    fadeGain.gain.value = 1;
    src.connect(fadeGain);
    fadeGain.connect(cueSplitter);

    const handler = () => {
        if (currentSource === src) {
            currentSource = null;
            currentSourceFade = null;
            currentEndedHandler = null;
        }
        onEnded?.();
    };
    src.addEventListener('ended', handler);
    currentSource = src;
    currentSourceFade = fadeGain;
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
        const src = currentSource;
        const fade = currentSourceFade;
        const endedHandler = currentEndedHandler;
        if (endedHandler) {
            src.removeEventListener('ended', endedHandler);
        }
        const now = ctx ? ctx.currentTime : 0;
        const FADE = 0.012;
        try {
            if (fade) {
                fade.gain.cancelScheduledValues(now);
                fade.gain.setValueAtTime(fade.gain.value, now);
                fade.gain.linearRampToValueAtTime(0, now + FADE);
            }
            src.stop(now + FADE + 0.005);
        } catch { /* already stopped */ }
        window.setTimeout(() => {
            try { src.disconnect(); } catch { /* ok */ }
            try { fade && fade.disconnect(); } catch { /* ok */ }
        }, Math.ceil((FADE + 0.02) * 1000));
        currentSource = null;
        currentSourceFade = null;
        currentEndedHandler = null;
        // Fire the displaced cue's onEnded so its owner can clear its UI.
        // Cue is a global single-player: when channel B's audition stops
        // channel A's (playBlob calls stopCue first), A's "playing" pill
        // otherwise sticks on forever — A never learns its cue ended.
        try { endedHandler?.(); } catch { /* listener error — ignore */ }
    }
}
