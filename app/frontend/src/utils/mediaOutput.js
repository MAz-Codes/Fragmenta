// Sends every page's <audio> player to the output device picked in Settings.
// The Performance engine and cue player route themselves; plain media elements
// otherwise stay on the system default, so the two could play on different
// devices. No-op where elements can't choose a device.

let sinkId = '';

function apply(el) {
    if (typeof el.setSinkId !== 'function' || el.sinkId === sinkId) return;
    el.setSinkId(sinkId).catch((err) => {
        console.warn('[mediaOutput] setSinkId failed', err);
    });
}

function onMediaEvent(e) {
    if (e.target instanceof HTMLAudioElement) apply(e.target);
}

// These don't bubble, so listen on capture. loadstart comes before any sound;
// play covers an element whose device changed while it sat idle.
if (typeof document !== 'undefined') {
    document.addEventListener('loadstart', onMediaEvent, true);
    document.addEventListener('play', onMediaEvent, true);
}

export function setMediaOutputDevice(deviceId) {
    sinkId = deviceId || '';
    document.querySelectorAll('audio').forEach(apply);
}
