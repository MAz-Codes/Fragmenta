// Shipped as its own untouched file and imported from there: the library
// builds its AudioWorklet by turning its own functions back into source text,
// and bundling + minifying it breaks that (renders hang, no error).
import stretchUrl from 'signalsmith-stretch?url';

let stretchLoad = null;
function loadStretch() {
    stretchLoad = stretchLoad || import(/* @vite-ignore */ stretchUrl).then((m) => m.default);
    return stretchLoad;
}

// Wrap-around context on each side of the clip. Longer than Stretch's
// analysis block (~120 ms) so the cut-off padding carries all of its
// edge effects.
const PAD_SECONDS = 0.5;
// Seam crossfade. The stretcher doesn't keep a tone's phase exactly periodic
// over the clip, so its end no longer meets its start; fading the last 20 ms
// into the audio that leads into the start makes the wrap continuous.
const SEAM_SECONDS = 0.02;

/**
 * A copy of `buffer` shifted by `semitones`, same length and timing, rendered
 * offline with Signalsmith Stretch (MIT). Nothing touches the file on disk —
 * the Performance channel plays this copy instead of the original.
 *
 * The input is wrapped with its own tail in front and its head behind, so the
 * stretcher sees a loop's real neighbours at both ends: no soft attack on the
 * first beat and no smear at the wrap. The padding is cut off again, and the
 * end is crossfaded into what precedes the start so the loop point is seamless.
 */
export async function renderPitchedBuffer(buffer, semitones) {
    const sr = buffer.sampleRate;
    const len = buffer.length;
    const nch = buffer.numberOfChannels;
    const pad = Math.min(len, Math.round(PAD_SECONDS * sr));
    const total = len + 2 * pad;

    const channels = [];
    for (let c = 0; c < nch; c++) {
        const src = buffer.getChannelData(c);
        const x = new Float32Array(total);
        x.set(src.subarray(len - pad), 0);
        x.set(src, pad);
        x.set(src.subarray(0, pad), pad + len);
        channels.push(x);
    }

    const SignalsmithStretch = await loadStretch();
    const ctx = new OfflineAudioContext(nch, total, sr);
    const node = await SignalsmithStretch(ctx, {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [nch],
    });
    await node.addBuffers(channels);
    // Await it: the change is a message to the worklet, and WebKit starts an
    // offline render before undelivered messages arrive (silent output).
    await node.schedule({ active: true, output: 0, input: 0, rate: 1, semitones });
    node.connect(ctx.destination);
    const rendered = await ctx.startRendering();

    const out = new AudioBuffer({ length: len, numberOfChannels: nch, sampleRate: sr });
    const fade = Math.min(Math.round(SEAM_SECONDS * sr), pad, Math.floor(len / 4));
    for (let c = 0; c < nch; c++) {
        const y = rendered.getChannelData(c);
        const body = y.slice(pad, pad + len);
        // Equal-power fade from the body's own end into y[pad - fade .. pad),
        // the rendered audio that runs straight into the first sample.
        for (let i = 0; i < fade; i++) {
            const k = (i + 1) / fade;
            const j = len - fade + i;
            body[j] = body[j] * Math.cos(k * Math.PI / 2) + y[pad - fade + i] * Math.sin(k * Math.PI / 2);
        }
        out.copyToChannel(body, c);
    }
    return out;
}
