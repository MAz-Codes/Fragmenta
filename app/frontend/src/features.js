/**
 * Build-time feature flags.
 *
 * These are not user settings and not runtime config — they're switches for
 * work that is finished on our side but blocked on something external. Each
 * one carries the condition for flipping it back on.
 */

/**
 * ── NOTE TO SELF ─────────────────────────────────────────────────────────
 * Audio-to-audio, inpaint and audio-extension are implemented end to end on
 * our side (EditPanel, the /api/generate init_audio + inpaint paths, and the
 * Performance channel's Variation control), but the vendor's Stable Audio 3
 * runtime does not currently produce usable results for any of them.
 *
 * Rather than delete working code, everything is gated on this one flag so
 * the whole surface comes back together and stays in sync in the meantime.
 *
 * ROLL BACK BY: setting this to `true`. Nothing else needs changing.
 *
 * DO THAT AS SOON AS: the vendor ships working audio-to-audio / inpaint /
 * extension in stable-audio-3. Re-check after any vendor bump under
 * vendor/stable-audio-3, then verify by hand:
 *   1. Generation page → the "Generate new / Edit existing" switch is back,
 *      and Edit existing produces sensible style-transfer / inpaint / extend
 *      output rather than noise.
 *   2. Performance page → each channel's Variation button (the shuffle icon)
 *      re-rolls from the current fragment instead of returning garbage.
 *
 * Hidden 2026-08-06.
 * ─────────────────────────────────────────────────────────────────────────
 */
export const VENDOR_AUDIO_EDIT_READY = false;
