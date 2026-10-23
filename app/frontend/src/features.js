/**
 * Build-time feature flags.
 *
 * These are not user settings and not runtime config — they're switches for
 * work that is finished on our side but blocked on something external. Each
 * one carries the condition for flipping it back on.
 */

/**
 * Generation page → "Edit existing" (style transfer, inpaint, extend), plus
 * the Generated Fragments drag row that feeds its drop zone. Hidden
 * 2026-08-06 while the vendor's audio-to-audio path looked unusable; back on
 * 2026-10-23.
 */
export const GENERATION_AUDIO_EDIT = true;

/**
 * ── NOTE TO SELF ─────────────────────────────────────────────────────────
 * The Performance channel's Variation control (the shuffle icon) re-rolls
 * the current fragment through the same vendor audio-to-audio path as the
 * Generation page's Edit existing mode. It stays hidden on the Performance
 * page for now.
 *
 * ROLL BACK BY: setting this to `true`. Nothing else needs changing.
 *
 * Verify by hand first: each channel's Variation button re-rolls from the
 * current fragment instead of returning garbage.
 *
 * Hidden 2026-08-06.
 * ─────────────────────────────────────────────────────────────────────────
 */
export const PERFORMANCE_VARIATION = false;
