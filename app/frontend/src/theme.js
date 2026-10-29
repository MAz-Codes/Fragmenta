import { createTheme, responsiveFontSizes } from '@mui/material/styles';

/* =====================================================================
 * Fragmenta 1.0.0 theme — Arcade (dark) × Paper (light)
 *
 *   Dark mode is modelled on Output Arcade's aesthetic: warm-tinted near
 *   black, amber/gold primary accent with a thin-outlined "pad" language,
 *   subtle frosted-glass on floating surfaces, cool blue secondary for
 *   selected items, gentle warm shadows. Studio gear, not admin panel.
 *
 *   Light mode is the warm-cream "paper" the user picked — generous
 *   off-white, deep gold accent (same family as dark mode), warm brown
 *   text. Reads as analog notebook.
 *
 *   Typography: Inter (variable, from Google Fonts) replaces Helvetica
 *   Neue. Tighter tracking, smaller body, denser hierarchy.
 *
 *   appStyles and the per-component style maps below stay as-is; they're
 *   layout-level concerns the page-by-page pass will touch.
 * =================================================================== */

const FONT_BODY  = '"Inter Tight", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif';
const FONT_MONO  = '"JetBrains Mono", "IBM Plex Mono", ui-monospace, Menlo, monospace';
// Display face used for Tier-1 section titles + Tier-2 accordion labels —
// gives the cards a strong, distinctive header voice that doesn't compete
// with body Inter.
const FONT_DISPLAY = '"Bricolage Grotesque", "Inter Tight", system-ui, sans-serif';

// --- Arcade (dark) palette --------------------------------------------------
// Neutral charcoal base — the amber accent does the warmth.
const DARK = {
    bg:        '#1F2021',      // neutral near-black, slight blue-gray bias
    bgElev:    '#26282A',      // one notch up (containers)
    paper:     '#2B2D2F',      // panels / cards
    paperHi:   '#34373A',      // hovered panel / menu items
    divider:   'rgba(255, 255, 255, 0.08)',  // neutral, very subtle
    text:      '#ECECEC',      // near-white, faintly cool
    textDim:   '#9B9B9D',      // neutral gray
    textFaint: '#65676A',
    amber:     '#279FBB',      // primary accent — saturated cyan
    amberHi:   '#4DBAD3',      // hover / lighter
    amberLo:   '#1F7E94',      // pressed / darker
    // Warm complement — used to signal "active / in progress" state
    // (training, generation, download) against the cool cyan UI chrome.
    warm:      '#FDA22B',      // golden amber-orange
    warmHi:    '#FFB855',
    warmLo:    '#D17F1A',
    // Deep blue used for moon-icon / night cues on dark mode.
    night:     '#3D6FA8',
    blue:      '#5BA9E8',      // secondary — selected-file cue
    blueDim:   '#84BFEE',
    success:   '#7AC795',
    error:     '#E26B5E',      // soft red
    warning:   '#E3A34B',
};

// --- Palette -----------------------------------------------------------------
// One palette, Linen (no dark mode; redesign/DESIGN_LANGUAGE.md). A cream
// background, an ink for text and icons, and four pastel accents in roles:
//   cool    → the main accent (primary: buttons, sliders, links, focus)
//   warm    → "active / in progress" (generation, training, Link/MIDI on)
//   fresh   → success and the secondary cue
//   hot     → errors, Record, the cue point
// Pastels fill large surfaces with ink text on them (5–6:1). Where an accent
// must read as text or a thin line on the cream, a deepened shade of the same
// hue is used instead (`deepen`, below).
export const PALETTES = {
    linen: {
        name: 'Linen', bg: '#F2EDE1', ink: '#141414',
        // Saturated mains for fills and marks; each has a faint partner for
        // waveform layers, tints and soft states.
        cool: '#A0B2D5', warm: '#E0B56F', fresh: '#B3BE8C', hot: '#C18992',   // blue, gold, green, red
        faint: { cool: '#A7AEE3', warm: '#E6BF7E', fresh: '#92BBA3', hot: '#F9BDAF' },
        exact: {
            surface: '#F8F7F1', control: '#EEE9E0', controlHover: '#E5E0D6', section: '#F6F2EA',
            textDim: '#4A4843', textFaint: '#66625B', textMuted: '#908C84', navSelected: '#D5D5E4',
            amber: '#1A49DE', amberHi: '#3C64E3', amberLo: '#153CB6',
            success: '#007B51', error: '#C81E0F', warning: '#9A6000',
        },
        lanes: {
            tints: ['#D5D5E4', '#F6DBCF', '#CDDACB', '#EEDCBC'],
            playing: ['#C7C9E4', '#F7D2C6', '#BBD1BF', '#ECD3A9'],
            deep: ['#A0B2D5', '#C18992', '#B3BE8C', '#E0B56F'],
            soft: ['#A7AEE3', '#F9BDAF', '#92BBA3', '#E6BF7E'],
        },
    },
};
export const DEFAULT_PALETTE = 'linen';

// Waveform colours for lists (dataset clips, generated fragments): the four
// accents in turn, green → blue → red → gold, so a long list isn't one hue.
const WAVE_ORDER = ['fresh', 'cool', 'hot', 'warm'];
export const waveColor = (theme, index = 0) => {
    const key = WAVE_ORDER[((index % 4) + 4) % 4];
    const f = theme.palette.fragmenta;
    return { main: f.accents[key], faint: f.faint[key] };
};

// Label colour for text/icons on a solid fill: white where it reads (the
// saturated mains), ink otherwise (gold, faint tones, surfaces).
export const onFill = (fill, ink = '#141414') => {
    if (typeof fill !== 'string' || !/^#[0-9a-f]{6}$/i.test(fill)) return ink;
    return contrastRatio('#FFFFFF', fill) >= 3.5 ? '#FFFFFF' : ink;
};

// Shape hierarchy, not one radius everywhere: large surfaces softly rounded,
// controls compact, 999 only for selected states and circular precision
// controls. px strings so they don't multiply by theme.shape.borderRadius.
// Controls sitting on a lane's tint.
export const ON_TINT = 'rgba(248, 247, 241, 0.72)';
export const ON_TINT_HOVER = 'rgba(248, 247, 241, 0.92)';
export const RADIUS = { card: '24px', section: '18px', panel: '18px', input: '14px', control: '12px', small: '8px', pill: '999px' };

const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const rgbHex = (c) => '#' + c.map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('').toUpperCase();
const mixHex = (a, b, t) => { const A = hexRgb(a), B = hexRgb(b); return rgbHex(A.map((v, i) => v + (B[i] - v) * t)); };
const rgbStr = (h) => hexRgb(h).join(', ');
const luminance = (h) => {
    const [r, g, b] = hexRgb(h).map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
function contrastRatio(a, b) { return contrast(a, b); }
const contrast = (a, b) => { const [x, y] = [luminance(a), luminance(b)].sort((m, n) => n - m); return (x + 0.05) / (y + 0.05); };
const hslToHex = (h, s, l) => {
    const k = (n) => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
    return rgbHex([0, 8, 4].map((n) => 255 * (l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1)))));
};
const hexHsl = (hex) => {
    const [r, g, b] = hexRgb(hex).map((v) => v / 255);
    const max = Math.max(r, g, b), min = Math.min(r, g, b), l = (max + min) / 2, d = max - min;
    if (!d) return [0, 0, l];
    const s = d / (1 - Math.abs(2 * l - 1));
    const h = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
    return [(h * 60 + 360) % 360, s, l];
};
// Same hue and saturation, darker until it reaches `target` contrast on `bg`.
const deepen = (hex, bg, target) => {
    const [h, s, l] = hexHsl(hex);
    for (let L = l; L > 0; L -= 0.002) {
        const c = hslToHex(h, s, L);
        if (contrast(c, bg) >= target) return c;
    }
    return '#000000';
};
// The ink mixed toward the background, as light as `target` contrast allows.
const inkAt = (ink, bg, target) => {
    for (let t = 1; t >= 0; t -= 0.01) {
        const c = mixHex(ink, bg, t);
        if (contrast(c, bg) >= target) return c;
    }
    return ink;
};

function paletteTokens(P) {
    const white = '#FFFFFF';
    const t = {
        name: P.name,
        bg: P.bg,
        bgElev: mixHex(P.bg, white, 0.3),
        paper: mixHex(P.bg, white, 0.45),
        paperHi: mixHex(P.bg, white, 0.6),
        bgTop: mixHex(P.bg, white, 0.4),
        bgBottom: mixHex(P.bg, P.ink, 0.06),
        divider: `rgba(${rgbStr(P.ink)}, 0.16)`,
        text: P.ink,
        textDim: inkAt(P.ink, P.bg, 7),
        textFaint: inkAt(P.ink, P.bg, 4.5),
        onAccent: mixHex(P.bg, white, 0.8),
        // Primary: deepened cool accent (text and thin lines stay legible).
        amber: deepen(P.cool, P.bg, 4.5),
        amberHi: deepen(P.cool, P.bg, 3),
        amberLo: deepen(P.cool, P.bg, 6),
        warm: deepen(P.warm, P.bg, 3),
        warmHi: P.warm,
        warmLo: deepen(P.warm, P.bg, 4.5),
        night: deepen(P.cool, P.bg, 6),
        blue: deepen(P.fresh, P.bg, 4.5),
        blueDim: deepen(P.fresh, P.bg, 3),
        blueLo: deepen(P.fresh, P.bg, 6),
        success: deepen(P.fresh, P.bg, 4.5),
        successHi: deepen(P.fresh, P.bg, 3),
        successLo: deepen(P.fresh, P.bg, 6),
        error: deepen(P.hot, P.bg, 4.5),
        errorHi: deepen(P.hot, P.bg, 3),
        errorLo: deepen(P.hot, P.bg, 6),
        warning: deepen(P.warm, P.bg, 4.5),
        accents: { cool: P.cool, warm: P.warm, fresh: P.fresh, hot: P.hot },
        faint: P.faint || { cool: P.cool, warm: P.warm, fresh: P.fresh, hot: P.hot },
        // Layered tones: ivory canvas → lifted surface → interactive control.
        surface: mixHex(P.bg, white, 0.55),
        control: mixHex(P.bg, P.ink, 0.06),
        controlHover: mixHex(P.bg, P.ink, 0.10),
        section: P.bg,
        navSelected: mixHex(P.cool, P.bg, 0.5),
    };
    Object.assign(t, P.exact || {});
    Object.assign(t, {
        inkRgb: rgbStr(P.ink),
        amberRgb: rgbStr(t.amber),
        warmRgb: rgbStr(t.warm),
        errorRgb: rgbStr(t.error),
        bgRgb: rgbStr(P.bg),
        glassRgb: rgbStr(mixHex(P.bg, white, 0.35)),
        glassHiRgb: rgbStr(mixHex(P.bg, white, 0.6)),
        hiRgb: rgbStr(t.onAccent),
    });
    // Diffuse, low-opacity shadows, only where something is elevated.
    t.shadowSoft = `0 1px 2px rgba(${t.inkRgb}, 0.04), 0 6px 20px rgba(${t.inkRgb}, 0.05)`;
    t.shadowRaised = `0 2px 6px rgba(${t.inkRgb}, 0.06), 0 14px 32px rgba(${t.inkRgb}, 0.08)`;
    // Performance tab: pastel fills with ink labels.
    const channels = [P.cool, P.hot, P.fresh, P.warm, P.cool, P.hot, P.fresh, P.warm];
    t.perf = {
        channels,
        // Lane cards sit on pale tints of their colour; arcs, waveforms and
        // other fine marks use a deeper shade so they read on that tint.
        laneTints: P.lanes ? [...P.lanes.tints, ...P.lanes.tints] : channels.map((c) => mixHex(c, P.bg, 0.8)),
        laneTintsPlaying: P.lanes ? [...P.lanes.playing, ...P.lanes.playing] : channels.map((c) => mixHex(c, P.bg, 0.66)),
        channelsDeep: P.lanes ? [...P.lanes.deep, ...P.lanes.deep] : channels.map((c) => deepen(c, mixHex(c, P.bg, 0.8), 3.2)),
        channelsSoft: P.lanes ? [...P.lanes.soft, ...P.lanes.soft] : channels.map((c) => mixHex(c, P.bg, 0.45)),
        master: P.cool,
        masterHover: mixHex(P.cool, white, 0.25),
        play: P.cool,
        stop: t.control,
        record: P.hot,
        cue: t.error,
        toggleOn: P.warm,
        toggleOnHover: mixHex(P.warm, white, 0.25),
        toggleOff: t.control,
        toggleOffHover: t.controlHover,
        label: P.ink,
        midiMappedRgb: rgbStr(P.fresh),
        midiLearn: P.warm,
        midiLearnRgb: rgbStr(P.warm),
    };
    return t;
}

// --- Raised-pad depth recipe ------------------------------------------------
// The "lively" contained buttons get their depth from a directional gradient +
// inset top highlight + layered drop shadow. These reproduce that on any filled
// element regardless of hue: a sheen overlay (light top → dark bottom) rides
// over the element's own backgroundColor, paired with a matching shadow set.
// Used by the toggle/chip overrides below and the performance-channel styles.
export const SHEEN_DARK  = 'linear-gradient(180deg, rgba(255,255,255,0.24) 0%, rgba(255,255,255,0.05) 46%, rgba(0,0,0,0.18) 100%)';
export const SHEEN_LIGHT = 'linear-gradient(180deg, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.12) 46%, rgba(0,0,0,0.10) 100%)';
export const RAISE_DARK  = 'inset 0 1px 0 rgba(255,255,255,0.30), 0 1px 2px rgba(0,0,0,0.45), 0 4px 10px rgba(0,0,0,0.34)';
export const RAISE_LIGHT = 'inset 0 1px 0 rgba(255,255,255,0.65), 0 1px 2px rgba(43,31,18,0.14), 0 4px 10px rgba(43,31,18,0.12)';

// Performance-tab colored buttons (Generate / Play All / Stop All / Record):
// the "contained" aesthetic — gradient fill + drop shadow, NO border (matches
// the Suggest-hyperparameters button). PERF_BTN_SHEEN is a diagonal
// light→dark overlay layered over the solid fill color for depth; the shadows
// mirror the MuiButton contained override.
export const PERF_BTN_SHEEN = 'linear-gradient(135deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0) 45%, rgba(0,0,0,0.20) 100%)';
export const PERF_BTN_SHADOW = '0 1px 2px rgba(0,0,0,0.16), 0 2px 5px rgba(0,0,0,0.12)';
export const PERF_BTN_SHADOW_HOVER = '0 2px 4px rgba(0,0,0,0.18), 0 4px 9px rgba(0,0,0,0.15)';

// Base slider color — the saturated cyan.
// Sliders now take their color from the root `color` (slots use currentColor),
// so this is the default and any caller can recolor a slider via `color` in sx.
export const NORDIC_BLUE = '#279FBB';

let theme = createTheme({
    palette: {
        mode: 'dark',
        primary: { main: DARK.amber, light: DARK.amberHi, dark: DARK.amberLo, contrastText: '#100E0A' },
        secondary: { main: DARK.blue, light: DARK.blueDim, dark: '#3F87C3', contrastText: '#0A1320' },
        background: { default: DARK.bg, paper: DARK.paper },
        text: { primary: DARK.text, secondary: DARK.textDim, disabled: DARK.textFaint },
        divider: DARK.divider,
        error: { main: DARK.error },
        warning: { main: DARK.warning },
        success: { main: DARK.success },
        info: { main: DARK.blue },
        warm: { main: DARK.warm, light: DARK.warmHi, dark: DARK.warmLo },
        night: { main: DARK.night },
    },
    shape: {
        borderRadius: 10,
    },
    // Slower, smoother defaults than MUI's stock 195/225/300ms transitions.
    // Applies to every component that reads from theme.transitions — Dialog
    // backdrop + paper, Fade, Slide, Collapse, Snackbar, Menu, Tooltip, etc.
    // Easing matches the custom curve already used on the button press
    // animations, so all motion in the app feels like one family.
    transitions: {
        duration: {
            shortest: 120,
            shorter: 150,
            short: 180,
            standard: 220,
            complex: 300,
            enteringScreen: 220,
            leavingScreen: 180,
        },
        easing: {
            easeInOut: 'cubic-bezier(0.16, 1, 0.3, 1)',
            easeOut: 'cubic-bezier(0.16, 1, 0.3, 1)',
            easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
            sharp: 'cubic-bezier(0.4, 0, 0.6, 1)',
        },
    },
    typography: {
        fontFamily: FONT_BODY,
        h1: { fontWeight: 700, letterSpacing: '-0.02em' },
        h2: { fontWeight: 700, letterSpacing: '-0.02em' },
        h3: { fontWeight: 600, letterSpacing: '-0.015em' },
        h4: { fontWeight: 600, letterSpacing: '-0.01em' },
        h5: { fontWeight: 600, letterSpacing: '-0.005em' },
        // Tier-1 section card titles — Bricolage at display opsz for
        // the wider, more characterful display cut.
        h6: {
            fontFamily: FONT_DISPLAY,
            fontWeight: 600,
            letterSpacing: '-0.015em',
            fontSize: '1.5rem',
            fontVariationSettings: '"opsz" 96',
        },
        // Tier-2 section/accordion labels — same family, slightly smaller.
        // "Annotator Labels", "Advanced Settings", "Edit existing audio".
        subtitle1: {
            fontFamily: FONT_DISPLAY,
            fontWeight: 600,
            letterSpacing: '-0.01em',
            fontSize: '1rem',
            textTransform: 'none',
            fontVariationSettings: '"opsz" 72',
        },
        subtitle2: { fontFamily: FONT_DISPLAY, fontWeight: 500, letterSpacing: 0,         fontSize: '0.825rem', textTransform: 'uppercase' },
        body1: { fontWeight: 400, letterSpacing: '-0.005em',    fontSize: '0.925rem' },
        body2: { fontWeight: 400, letterSpacing: '-0.005em',    fontSize: '0.825rem' },
        body3: { fontWeight: 400, letterSpacing: '-0.005em',    fontSize: '0.7rem' },
        button: { fontFamily: FONT_DISPLAY, fontWeight: 500, letterSpacing: '0.01em',     textTransform: 'none', fontSize: '0.8rem' },
        caption: { fontFamily: FONT_DISPLAY, fontWeight: 400, letterSpacing: '0.005em',   fontSize: '0.75rem' },
        overline: { fontFamily: FONT_DISPLAY, fontWeight: 600, letterSpacing: '0.12em',   textTransform: 'uppercase', fontSize: '0.7rem' },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                ':root': { colorScheme: 'dark' },
                // ---- Motion keyframes (Phase 2) ----
                '@keyframes fragmenta-fade-up': {
                    from: { opacity: 0, transform: 'translateY(12px)' },
                    to:   { opacity: 1, transform: 'translateY(0)' },
                },
                // Generate: a brief expansion as a lane starts generating.
                '@keyframes fragmenta-generate-pop': {
                    '0%': { transform: 'scale(1)' },
                    '40%': { transform: 'scale(1.035)' },
                    '100%': { transform: 'scale(1)' },
                },
                // A soft band sweeping across a lane's waveform while it generates.
                '@keyframes fragmenta-wave-sweep': {
                    from: { backgroundPosition: '-60% 0' },
                    to: { backgroundPosition: '160% 0' },
                },
                '@keyframes fragmenta-fade-in': {
                    from: { opacity: 0 },
                    to:   { opacity: 1 },
                },
                '@keyframes fragmenta-press': {
                    '0%':   { transform: 'scale(1)' },
                    '40%':  { transform: 'scale(0.97)' },
                    '100%': { transform: 'scale(1)' },
                },
                // Respect user preference for reduced motion.
                '@media (prefers-reduced-motion: reduce)': {
                    '*, *::before, *::after': {
                        animationDuration: '0.01ms !important',
                        animationIterationCount: '1 !important',
                        transitionDuration: '0.01ms !important',
                    },
                },
                body: {
                    margin: 0,
                    minHeight: '100vh',
                    // Three-layer backdrop: warm radial in the bottom-
                    // right corner, a softer cyan wash on the left, and
                    // the diagonal grey gradient. The cyan is gentle so
                    // the upper-right corner still reads neutral.
                    backgroundColor: DARK.bg,
                    backgroundImage:
                        `radial-gradient(900px 700px at -5% 50%, rgba(39, 159, 187, 0.14), transparent 60%), ` +
                        `radial-gradient(1100px 700px at 95% 108%, rgba(253, 162, 43, 0.11), transparent 55%), ` +
                        `linear-gradient(165deg, #181A1B 0%, ${DARK.bg} 42%, #1A1B1C 100%)`,
                    backgroundAttachment: 'fixed',
                    color: DARK.text,
                    fontFeatureSettings: '"cv11", "ss01", "ss03"',  // Inter stylistic alts
                },
                '#root': { minHeight: '100vh' },
                // Always reserve a gutter for the page scrollbar so the
                // layout doesn't shift when it appears/disappears. The
                // previous `overflow: overlay` trick stopped working in
                // newer Chromium/Electron builds (the keyword was removed),
                // which caused content to jump left when a scrollbar showed.
                html: { scrollbarGutter: 'stable' },
                'body, body *': { scrollbarGutter: 'auto' },
                '*::-webkit-scrollbar': { width: '8px', height: '8px' },
                '*::-webkit-scrollbar-track': {
                    background: 'transparent',
                    borderRadius: '999px',
                },
                '*::-webkit-scrollbar-thumb': {
                    // Neutral gray — no accent color in chrome.
                    background: 'rgba(255, 255, 255, 0.16)',
                    borderRadius: '999px',
                    border: '2px solid rgba(0, 0, 0, 0)',
                    backgroundClip: 'padding-box',
                    '&:hover': { background: 'rgba(255, 255, 255, 0.28)' },
                },
                '*::-webkit-scrollbar-corner': { background: 'transparent' },
                '*': {
                    scrollbarWidth: 'thin',
                    scrollbarColor: 'rgba(255, 255, 255, 0.16) transparent',
                },
            },
        },
        MuiTypography: {
            defaultProps: {
                variantMapping: {
                    body3: 'p',
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    // Liquid Glass: highly translucent so the body's
                    // radial cyan/warm glows bleed through visibly.
                    // `saturate` amplifies that bleed to make tint pop.
                    backgroundColor: 'rgba(38, 41, 44, 0.38)',
                    backgroundImage: 'none',
                    backdropFilter: 'blur(28px) saturate(200%)',
                    WebkitBackdropFilter: 'blur(28px) saturate(200%)',
                    border: 'none',
                    boxShadow:
                        '0 24px 48px rgba(0, 0, 0, 0.55), ' +
                        '0 4px 12px rgba(0, 0, 0, 0.35), ' +
                        'inset 0 1px 0 rgba(255, 255, 255, 0.22), ' +
                        'inset 0 -1px 0 rgba(0, 0, 0, 0.35), ' +
                        'inset 1px 0 0 rgba(255, 255, 255, 0.08), ' +
                        'inset -1px 0 0 rgba(0, 0, 0, 0.20)',
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(38, 41, 44, 0.38)',
                    backgroundImage: 'none',
                    backdropFilter: 'blur(28px) saturate(200%)',
                    WebkitBackdropFilter: 'blur(28px) saturate(200%)',
                    border: 'none',
                    boxShadow:
                        '0 24px 48px rgba(0, 0, 0, 0.55), ' +
                        '0 4px 12px rgba(0, 0, 0, 0.35), ' +
                        'inset 0 1px 0 rgba(255, 255, 255, 0.22), ' +
                        'inset 0 -1px 0 rgba(0, 0, 0, 0.35), ' +
                        'inset 1px 0 0 rgba(255, 255, 255, 0.08), ' +
                        'inset -1px 0 0 rgba(0, 0, 0, 0.20)',
                    transition: 'box-shadow 220ms ease, background-color 220ms ease',
                    '&:hover': {
                        backgroundColor: 'rgba(46, 50, 54, 0.46)',
                        boxShadow:
                            '0 32px 64px rgba(0, 0, 0, 0.6), ' +
                            '0 6px 16px rgba(0, 0, 0, 0.45), ' +
                            'inset 0 1px 0 rgba(255, 255, 255, 0.30), ' +
                            'inset 0 -1px 0 rgba(0, 0, 0, 0.40), ' +
                            '0 0 0 1px rgba(39, 159, 187, 0.30)',
                    },
                },
            },
        },
        MuiButtonBase: {
            styleOverrides: {
                root: { fontFamily: FONT_DISPLAY },
            },
        },
        MuiButton: {
            variants: [
                {
                    props: { color: 'warm', variant: 'contained' },
                    style: {
                        backgroundImage: `linear-gradient(135deg, ${DARK.warmHi} 0%, ${DARK.warm} 55%, ${DARK.warmLo} 100%)`,
                        color: '#1A0F00',
                        '&:hover': {
                            backgroundImage: `linear-gradient(135deg, ${DARK.warmHi} 0%, ${DARK.warmHi} 55%, ${DARK.warm} 100%)`,
                        },
                    },
                },
            ],
            styleOverrides: {
                root: {
                    fontFamily: FONT_DISPLAY,
                    textTransform: 'none',
                    borderRadius: 999,
                    fontWeight: 400,
                    paddingInline: 18,
                    lineHeight: 1.2,
                    letterSpacing: '0.01em',
                    transition: 'transform 220ms cubic-bezier(0.16, 1, 0.3, 1), box-shadow 220ms cubic-bezier(0.16, 1, 0.3, 1), border-color 220ms ease, background-color 220ms ease, color 220ms ease',
                    // Tactile press feedback — quick squeeze on mousedown.
                    '&:active:not(.Mui-disabled)': {
                        transform: 'scale(0.96)',
                        transition: 'transform 80ms ease-out',
                    },
                },
                contained: {
                    boxShadow: '0 3px 8px rgba(0, 0, 0, 0.45), 0 12px 24px rgba(0, 0, 0, 0.55)',
                    '&:hover': {
                        boxShadow: '0 5px 12px rgba(0, 0, 0, 0.5), 0 16px 32px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(39, 159, 187, 0.4)',
                        transform: 'translateY(-1px)',
                    },
                    // Strip the gradient + colored fill when disabled so
                    // every contained variant reads as a neutral gray
                    // chip (matches the outlined "Stop" button's disabled
                    // look).
                    '&.Mui-disabled': {
                        backgroundImage: 'none',
                        backgroundColor: 'rgba(255, 255, 255, 0.06)',
                        color: 'rgba(255, 255, 255, 0.26)',
                        boxShadow: 'none',
                    },
                },
                containedPrimary: {
                    backgroundImage: `linear-gradient(135deg, ${DARK.amberHi} 0%, ${DARK.amber} 55%, ${DARK.amberLo} 100%)`,
                    color: '#100E0A',
                },
                containedSecondary: {
                    backgroundImage: `linear-gradient(135deg, ${DARK.blueDim} 0%, ${DARK.blue} 100%)`,
                },
                containedError: {
                    backgroundImage: 'linear-gradient(135deg, #ED7B6E 0%, #C95A4F 100%)',
                },
                outlined: {
                    // Cyan accent by default — outlined still reads as
                    // the secondary action vs filled contained, but
                    // carries enough color to feel active.
                    borderColor: 'rgba(39, 159, 187, 0.50)',
                    color: DARK.amberHi,
                    '&:hover': {
                        borderColor: DARK.amber,
                        backgroundColor: 'rgba(39, 159, 187, 0.10)',
                        color: DARK.amberHi,
                    },
                    '&.Mui-disabled': {
                        borderColor: 'rgba(240, 237, 229, 0.12)',
                        color: 'rgba(255, 255, 255, 0.26)',
                    },
                },
                // The generic `outlined` rule above paints cyan; without an
                // explicit error carve-out, `color="error"` on outlined
                // buttons (Delete project, Clear annotations) gets clobbered
                // and silently reads as cyan.
                outlinedError: {
                    borderColor: 'rgba(226, 107, 94, 0.55)',
                    color: DARK.error,
                    '&:hover': {
                        borderColor: DARK.error,
                        backgroundColor: 'rgba(226, 107, 94, 0.10)',
                        color: DARK.error,
                    },
                },
                text: {
                    color: DARK.text,
                    '&:hover': {
                        backgroundColor: 'rgba(39, 159, 187, 0.08)',
                        color: DARK.amberHi,
                    },
                },
            },
        },
        MuiInputBase: {
            styleOverrides: {
                root: {
                    fontFamily: FONT_DISPLAY,
                    fontSize: '0.8rem',
                    '&:not(.MuiInputBase-multiline)': { alignItems: 'center' },
                },
                input: {
                    fontFamily: FONT_DISPLAY,
                    fontSize: '0.8rem',
                    lineHeight: 1.4,
                    '&::placeholder': { fontFamily: FONT_DISPLAY, fontSize: '0.8rem', opacity: 0.6 },
                },
            },
        },
        MuiInputLabel: {
            styleOverrides: {
                root: { fontFamily: FONT_DISPLAY, fontSize: '0.8rem' },
            },
        },
        MuiFormLabel: {
            styleOverrides: {
                root: { fontFamily: FONT_DISPLAY, fontSize: '0.8rem' },
            },
        },
        MuiFormHelperText: {
            styleOverrides: {
                root: { fontFamily: FONT_DISPLAY, fontSize: '0.7rem' },
            },
        },
        MuiTextField: {
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        backgroundColor: 'rgba(10, 8, 6, 0.5)',
                        borderRadius: 8,
                        '& fieldset': { borderColor: 'rgba(240, 237, 229, 0.14)' },
                        '&:hover fieldset': { borderColor: 'rgba(240, 237, 229, 0.32)' },
                        '&.Mui-focused fieldset': {
                            borderColor: DARK.amber,
                            boxShadow: '0 0 0 3px rgba(39, 159, 187, 0.12)',
                        },
                    },
                },
            },
        },
        MuiSelect: {
            styleOverrides: {
                root: {
                    fontFamily: FONT_DISPLAY,
                    fontSize: '0.8rem',
                    backgroundColor: 'rgba(10, 8, 6, 0.5)',
                    borderRadius: 8,
                    '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(240, 237, 229, 0.14)' },
                    '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(240, 237, 229, 0.32)' },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: DARK.amber,
                        boxShadow: '0 0 0 3px rgba(39, 159, 187, 0.12)',
                    },
                },
                select: { display: 'flex', alignItems: 'center', fontSize: '0.8rem' },
            },
        },
        // Every overlay component (Select, Menu, Autocomplete, Dialog, Drawer)
        // bottoms out at MuiModal, which by default locks body scroll AND pads
        // the body by the scrollbar width to keep layout stable. With a
        // scrollable page, that pad reads as a right-side gutter every time
        // anything overlays the page. We disable the lock globally; the
        // backdrop still catches clicks, so modals stay modal — the user just
        // doesn't get a visible content shift. Dialog/Drawer get their own
        // entries below; this catches Popover/Menu/Autocomplete which already
        // share styleOverrides blocks here.
        MuiModal: {
            defaultProps: { disableScrollLock: true },
        },
        // Dropdown / popover paper. Promotes the look the performance MIDI /
        // Audio / Save menus pioneered (crisp 1px divider border + 20px
        // rounding + clean overflow, layered over the global glass Paper) to
        // every Select dropdown, Menu, and Popover. 20px === sx borderRadius:
        // 2 at shape.borderRadius 10.
        MuiPopover: {
            defaultProps: { disableScrollLock: true },
            styleOverrides: {
                paper: {
                    border: `1px solid ${DARK.divider}`,
                    borderRadius: 20,
                    overflow: 'hidden',
                },
            },
        },
        MuiMenu: {
            defaultProps: { disableScrollLock: true },
            styleOverrides: {
                paper: {
                    border: `1px solid ${DARK.divider}`,
                    borderRadius: 20,
                    overflow: 'hidden',
                },
            },
        },
        MuiDrawer: {
            defaultProps: { disableScrollLock: true },
        },
        MuiMenuItem: {
            styleOverrides: {
                root: {
                    fontFamily: FONT_DISPLAY,
                    fontSize: '0.8rem',
                    backgroundColor: DARK.paper,
                    '&:hover': { backgroundColor: DARK.paperHi },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(39, 159, 187, 0.14)',
                        color: DARK.amberHi,
                        '&:hover': { backgroundColor: 'rgba(39, 159, 187, 0.20)' },
                    },
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    fontFamily: FONT_DISPLAY,
                    backgroundColor: 'rgba(240, 237, 229, 0.06)',
                    color: DARK.text,
                    border: `1px solid rgba(240, 237, 229, 0.12)`,
                    borderRadius: 999,            // pill, matches button language
                    // Filled colored chips become raised gradient pads (depth
                    // scoped to .MuiChip-filled so outlined chips stay quiet).
                    '&.MuiChip-filled.MuiChip-colorPrimary': {
                        backgroundColor: DARK.amber,
                        backgroundImage: `linear-gradient(135deg, ${DARK.amberHi} 0%, ${DARK.amber} 55%, ${DARK.amberLo} 100%)`,
                        color: '#082028',
                        borderColor: DARK.amberHi,
                        boxShadow: RAISE_DARK,
                    },
                    '&.MuiChip-filled.MuiChip-colorSuccess': {
                        backgroundColor: DARK.success,
                        backgroundImage: `linear-gradient(135deg, #9FDDB5 0%, ${DARK.success} 55%, #5BA579 100%)`,
                        color: '#06200F',
                        borderColor: '#9FDDB5',
                        boxShadow: RAISE_DARK,
                    },
                },
                outlined: {
                    borderColor: 'rgba(240, 237, 229, 0.22)',
                    '&.MuiChip-colorPrimary': { borderColor: DARK.amber, color: DARK.amberHi },
                },
            },
        },
        MuiAccordion: {
            styleOverrides: {
                root: {
                    // 25px matches every other Tier-1 card (borderRadius: 2.5
                    // in sx === 2.5 × theme.shape.borderRadius (10) = 25px).
                    // The :first/:last-of-type overrides re-assert 25 against
                    // MUI's defaults, which otherwise clamp matching corners
                    // back to theme.shape.borderRadius (10) and produce an
                    // asymmetric squash when the Accordion has a div sibling
                    // on only one side.
                    borderRadius: 25,
                    overflow: 'hidden',
                    marginBottom: 16,
                    '&:before': { display: 'none' },
                    '&:first-of-type': {
                        borderTopLeftRadius: 25,
                        borderTopRightRadius: 25,
                    },
                    '&:last-of-type': {
                        borderBottomLeftRadius: 25,
                        borderBottomRightRadius: 25,
                    },
                    '&.Mui-expanded': {
                        marginTop: 0,
                        marginBottom: 16,
                    },
                    // MUI's default collapses marginBottom to 0 when an
                    // expanded Accordion is :last-of-type — that's more
                    // specific than our &.Mui-expanded rule, so we match
                    // its specificity here to keep the gap consistent.
                    '&.Mui-expanded:last-of-type': { marginBottom: 16 },
                },
            },
        },
        MuiAccordionSummary: {
            styleOverrides: {
                root: {
                    backgroundColor: 'transparent',
                    minHeight: 48,
                    '& .MuiAccordionSummary-content': { margin: '12px 0', alignItems: 'center' },
                    '&.Mui-expanded': { minHeight: 48 },
                    '&.Mui-expanded .MuiAccordionSummary-content': { margin: '12px 0' },
                    '&:hover': { backgroundColor: 'rgba(39, 159, 187, 0.06)' },
                },
            },
        },
        MuiDialog: {
            defaultProps: { disableScrollLock: true },
            styleOverrides: {
                paper: {
                    // Liquid Glass — matches the global MuiPaper/MuiCard
                    // treatment so dialogs read as glass over the body's
                    // cyan/warm radial bleed, not as a solid card.
                    backgroundColor: 'rgba(38, 41, 44, 0.38)',
                    backgroundImage: 'none',
                    backdropFilter: 'blur(28px) saturate(200%)',
                    WebkitBackdropFilter: 'blur(28px) saturate(200%)',
                    border: 'none',
                    borderRadius: 14,
                    boxShadow:
                        '0 32px 60px rgba(0, 0, 0, 0.7), ' +
                        '0 4px 12px rgba(0, 0, 0, 0.35), ' +
                        'inset 0 1px 0 rgba(255, 255, 255, 0.22), ' +
                        'inset 0 -1px 0 rgba(0, 0, 0, 0.35), ' +
                        'inset 1px 0 0 rgba(255, 255, 255, 0.08), ' +
                        'inset -1px 0 0 rgba(0, 0, 0, 0.20)',
                },
            },
        },
        MuiDialogTitle: {
            styleOverrides: {
                root: {
                    fontFamily: FONT_DISPLAY,
                    backgroundColor: 'transparent',
                    borderBottom: `1px solid ${DARK.divider}`,
                    color: DARK.text,
                    fontWeight: 600,
                    fontSize: '1.1rem',
                    letterSpacing: '-0.01em',
                },
            },
        },
        MuiDialogContent: {
            styleOverrides: {
                root: { backgroundColor: 'transparent', color: DARK.text },
            },
        },
        MuiDialogActions: {
            styleOverrides: {
                root: {
                    backgroundColor: 'transparent',
                    borderTop: `1px solid ${DARK.divider}`,
                    padding: '14px 20px',
                    gap: 8,
                },
            },
        },
        MuiListItem: {
            styleOverrides: {
                root: {
                    '&:hover': { backgroundColor: 'rgba(39, 159, 187, 0.06)' },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(39, 159, 187, 0.14)',
                        '&:hover': { backgroundColor: 'rgba(39, 159, 187, 0.20)' },
                    },
                },
            },
        },
        MuiCheckbox: {
            styleOverrides: {
                root: {
                    color: DARK.textDim,
                    '&.Mui-checked': { color: DARK.amber },
                    '&:hover': { backgroundColor: 'rgba(39, 159, 187, 0.08)' },
                },
            },
        },
        MuiFormControlLabel: {
            styleOverrides: { label: { fontFamily: FONT_DISPLAY, color: DARK.text, fontSize: '0.875rem' } },
        },
        MuiSlider: {
            styleOverrides: {
                // Color is driven by the root `color` via currentColor on the
                // slots, so any slider recolors just by setting `color` in sx
                // (channel strips pass their channel color). Default = the accent cyan.
                root: {
                    color: NORDIC_BLUE,
                    height: 4,
                    // Beat MUI's color="primary" root color (which is the
                    // accent cyan) so the DEFAULT slider is the accent cyan —
                    // needed in dark mode where the plain root rule lost the
                    // specificity battle. sx `color` on channel sliders still
                    // overrides this, so per-channel coloring is unaffected.
                    '&.MuiSlider-colorPrimary, &.MuiSlider-colorSecondary': {
                        color: NORDIC_BLUE,
                    },
                },
                rail: { backgroundColor: 'rgba(240, 237, 229, 0.12)', opacity: 1 },
                track: { backgroundColor: 'currentColor', border: 0 },
                thumb: {
                    backgroundColor: 'currentColor',
                    width: 16,
                    height: 16,
                    boxShadow: '0 2px 6px rgba(0, 0, 0, 0.5)',
                    '&:hover, &.Mui-focusVisible': {
                        boxShadow: '0 0 0 8px rgba(39, 159, 187, 0.18)',
                    },
                    '&.Mui-active': {
                        boxShadow: '0 0 0 12px rgba(39, 159, 187, 0.24)',
                    },
                },
                valueLabel: {
                    backgroundColor: DARK.paper,
                    color: DARK.text,
                    border: `1px solid ${DARK.divider}`,
                    borderRadius: 6,
                    fontWeight: 400,
                    fontSize: '0.75rem',
                },
                mark: { backgroundColor: 'rgba(240, 237, 229, 0.24)' },
                markActive: { backgroundColor: 'currentColor' },
            },
        },
        MuiLinearProgress: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(240, 237, 229, 0.08)',
                    borderRadius: 999,
                    overflow: 'hidden',
                },
                // Warm complement on the bar tells the eye "work in progress"
                // — distinct from the cool cyan that drives interactive chrome.
                bar: {
                    backgroundImage: `linear-gradient(90deg, ${DARK.warmLo} 0%, ${DARK.warm} 100%)`,
                },
            },
        },
        MuiCircularProgress: { styleOverrides: { root: { color: DARK.amber } } },
        MuiTabs: {
            styleOverrides: {
                root: {
                    '& .MuiTabs-indicator': {
                        backgroundColor: DARK.amber,
                        height: 2,
                        borderRadius: '2px 2px 0 0',
                    },
                },
            },
        },
        MuiTab: {
            styleOverrides: {
                root: {
                    fontFamily: FONT_DISPLAY,
                    color: DARK.textDim,
                    textTransform: 'none',
                    // Match h6 exactly so the side rail label and the
                    // in-card section title read as the same hierarchy.
                    fontWeight: 400,
                    fontSize: '1.2rem',
                    letterSpacing: '-0.01em',
                    '&.Mui-selected': { color: DARK.amberHi },
                    '&:hover': { color: DARK.text },
                },
            },
        },
        MuiBackdrop: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(5, 4, 3, 0.6)',
                    backdropFilter: 'blur(6px)',
                },
            },
        },
        MuiDivider: { styleOverrides: { root: { borderColor: DARK.divider } } },
        MuiIconButton: {
            styleOverrides: {
                root: {
                    color: DARK.textDim,
                    transition: 'background-color 220ms ease, color 220ms ease, transform 220ms cubic-bezier(0.16, 1, 0.3, 1)',
                    '&:hover': {
                        backgroundColor: 'rgba(39, 159, 187, 0.10)',
                        color: DARK.amberHi,
                    },
                    '&:active:not(.Mui-disabled)': {
                        transform: 'scale(0.92)',
                        transition: 'transform 80ms ease-out',
                    },
                },
            },
        },
        MuiToggleButton: {
            styleOverrides: {
                root: {
                    fontFamily: FONT_DISPLAY,
                    textTransform: 'none',
                    border: `1px solid ${DARK.divider}`,
                    color: DARK.textDim,
                    fontWeight: 400,
                    letterSpacing: '-0.005em',
                    '&:hover': {
                        backgroundColor: 'rgba(39, 159, 187, 0.06)',
                        color: DARK.text,
                    },
                    '&.Mui-selected': {
                        backgroundColor: DARK.amber,
                        backgroundImage: `linear-gradient(135deg, ${DARK.amberHi} 0%, ${DARK.amber} 55%, ${DARK.amberLo} 100%)`,
                        color: '#082028',
                        borderColor: DARK.amberHi,
                        boxShadow: RAISE_DARK,
                        '&:hover': {
                            backgroundImage: `linear-gradient(135deg, ${DARK.amberHi} 0%, ${DARK.amberHi} 55%, ${DARK.amber} 100%)`,
                            backgroundColor: DARK.amberHi,
                        },
                    },
                },
            },
        },
        MuiSwitch: {
            styleOverrides: {
                switchBase: {
                    '&.Mui-checked': {
                        color: DARK.amberHi,
                        '& + .MuiSwitch-track': {
                            backgroundColor: DARK.amber,
                            opacity: 0.6,
                        },
                    },
                },
                track: { backgroundColor: 'rgba(240, 237, 229, 0.18)' },
            },
        },
        MuiTooltip: {
            styleOverrides: {
                tooltip: {
                    fontFamily: FONT_DISPLAY,
                    backgroundColor: DARK.bg,
                    color: DARK.text,
                    border: `1px solid ${DARK.divider}`,
                    fontWeight: 400,
                    fontSize: '0.75rem',
                    borderRadius: 6,
                    boxShadow: '0 10px 24px rgba(0, 0, 0, 0.6)',
                    backdropFilter: 'blur(6px)',
                },
                arrow: { color: DARK.bg },
            },
        },
        MuiContainer: { styleOverrides: { root: { backgroundColor: 'transparent', background: 'transparent' } } },
        MuiAutocomplete: {
            styleOverrides: {
                root: { fontFamily: FONT_DISPLAY },
                option: { fontFamily: FONT_DISPLAY },
                noOptions: { fontFamily: FONT_DISPLAY },
                loading: { fontFamily: FONT_DISPLAY },
            },
        },
        MuiListItemText: {
            styleOverrides: {
                primary: { fontFamily: FONT_DISPLAY },
                secondary: { fontFamily: FONT_DISPLAY },
            },
        },
        MuiListItemButton: {
            styleOverrides: {
                root: { fontFamily: FONT_DISPLAY },
            },
        },
        MuiListSubheader: {
            styleOverrides: {
                root: { fontFamily: FONT_DISPLAY },
            },
        },
        MuiAlert: {
            styleOverrides: {
                root: { fontFamily: FONT_DISPLAY },
            },
        },
    },
});

theme = responsiveFontSizes(theme, {
    breakpoints: ['sm', 'md', 'lg'],
    factor: 2.4,
});

function makePaletteTheme(LIGHT) {
    return createTheme(theme, {
        palette: {
            mode: 'light',
            primary: { main: LIGHT.amber, light: LIGHT.amberHi, dark: LIGHT.amberLo, contrastText: onFill(LIGHT.amber, LIGHT.text) },
            secondary: { main: LIGHT.blue, light: LIGHT.blueDim, dark: LIGHT.blueLo, contrastText: LIGHT.onAccent },
            background: { default: LIGHT.bg, paper: LIGHT.paper },
            text: { primary: LIGHT.text, secondary: LIGHT.textDim, disabled: LIGHT.textFaint },
            action: {
                active: `rgba(${LIGHT.inkRgb}, 0.6)`,
                hover: `rgba(${LIGHT.inkRgb}, 0.04)`,
                selected: `rgba(${LIGHT.inkRgb}, 0.08)`,
                disabled: `rgba(${LIGHT.inkRgb}, 0.42)`,
                disabledBackground: `rgba(${LIGHT.inkRgb}, 0.10)`,
            },
            divider: LIGHT.divider,
            error: { main: LIGHT.error, contrastText: onFill(LIGHT.error, LIGHT.text) },
            warning: { main: LIGHT.warning, contrastText: onFill(LIGHT.warning, LIGHT.text) },
            success: { main: LIGHT.success, contrastText: onFill(LIGHT.success, LIGHT.text) },
            info: { main: LIGHT.blue },
            warm: { main: LIGHT.warm, light: LIGHT.warmHi, dark: LIGHT.warmLo, contrastText: onFill(LIGHT.accents.warm, LIGHT.text) },
            night: { main: LIGHT.night },
            // All palette tokens, and the Performance tab's colours.
            fragmenta: LIGHT,
            perf: LIGHT.perf,
        },
        components: {
            MuiCssBaseline: {
                styleOverrides: {
                    ':root': { colorScheme: 'light' },
                    body: {
                        margin: 0,
                        minHeight: '100vh',
                        // Three-layer backdrop, mirroring dark mode:
                        // soft cyan radial on the left, warm radial on the
                        // bottom-right, and a gentle diagonal cream linear.
                        // Opacities are low so the paper still reads neutral.
                        backgroundColor: LIGHT.bg,
                        // Warm ivory with only a faint top-to-bottom tone shift.
                        backgroundImage: `linear-gradient(180deg, ${LIGHT.bgTop} 0%, ${LIGHT.bg} 55%)`,
                        backgroundAttachment: 'fixed',
                        color: LIGHT.text,
                        fontFeatureSettings: '"cv11", "ss01", "ss03"',
                    },
                    '#root': { minHeight: '100vh' },
                    html: { scrollbarGutter: 'stable' },
                    '*::-webkit-scrollbar-track': {
                        background: 'transparent',
                        borderRadius: '999px',
                    },
                    '*::-webkit-scrollbar-thumb': {
                        background: `rgba(${LIGHT.inkRgb}, 0.18)`,
                        borderRadius: '999px',
                        border: '2px solid rgba(0, 0, 0, 0)',
                        backgroundClip: 'padding-box',
                        '&:hover': { background: `rgba(${LIGHT.inkRgb}, 0.32)` },
                    },
                    '*::-webkit-scrollbar-corner': { background: 'transparent' },
                    '*': {
                        scrollbarWidth: 'thin',
                        scrollbarColor: `rgba(${LIGHT.inkRgb}, 0.18) transparent`,
                    },
                },
            },
            MuiPaper: {
                styleOverrides: {
                    // A lifted, solid surface (menus, dialogs, panels): tone and
                    // a diffuse shadow do the separating, no glass or rims.
                    root: {
                        backgroundColor: LIGHT.surface,
                        backgroundImage: 'none',
                        backdropFilter: 'none',
                        WebkitBackdropFilter: 'none',
                        border: 'none',
                        boxShadow: LIGHT.shadowSoft,
                    },
                },
            },
            MuiCard: {
                styleOverrides: {
                    root: {
                        backgroundColor: LIGHT.surface,
                        backgroundImage: 'none',
                        backdropFilter: 'none',
                        WebkitBackdropFilter: 'none',
                        border: 'none',
                        borderRadius: RADIUS.card,
                        boxShadow: LIGHT.shadowSoft,
                        transition: 'background-color 180ms ease',
                    },
                },
            },
            MuiButton: {
                variants: [
                    {
                        props: { color: 'warm', variant: 'contained' },
                        style: {
                            backgroundImage: 'none',
                            backgroundColor: LIGHT.accents.warm,
                            color: onFill(LIGHT.accents.warm),
                            '&:hover': { backgroundImage: 'none', backgroundColor: mixHex(LIGHT.accents.warm, '#000000', 0.1) },
                        },
                    },
                ],
                styleOverrides: {
                    root: {
                        borderRadius: 12,
                        transition: 'background-color 160ms ease, color 160ms ease, border-color 160ms ease, box-shadow 160ms ease',
                    },
                    sizeLarge: { borderRadius: 16, minHeight: 48 },
                    contained: {
                        backgroundImage: 'none',
                        boxShadow: 'none',
                        '&:hover': { boxShadow: 'none' },
                        '&.Mui-disabled': {
                            backgroundImage: 'none',
                            backgroundColor: LIGHT.control,
                            color: `rgba(${LIGHT.inkRgb}, 0.38)`,
                            boxShadow: 'none',
                        },
                    },
                    // Actions are pastel fills with ink text: primary cool,
                    // secondary/success sage, error rose (warm is a variant).
                    containedPrimary: {
                        backgroundImage: 'none',
                        backgroundColor: LIGHT.accents.cool,
                        color: onFill(LIGHT.accents.cool),
                        '&:hover': { backgroundImage: 'none', backgroundColor: mixHex(LIGHT.accents.cool, '#000000', 0.1) },
                    },
                    containedSecondary: {
                        backgroundImage: 'none',
                        backgroundColor: LIGHT.accents.fresh,
                        color: onFill(LIGHT.accents.fresh),
                        '&:hover': { backgroundImage: 'none', backgroundColor: mixHex(LIGHT.accents.fresh, '#000000', 0.1) },
                    },
                    containedSuccess: {
                        backgroundImage: 'none',
                        backgroundColor: LIGHT.accents.fresh,
                        color: onFill(LIGHT.accents.fresh),
                        '&:hover': { backgroundImage: 'none', backgroundColor: mixHex(LIGHT.accents.fresh, '#000000', 0.1) },
                    },
                    containedError: {
                        backgroundImage: 'none',
                        backgroundColor: LIGHT.accents.hot,
                        color: onFill(LIGHT.accents.hot),
                        '&:hover': { backgroundImage: 'none', backgroundColor: mixHex(LIGHT.accents.hot, '#000000', 0.1) },
                    },
                    outlined: {
                        // No outline at rest: a tonal control surface, like the
                        // rest of the interactive chrome.
                        border: '1px solid transparent',
                        backgroundColor: LIGHT.control,
                        color: LIGHT.text,
                        '&:hover': {
                            border: '1px solid transparent',
                            backgroundColor: LIGHT.controlHover,
                            color: LIGHT.text,
                        },
                        '&.Mui-disabled': {
                            border: '1px solid transparent',
                            backgroundColor: LIGHT.control,
                            color: `rgba(${LIGHT.inkRgb}, 0.38)`,
                        },
                    },
                    outlinedError: {
                        border: '1px solid transparent',
                        backgroundColor: `rgba(${LIGHT.errorRgb}, 0.10)`,
                        color: LIGHT.error,
                        '&:hover': {
                            border: '1px solid transparent',
                            backgroundColor: `rgba(${LIGHT.errorRgb}, 0.16)`,
                            color: LIGHT.errorLo,
                        },
                    },
                    text: {
                        color: LIGHT.text,
                        '&:hover': {
                            backgroundColor: LIGHT.control,
                            color: LIGHT.text,
                        },
                    },
                },
            },
            // Fields sit on the control tone with no outline at rest; focus
            // shows a fine accent ring so it stays clearly visible.
            MuiTextField: {
                styleOverrides: {
                    root: {
                        '& .MuiOutlinedInput-root': {
                            backgroundColor: LIGHT.control,
                            borderRadius: RADIUS.input,
                            transition: 'background-color 160ms ease',
                            '& fieldset': { borderColor: 'transparent' },
                            '&:hover': { backgroundColor: LIGHT.controlHover },
                            '&:hover fieldset': { borderColor: 'transparent' },
                            '&.Mui-focused': { backgroundColor: LIGHT.surface },
                            '&.Mui-focused fieldset': { borderColor: LIGHT.amber, borderWidth: '1.5px' },
                        },
                    },
                },
            },
            MuiSelect: {
                styleOverrides: {
                    root: {
                        backgroundColor: LIGHT.control,
                        borderRadius: RADIUS.input,
                        transition: 'background-color 160ms ease',
                        '& .MuiOutlinedInput-notchedOutline': { borderColor: 'transparent' },
                        '&:hover': { backgroundColor: LIGHT.controlHover },
                        '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'transparent' },
                        '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: LIGHT.amber, borderWidth: '1.5px' },
                    },
                },
            },
            MuiModal: {
                defaultProps: { disableScrollLock: true },
            },
            // See dark theme above — same dropdown/popover paper treatment.
            MuiPopover: {
                defaultProps: { disableScrollLock: true },
                styleOverrides: {
                    paper: { border: 'none', borderRadius: 16, overflow: 'hidden', boxShadow: LIGHT.shadowRaised },
                },
            },
            MuiMenu: {
                defaultProps: { disableScrollLock: true },
                styleOverrides: {
                    paper: { border: 'none', borderRadius: 16, overflow: 'hidden', boxShadow: LIGHT.shadowRaised },
                    list: { padding: 4 },
                },
            },
            MuiDrawer: {
                defaultProps: { disableScrollLock: true },
            },
            MuiMenuItem: {
                styleOverrides: {
                    root: {
                        backgroundColor: LIGHT.paper,
                        '&:hover': { backgroundColor: LIGHT.paperHi },
                        '&.Mui-selected': {
                            backgroundColor: `rgba(${LIGHT.amberRgb}, 0.14)`,
                            color: LIGHT.amberLo,
                            '&:hover': { backgroundColor: `rgba(${LIGHT.amberRgb}, 0.20)` },
                        },
                    },
                },
            },
            // Chips and tags: 999 pills on the control tone; coloured ones are
            // the pastel (filled) or the pale tint (outlined), with ink text.
            MuiChip: {
                styleOverrides: {
                    root: {
                        backgroundColor: LIGHT.control,
                        color: LIGHT.text,
                        border: 'none',
                        borderRadius: 999,
                        '&.MuiChip-filled.MuiChip-colorPrimary': { backgroundColor: LIGHT.accents.cool, color: onFill(LIGHT.accents.cool) },
                        '&.MuiChip-filled.MuiChip-colorSuccess': { backgroundColor: LIGHT.accents.fresh, color: onFill(LIGHT.accents.fresh) },
                        '&.MuiChip-filled.MuiChip-colorWarning': { backgroundColor: LIGHT.accents.warm, color: onFill(LIGHT.accents.warm) },
                        '&.MuiChip-filled.MuiChip-colorError': { backgroundColor: LIGHT.accents.hot, color: onFill(LIGHT.accents.hot) },
                    },
                    outlined: {
                        border: 'none',
                        backgroundColor: LIGHT.control,
                        '&.MuiChip-colorPrimary': { backgroundColor: LIGHT.perf.laneTints[0], color: LIGHT.text },
                        '&.MuiChip-colorSuccess': { backgroundColor: LIGHT.perf.laneTints[2], color: LIGHT.text },
                        '&.MuiChip-colorWarning': { backgroundColor: LIGHT.perf.laneTints[3], color: LIGHT.text },
                        '&.MuiChip-colorError': { backgroundColor: LIGHT.perf.laneTints[1], color: LIGHT.text },
                    },
                    clickable: { '&:hover': { filter: 'brightness(0.97)' } },
                },
            },
            // Sections inside a card: a tonal surface, no shadow (not a card
            // within a card).
            // A section inside a card: the section tone, no shadow or rule.
            MuiAccordion: {
                styleOverrides: {
                    root: {
                        borderRadius: 18,
                        overflow: 'hidden',
                        marginBottom: 16,
                        backgroundColor: LIGHT.section,
                        backgroundImage: 'none',
                        boxShadow: 'none',
                        border: 'none',
                        '&:before': { display: 'none' },
                        '&:first-of-type': { borderTopLeftRadius: 18, borderTopRightRadius: 18 },
                        '&:last-of-type': { borderBottomLeftRadius: 18, borderBottomRightRadius: 18 },
                        '&.Mui-expanded': { marginTop: 0, marginBottom: 16 },
                        '&.Mui-expanded:last-of-type': { marginBottom: 16 },
                    },
                },
            },
            MuiAccordionSummary: {
                styleOverrides: {
                    root: {
                        backgroundColor: 'transparent',
                        '&:hover': { backgroundColor: LIGHT.controlHover },
                    },
                },
            },
            MuiDialog: {
                defaultProps: { disableScrollLock: true },
                styleOverrides: {
                    paper: {
                        backgroundColor: LIGHT.surface,
                        backgroundImage: 'none',
                        backdropFilter: 'none',
                        WebkitBackdropFilter: 'none',
                        border: 'none',
                        borderRadius: 24,
                        boxShadow: LIGHT.shadowRaised,
                    },
                },
            },
            MuiDialogTitle: {
                styleOverrides: {
                    root: {
                        backgroundColor: 'transparent',
                        borderBottom: 'none',
                        color: LIGHT.text,
                        fontFamily: FONT_DISPLAY,
                        fontWeight: 600,
                        fontSize: '1.35rem',
                        letterSpacing: '-0.015em',
                        paddingTop: 22,
                    },
                },
            },
            MuiDialogContent: { styleOverrides: { root: { backgroundColor: 'transparent', color: LIGHT.text } } },
            MuiDialogActions: {
                styleOverrides: {
                    root: {
                        backgroundColor: 'transparent',
                        borderTop: `1px solid ${LIGHT.divider}`,
                    },
                },
            },
            MuiListItem: {
                styleOverrides: {
                    root: {
                        '&:hover': { backgroundColor: `rgba(${LIGHT.amberRgb}, 0.06)` },
                        '&.Mui-selected': {
                            backgroundColor: `rgba(${LIGHT.amberRgb}, 0.14)`,
                            '&:hover': { backgroundColor: `rgba(${LIGHT.amberRgb}, 0.20)` },
                        },
                    },
                },
            },
            MuiCheckbox: {
                styleOverrides: {
                    root: {
                        color: LIGHT.textDim,
                        '&.Mui-checked': { color: LIGHT.amberHi },
                        '&:hover': { backgroundColor: LIGHT.control },
                    },
                },
            },
            MuiFormControlLabel: { styleOverrides: { label: { color: LIGHT.text } } },
            MuiSlider: {
                styleOverrides: {
                    root: {
                        color: LIGHT.amberHi,
                        height: 6,
                        '&.MuiSlider-colorPrimary, &.MuiSlider-colorSecondary': {
                            color: LIGHT.amberHi,
                        },
                    },
                    // Shallow inset track.
                    rail: {
                        backgroundColor: `rgba(${LIGHT.inkRgb}, 0.10)`,
                        boxShadow: `inset 0 1px 1px rgba(${LIGHT.inkRgb}, 0.12)`,
                        opacity: 1,
                    },
                    track: { backgroundColor: 'currentColor', border: 0 },
                    // Circular handle: ivory with a ring in the slider's colour.
                    thumb: {
                        width: 16,
                        height: 16,
                        backgroundColor: LIGHT.onAccent,
                        border: '2px solid currentColor',
                        boxShadow: `0 1px 3px rgba(${LIGHT.inkRgb}, 0.22)`,
                        transition: 'box-shadow 150ms ease, width 150ms ease, height 150ms ease',
                        '&::before': { boxShadow: 'none' },
                        '&:hover, &.Mui-focusVisible': {
                            boxShadow: `0 1px 3px rgba(${LIGHT.inkRgb}, 0.22), 0 0 0 6px rgba(${LIGHT.inkRgb}, 0.07)`,
                        },
                        '&.Mui-active': {
                            boxShadow: `0 1px 3px rgba(${LIGHT.inkRgb}, 0.22), 0 0 0 8px rgba(${LIGHT.inkRgb}, 0.09)`,
                        },
                    },
                    valueLabel: {
                        backgroundColor: LIGHT.text,
                        color: LIGHT.onAccent,
                        borderRadius: 6,
                        fontWeight: 400,
                    },
                    mark: { backgroundColor: `rgba(${LIGHT.inkRgb}, 0.28)` },
                    markActive: { backgroundColor: 'currentColor' },
                },
            },
            MuiLinearProgress: {
                styleOverrides: {
                    root: {
                        backgroundColor: `rgba(${LIGHT.inkRgb}, 0.10)`,
                        borderRadius: 999,
                        overflow: 'hidden',
                    },
                    bar: {
                        backgroundImage: `linear-gradient(90deg, ${LIGHT.warmLo} 0%, ${LIGHT.warm} 100%)`,
                    },
                },
            },
            MuiCircularProgress: { styleOverrides: { root: { color: LIGHT.amber } } },
            MuiTabs: {
                styleOverrides: {
                    root: {
                        '& .MuiTabs-indicator': {
                            backgroundColor: LIGHT.amber,
                            height: 2,
                            borderRadius: '2px 2px 0 0',
                        },
                    },
                },
            },
            MuiTab: {
                styleOverrides: {
                    root: {
                        color: LIGHT.textDim,
                        textTransform: 'none',
                        fontWeight: 400,
                        '&.Mui-selected': { color: LIGHT.amberLo },
                        '&:hover': { color: LIGHT.text },
                    },
                },
            },
            MuiBackdrop: {
                styleOverrides: {
                    root: {
                        backgroundColor: `rgba(${LIGHT.inkRgb}, 0.32)`,
                        backdropFilter: 'blur(4px)',
                    },
                },
            },
            MuiDivider: { styleOverrides: { root: { borderColor: LIGHT.divider } } },
            MuiIconButton: {
                styleOverrides: {
                    root: {
                        color: LIGHT.textDim,
                        transition: 'all 160ms ease',
                        '&:hover': {
                            backgroundColor: `rgba(${LIGHT.amberRgb}, 0.10)`,
                            color: LIGHT.amberLo,
                        },
                    },
                },
            },
            // Segmented controls: a tonal track, the selected segment a soft
            // pill of the accent (the one place a fully rounded end is used).
            // Segmented controls: a control-tone track; the selected segment is
            // a 999 pill in the cool pastel.
            MuiToggleButtonGroup: {
                styleOverrides: {
                    root: { backgroundColor: LIGHT.control, borderRadius: 14, padding: 3, gap: 2 },
                    grouped: { border: 'none !important', borderRadius: '999px !important', margin: '0 !important' },
                },
            },
            MuiToggleButton: {
                styleOverrides: {
                    root: {
                        textTransform: 'none',
                        border: 'none',
                        borderRadius: 999,
                        color: LIGHT.textDim,
                        fontWeight: 400,
                        padding: '6px 16px',
                        transition: 'background-color 160ms ease, color 160ms ease',
                        '&:hover': { backgroundColor: LIGHT.controlHover, color: LIGHT.text },
                        '&.Mui-selected': {
                            backgroundColor: LIGHT.accents.cool,
                            backgroundImage: 'none',
                            color: onFill(LIGHT.accents.cool),
                            fontWeight: 600,
                            boxShadow: 'none',
                            '&:hover': { backgroundImage: 'none', backgroundColor: LIGHT.accents.cool },
                        },
                    },
                },
            },
            // Switches: the cool pastel track when on, an ivory thumb.
            MuiSwitch: {
                styleOverrides: {
                    switchBase: {
                        color: LIGHT.onAccent,
                        '&.Mui-checked': {
                            color: LIGHT.onAccent,
                            '& + .MuiSwitch-track': { backgroundColor: LIGHT.accents.cool, opacity: 1 },
                        },
                    },
                    thumb: { boxShadow: `0 1px 2px rgba(${LIGHT.inkRgb}, 0.25)` },
                    track: { backgroundColor: LIGHT.controlHover, opacity: 1, borderRadius: 999 },
                },
            },
            MuiTooltip: {
                styleOverrides: {
                    tooltip: {
                        backgroundColor: LIGHT.text,
                        color: LIGHT.onAccent,
                        fontWeight: 400,
                        fontSize: '0.75rem',
                        borderRadius: 6,
                        boxShadow: `0 8px 18px rgba(${LIGHT.inkRgb}, 0.24)`,
                    },
                    arrow: { color: LIGHT.text },
                },
            },
        },
    });
}

export const appStyles = {
    root: {
        minHeight: '100vh',
        background: 'transparent',
        backgroundColor: 'transparent',
        overflow: 'visible',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
    },
    container: (showWelcomePage) => ({
        // Symmetric outer padding. Top is 0 — the header has its own
        // internal top padding so it sticks at viewport y=0 without
        // any visible movement during the first pixels of scroll.
        pt: 0,
        pb: { xs: 1.5, sm: 2, md: 3 },
        px: { xs: 1.5, sm: 2, md: 3 },
        // Fill `root` (already minHeight:100vh) rather than re-declaring 100vh.
        // Two stacked 100vh layers + the header tipped scrollHeight a few px
        // past the viewport, producing a phantom page scroll with nothing
        // actually clipped.
        flex: 1,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'transparent',
        background: 'transparent',
        overflow: 'visible',
        boxSizing: 'border-box',
        width: '100%',
        maxWidth: '100%',
        filter: showWelcomePage ? 'blur(8px)' : 'none',
        transition: 'filter 0.3s ease-in-out',
    }),
    headerRow: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: { xs: 'stretch', md: 'center' },
        flexDirection: { xs: 'column', md: 'row' },
        gap: { xs: 1.25, sm: 1.75, md: 2 },
        mb: { xs: 0.5, sm: 0.75 },
        // Sticky to viewport top — no chrome by default; the scrolled
        // style overlays the glass treatment once the page moves. The
        // top padding lives inside this element (not on Container) so
        // the header sticks at y=0 from the very first pixel of scroll
        // — zero movement.
        position: 'sticky',
        top: 0,
        zIndex: 100,
        pt: { xs: 1.5, sm: 2, md: 3 },
        pb: { xs: 0.75, sm: 1 },
        transition: 'background-color 220ms ease, backdrop-filter 220ms ease, border-color 220ms ease',
        backdropFilter: 'blur(0px)',
        WebkitBackdropFilter: 'blur(0px)',
        backgroundColor: 'transparent',
        borderBottom: '1px solid transparent',
    },
    // Scrolled: the canvas colour behind the header, no glass.
    headerRowScrolled: (theme) => ({
        backgroundColor: `rgba(${theme.palette.fragmenta.bgRgb}, 0.94)`,
        boxShadow: `0 1px 0 rgba(${theme.palette.fragmenta.inkRgb}, 0.06)`,
    }),
    headerBrand: {
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        gap: { xs: 1, sm: 1.5 },
        py: 0,
    },
    logo: {
        width: 40,
        height: 40,
        backgroundImage: 'url(/fragmenta_icon_1024.png)',
        backgroundSize: 'contain',
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'center',
        filter: 'drop-shadow(0 2px 6px rgba(55, 58, 48, 0.18))',
    },
    title: {
        color: 'text.primary',
        fontFamily: FONT_DISPLAY,
        fontWeight: 600,
        fontSize: { xs: '1.3rem', sm: '1.4rem' },
        letterSpacing: '-0.02em',
        fontVariationSettings: '"opsz" 72',
        lineHeight: 1.1,
    },
    headerActionsContainer: (isCompactLayout) => ({
        display: 'flex',
        alignItems: 'stretch',
        justifyContent: { xs: 'flex-start', md: 'flex-end' },
        gap: { xs: 1, sm: 1.5 },
        flexDirection: isCompactLayout ? 'column' : 'row',
        flexWrap: isCompactLayout ? 'wrap' : 'nowrap',
        width: { xs: '100%', md: 'auto' },
        // Cards inset from the Container only — no extra pr needed.
    }),
    emphasizedPrimaryBody2: {
        fontWeight: 'bold',
        color: 'primary.main',
    },
    advancedSettingsDetails: {
        // Pure pass-through. No maxHeight / inner scroll — that was
        // truncating the bottom before the parent's rounded corner
        // (the "cut" look). Content flows naturally; the whole page
        // scrolls if the accordion gets tall.
    },
    mainLayout: (isCompactLayout, isIconOnly) => ({
        display: 'flex',
        flexDirection: { xs: 'column', md: 'row' },
        width: '100%',
        flex: 1,
        gap: { xs: 1.25, sm: 1.75, md: 2.5 },
        borderRadius: 3,
        minHeight: 0,
        // Reserve space on the left for the fixed nav rail in vertical
        // mode. Compact mode keeps the rail in flow (horizontal at top).
        pl: isCompactLayout
            ? 0
            : isIconOnly
                ? `calc(64px + ${24}px)`
                : `calc(220px + ${24}px)`,
    }),
    navPaper: (isCompactLayout, isIconOnly) => ({
        width: isCompactLayout ? '100%' : isIconOnly ? 64 : 220,
        borderRadius: RADIUS.card,
        overflow: 'hidden',
        // Its own card, lifted off the canvas like the page cards.
        backgroundColor: (t) => t.palette.fragmenta.surface,
        boxShadow: (t) => t.palette.fragmenta.shadowSoft,
        display: 'flex',
        flexDirection: 'column',
        // Vertical mode: `position: fixed` so the rail is anchored to
        // the viewport directly. `top` is provided dynamically from
        // App.js via a JS measurement of the first card's natural top
        // edge — guarantees pixel-perfect alignment regardless of
        // header content height or breakpoint.
        ...(isCompactLayout
            ? { height: '100%' }
            : {
                position: 'fixed',
                left: { xs: 12, sm: 16, md: 24 },
                maxHeight: { xs: 'calc(100vh - 90px)', md: 'calc(100vh - 120px)' },
                zIndex: 50,
            }),
    }),
    navigationTabs: (isCompactLayout, isIconOnly = false) => ({
        height: isCompactLayout ? 'auto' : '100%',
        p: { xs: 0.5, sm: 1 },
        '& .MuiTabs-flexContainer': { gap: { xs: 0.25, sm: 0.75 } },
        '& .MuiTabs-indicator': {
            display: 'none',
        },
        '& .MuiTab-root': {
            alignItems: 'center',
            justifyContent: (isCompactLayout || isIconOnly) ? 'center' : 'flex-start',
            textAlign: (isCompactLayout || isIconOnly) ? 'center' : 'left',
            minHeight: 44,
            fontSize: '0.875rem',
            fontWeight: 400,
            textTransform: 'none',
            color: 'text.secondary',
            borderRadius: 2,
            px: isIconOnly ? 0 : { xs: 1, sm: 1.5 },
            py: { xs: 0.75, sm: 1 },
            mx: isCompactLayout ? 0.25 : 0,
            minWidth: isIconOnly ? 0 : undefined,
            '& .MuiTab-iconWrapper': {
                marginBottom: '0 !important',
            },
            borderRadius: '14px',
            transition: 'background-color 160ms ease, color 160ms ease',
            '& svg': { strokeWidth: 1.6 },
            // Selected: a soft tint of the accent, no outline.
            '&.Mui-selected': {
                color: 'text.primary',
                fontWeight: 600,
                backgroundColor: (t) => t.palette.fragmenta.navSelected,
                '& svg': { color: (t) => t.palette.primary.main },
            },
            '&:hover': {
                color: 'text.primary',
                backgroundColor: (t) => t.palette.fragmenta.control,
            },
            '&.Mui-selected:hover': {
                backgroundColor: (t) => t.palette.fragmenta.navSelected,
            },
        },
    }),
    mainContentBox: (muiTheme) => ({
        // Layout-only Box (no Paper chrome). Lives as a flex sibling to
        // the nav rail; the cards inside sit directly on the app bg.
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: { xs: 'auto', md: 0 },
        overflow: 'visible',
        // Defaults only: :where() keeps these at zero extra specificity, so a
        // component's own sx size (big readouts, labels) still wins.
        '& :where(.MuiTypography-body1, .MuiTypography-body2, .MuiTypography-caption, .MuiTypography-subtitle2)': {
            fontSize: muiTheme.typography.body2.fontSize,
            lineHeight: muiTheme.typography.body2.lineHeight,
        },
        '& :where(.MuiButton-startIcon svg, .MuiButton-endIcon svg, .MuiIconButton-root svg, .MuiAccordionSummary-expandIconWrapper svg)': {
            width: 20,
            height: 20,
        },
    }),
    // Used by Dataset Status / Training Configuration / Audio Generation /
    // Selected Model cards. Layout + motion only — MuiPaper owns bg/border/
    // shadow so the theme palette actually applies.
    elevatedInfoCard: {
        // Phase 3 spacing: generous interior padding so Tier-1 cards
        // breathe. Aligns with the industry-app feel — cards have room.
        p: { xs: 2.25, sm: 3 },
        mb: 2,
        borderRadius: 2.5,
        transition: 'all 0.3s ease',
    },
    // Sticks the Dataset Status card to the top of the viewport during
    // page scroll, mirroring the left nav rail's anchored position.
    // Sticky (not fixed) so the card stays inside the Grid layout — no
    // need to reserve space. Only enabled on md+ where the card sits in
    // a side column; on compact widths it falls below the upload area
    // and sticky would be useless.
    datasetStatusSticky: (navTopPx) => ({
        position: { md: 'sticky' },
        top: { md: `${navTopPx}px` },
        // Cancel the hover lift while sticky — translateY would offset
        // the stuck position and make the card appear to nudge upward.
        '&:hover': { transform: 'none' },
    }),
    modelMissingAlert: {
        mt: 2,
        backgroundColor: 'rgba(219, 80, 68, 0)',
        border: '1px solid #DB5044',
        borderRadius: 2,
        '& .MuiAlert-icon': {
            color: '#DB5044',
        },
    },
    modelMissingLink: {
        fontWeight: 600,
        color: 'primary.main',
        textDecoration: 'none',
        '&:hover': {
            textDecoration: 'underline',
        },
    },
    dataProcessingGrid: {
        flex: 1,
        minHeight: 0,
        flexWrap: 'wrap',
        alignItems: 'stretch',
    },
    primaryPaneItem: {
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        overflow: 'visible',
    },
    primaryPaneContent: {
        flex: 1,
        overflow: 'visible',
        pr: { xs: 0, md: 1 },
        display: 'flex',
        flexDirection: 'column',
        gap: 3,
    },
    secondaryPaneItem: {
        display: 'flex',
        flexDirection: 'column',
        overflow: 'visible',
    },
    secondaryPaneContent: {
        flex: 1,
        overflow: 'visible',
        pl: { xs: 0, md: 1 },
    },
    addRowButton: {
        mb: { xs: 2, sm: 3 },
    },
    sectionInfoAlert: {
        mb: 2,
    },
    recentFilesBlock: {
        mt: 1,
        minWidth: 0,
        overflowWrap: 'anywhere',
        wordBreak: 'break-word',
    },
    responsiveGrid: {
        height: 'auto',
        flexWrap: 'wrap',
    },
    formControlMarginBottom: {
        mb: 2,
    },
    fieldMarginBottom: {
        mb: 2,
    },
    fieldMarginBottomLarge: {
        mb: 3,
    },
    sliderRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 2,
    },
    // Header row for a field: label + an info icon that exposes the help text
    // on hover. Used across the Training tab's Advanced settings to keep the
    // form compact — captions live in tooltips, not below every control.
    fieldLabelRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 0.5,
        mb: 1,
    },
    fieldHelpIcon: {
        display: 'inline-flex',
        alignItems: 'center',
        cursor: 'help',
        color: 'text.secondary',
        opacity: 0.5,
        transition: 'opacity 150ms ease',
        '&:hover': { opacity: 0.95 },
    },
    sliderFlexGrow: {
        flex: 1,
    },
    sliderInputSmall: {
        width: { xs: '72px', sm: '80px' },
    },
    sliderInputMedium: {
        width: { xs: '88px', sm: '100px' },
    },
    trainingActionRow: {
        display: 'flex',
        gap: { xs: 1.25, sm: 2 },
        flexDirection: { xs: 'column', sm: 'row' },
    },
    actionButtonFlexGrow: {
        flex: 1,
    },
    mediumWeightBodyText: {
        fontWeight: 500,
    },
    trainingMonitorWrap: {
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
    },
    trainingMonitorHeaderRow: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        mb: 1,
    },
    trainingMonitorTitle: {
        mb: 0,
    },
    trainingMonitorStatusInline: (muiTheme) => ({
        display: 'flex',
        alignItems: 'center',
        gap: 0.75,
        px: 1,
        py: 0.35,
        borderRadius: 999,
        border: '1px solid',
        borderColor: 'divider',
        backgroundColor: muiTheme.palette.mode === 'dark'
            ? 'rgba(10, 15, 23, 0.7)'
            : 'rgba(255, 255, 255, 0.9)',
    }),
    trainingMonitorStatusDot: (status, animate = false) => ({
        width: 8,
        height: 8,
        borderRadius: '50%',
        bgcolor: status === 'live'
            ? 'success.main'
            : status === 'error'
                ? 'error.main'
                : status === 'complete'
                    ? 'primary.main'
                    : 'text.secondary',
        animation: animate ? 'pulse 2s infinite' : 'none',
        '@keyframes pulse': {
            '0%': { opacity: 1 },
            '50%': { opacity: 0.5 },
            '100%': { opacity: 1 },
        },
    }),
    trainingMonitorStatusText: (status) => ({
        color: status === 'live'
            ? 'success.main'
            : status === 'error'
                ? 'error.main'
                : status === 'complete'
                    ? 'primary.main'
                    : 'text.secondary',
        fontWeight: 600,
        letterSpacing: '0.03em',
        textTransform: 'uppercase',
        fontSize: { xs: '0.64rem', sm: '0.7rem' },
        lineHeight: 1,
    }),
    generationModelRow: {
        display: 'flex',
        alignItems: 'center',
        gap: { xs: 1, sm: 2 },
        mb: 2,
    },
    refreshModelsButton: {
        minWidth: 40,
    },
    durationRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        mb: 2.5,
    },
    // Small caps field label (DURATION, PROMPT …).
    fieldCaps: {
        fontSize: '0.625rem',
        fontWeight: 600,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: 'text.secondary',
        minWidth: 64,
    },
    // A value readout set in the display face (10 s, 64%).
    bigValue: {
        fontFamily: FONT_DISPLAY,
        fontSize: '0.95rem',
        fontWeight: 600,
        fontVariantNumeric: 'tabular-nums',
        minWidth: 44,
        textAlign: 'right',
    },
    generatingWrap: {
        mb: 3,
    },
    generatingHeader: {
        display: 'flex',
        alignItems: 'center',
        mb: 1,
    },
    generatingSpinner: {
        mr: 1,
    },
    generatingProgress: {
        // Explicit width + relative positioning containment. Without these,
        // some browsers / window configurations allowed the bar's inner
        // fill (a CSS-transformed pseudo-element) to be measured against a
        // transformed ancestor's frame and visually escape to the page's
        // left edge before snapping back. Belt-and-suspenders containment.
        height: 8,
        borderRadius: 4,
        width: '100%',
        display: 'block',
        position: 'relative',
        overflow: 'hidden',
    },
    generatingHint: {
        mt: 1,
        display: 'block',
    },
    generateButton: {
        mb: 2,
    },
    warningAlertTop: {
        mt: 2,
    },
    sectionCardHeader: {
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        mb: 1.5,
    },
    // Card titles stand alone (no icon), as in the design.
    sectionCardIcon: {
        display: 'none',
    },
    // Layout glue only — typography is owned by the Tier-1 (h6) variant.
    // Leaving fontWeight here would override the canonical 600.
    sectionCardTitle: {},
    // Same treatment as elevatedInfoCard — layout + motion only, theme
    // owns colour. Kept as its own export because the Selected Model card
    // sits in a different grid pane and may want page-specific tweaks
    // later in the fine-pass.
    selectedModelCard: {
        p: { xs: 2.25, sm: 3 },
        mb: 2,
        borderRadius: 2.5,
        transition: 'all 0.3s ease',
    },
    boldBodyText: {
        fontWeight: 'bold',
    },
    selectedModelMetaText: {
        display: 'block',
        mt: 0.5,
    },
    unwrappedInfoWrap: {
        mt: 2,
    },
    dialogBodyText: {
        mt: 3,
    },
    dialogErrorText: {
        mt: 2,
    },
    // Bottom-left floating dock — vertical strip mirroring the icon-only
    // nav rail. Icon buttons inside use the same flat rounded-square
    // language as MuiTab in icon-only mode; hovering anywhere on the dock
    // fades in each item's label to the right (via .dock-label).
    // Layout-only — bg/border/shadow come from MuiPaper.root theme override.
    bottomDock: (muiTheme) => ({
        position: 'fixed',
        left: { xs: muiTheme.spacing(1.5), sm: muiTheme.spacing(2), md: muiTheme.spacing(3) },
        // Bottom = page Container.pb + TabPanel.pb so the dock's bottom
        // edge sits at the same y as the page's lower bar (e.g. the
        // Performance model/LoRA/Ckpt strip), through any resize.
        // Container.pb = { xs: 1.5, sm: 2, md: 3 } (theme.container)
        // TabPanel.pb  = { xs: 2,   sm: 2, md: 3 } (tabPanelStyles.root)
        bottom: { xs: muiTheme.spacing(3.5), sm: muiTheme.spacing(4), md: muiTheme.spacing(6) },
        zIndex: 1350,
        display: 'flex',
        flexDirection: 'column',
        p: { xs: 0.5, sm: 1 },
        gap: { xs: 0.25, sm: 0.5 },
        borderRadius: RADIUS.panel,
        backgroundColor: 'transparent',
        boxShadow: 'none',
        '& .dock-label': {
            opacity: 0,
            transform: 'translate(-8px, -50%)',
            pointerEvents: 'none',
            transition: 'opacity 220ms ease, transform 220ms ease',
        },
        '&:hover .dock-label, &:focus-within .dock-label': {
            opacity: 1,
            transform: 'translate(0, -50%)',
            pointerEvents: 'auto',
        },
    }),
    dockItem: {
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
    },
    // Matches `& .MuiTab-root` styling in navigationTabs so dock icons read
    // exactly like nav tab icons.
    dockIconButton: {
        width: { xs: 40, sm: 46 },
        height: { xs: 40, sm: 46 },
        flexShrink: 0,
        borderRadius: 2,
        color: 'text.secondary',
        transition: 'background-color 160ms ease, color 160ms ease',
        '&:hover': {
            color: 'text.primary',
            backgroundColor: (t) => `rgba(${t.palette.fragmenta.amberRgb}, 0.08)`,
        },
        '&.Mui-disabled': {
            opacity: 0.45,
        },
    },
    dockIconButtonAccent: {
        color: 'primary.main',
        '&:hover': {
            color: 'primary.light',
            backgroundColor: (t) => `rgba(${t.palette.fragmenta.amberRgb}, 0.10)`,
        },
    },
    // Hamburger trigger shown in place of the full dock when the viewport
    // is too small to fit the 6-icon stack. Same visual treatment as a
    // single dock icon button so the surface stays consistent.
    dockHamburger: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            position: 'fixed',
            left: { xs: muiTheme.spacing(1.5), sm: muiTheme.spacing(2), md: muiTheme.spacing(3) },
            // Mirror bottomDock's bottom math so the collapsed hamburger
            // sits at the same y as the page's lower bar — see bottomDock
            // above for the breakdown.
            bottom: { xs: muiTheme.spacing(3.5), sm: muiTheme.spacing(4), md: muiTheme.spacing(6) },
            zIndex: 1350,
            width: { xs: 38, sm: 42 },
            height: { xs: 38, sm: 42 },
            border: isDark
                ? '1px solid rgba(194, 207, 228, 0.22)'
                : '1px solid rgba(15, 23, 42, 0.16)',
            background: isDark
                ? 'linear-gradient(145deg, rgba(18, 25, 38, 0.96) 0%, rgba(12, 19, 30, 0.96) 100%)'
                : 'linear-gradient(145deg, rgba(255, 255, 255, 0.98) 0%, rgba(245, 250, 255, 0.98) 100%)',
            color: isDark ? 'primary.light' : 'primary.main',
            boxShadow: isDark
                ? '0 14px 24px rgba(4, 8, 14, 0.5)'
                : '0 14px 24px rgba(15, 23, 42, 0.14)',
            '&:hover': {
                transform: 'translateY(-1px)',
                background: isDark
                    ? 'linear-gradient(145deg, rgba(20, 28, 42, 1) 0%, rgba(14, 22, 34, 1) 100%)'
                    : 'linear-gradient(145deg, rgba(244, 250, 255, 1) 0%, rgba(236, 245, 252, 1) 100%)',
                boxShadow: isDark
                    ? '0 18px 28px rgba(4, 8, 14, 0.6)'
                    : '0 18px 28px rgba(15, 23, 42, 0.18)',
            },
        };
    },
    dockLabel: {
        position: 'absolute',
        left: '100%',
        top: '50%',
        transform: 'translateY(-50%)',
        ml: 2.5,
        whiteSpace: 'nowrap',
        color: 'text.primary',
        fontSize: { xs: '0.78rem', sm: '0.82rem' },
        fontWeight: 400,
    },
    infoDialogTitleRow: {
        display: 'inline-flex',
        alignItems: 'center',
        gap: 1,
    },
    infoDialogIntro: {
        mt: 2.5,
        mb: 4.5,
        color: 'text.secondary',
    },
    infoDialogSectionTitle: {
        mt: 2.25,
        mb: 1,
        color: 'text.primary',
        fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
        fontWeight: 700,
        letterSpacing: '0.03em',
        textTransform: 'uppercase',
        fontSize: { xs: '0.7rem', sm: '0.74rem' },
    },
    infoDialogActionStack: {
        mt: 0.25,
        display: 'flex',
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'center',
        gap: 1,
    },
    infoDocButton: {
        fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
    },
};

// Top-right app toolbar: GPU memory, then a tonal pill of icon buttons.
export const appToolbarStyles = {
    root: (t) => ({
        display: 'flex',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 1.25,
        py: '5px',
        pr: '5px',
        pl: 2,
        borderRadius: RADIUS.pill,
        backgroundColor: t.palette.fragmenta.surface,
        boxShadow: t.palette.fragmenta.shadowSoft,
    }),
    gpu: { display: 'flex', alignItems: 'center', gap: 1.25 },
    gpuLabel: {
        fontSize: '0.62rem',
        fontWeight: 600,
        letterSpacing: '0.08em',
        color: 'text.secondary',
    },
    meter: (t) => ({
        width: 72,
        height: 6,
        borderRadius: RADIUS.pill,
        backgroundColor: t.palette.fragmenta.controlHover,
        overflow: 'hidden',
    }),
    meterFill: (pct) => (t) => ({
        width: `${pct}%`,
        height: '100%',
        borderRadius: RADIUS.pill,
        background: pct > 80
            ? `linear-gradient(90deg, ${t.palette.fragmenta.accents.warm}, ${t.palette.perf.channelsDeep[3]})`
            : `linear-gradient(90deg, ${t.palette.fragmenta.accents.cool}, ${t.palette.perf.channelsDeep[0]})`,
        transition: 'width 300ms ease',
    }),
    gpuText: {
        fontSize: '0.7rem',
        color: 'text.secondary',
        fontVariantNumeric: 'tabular-nums',
        whiteSpace: 'nowrap',
    },
    actions: (t) => ({
        display: 'flex',
        gap: '2px',
        p: '2px',
        borderRadius: RADIUS.pill,
        backgroundColor: t.palette.fragmenta.control,
    }),
    button: {
        width: 34,
        height: 34,
        color: 'text.secondary',
        borderRadius: RADIUS.pill,
        '&:hover': { color: 'text.primary', backgroundColor: (t) => t.palette.fragmenta.surface },
    },
    toggle: (on) => (t) => ({
        width: 34,
        height: 34,
        borderRadius: RADIUS.pill,
        color: on ? onFill(t.palette.fragmenta.accents.cool) : t.palette.text.secondary,
        backgroundColor: on ? t.palette.fragmenta.accents.cool : 'transparent',
        '&:hover': { color: on ? onFill(t.palette.fragmenta.accents.cool) : t.palette.text.primary, backgroundColor: on ? t.palette.fragmenta.accents.cool : t.palette.fragmenta.surface },
    }),
    update: (t) => ({
        width: 34,
        height: 34,
        borderRadius: RADIUS.pill,
        color: onFill(t.palette.fragmenta.accents.warm),
        backgroundColor: t.palette.fragmenta.accents.warm,
        '&:hover': { backgroundColor: t.palette.fragmenta.accents.warm, filter: 'brightness(0.97)' },
    }),
};

// Dataset workbench: project header, clip rows, rail cards.
export const datasetStyles = {
    kicker: {
        fontSize: '0.625rem',
        fontWeight: 600,
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        lineHeight: 1.4,
        color: 'text.secondary',
    },
    projectName: {
        fontFamily: FONT_DISPLAY,
        fontSize: '1.75rem',
        fontWeight: 600,
        letterSpacing: '-0.02em',
        lineHeight: 1.15,
        my: 0.25,
    },
    // Clip list: rows with soft rounded backgrounds instead of rule lines.
    clipTable: (t) => ({
        borderCollapse: 'separate',
        borderSpacing: '0 4px',
        '& th': {
            borderBottom: 'none',
            fontSize: '0.625rem',
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: t.palette.text.secondary,
            py: 0.75,
        },
        '& td': { borderBottom: 'none', py: 1.25, verticalAlign: 'middle' },
        '& td:first-of-type': { borderRadius: '18px 0 0 18px' },
        '& td:last-of-type': { borderRadius: '0 18px 18px 0' },
    }),
    clipRow: (isPlaying) => (t) => ({
        '& td': {
            backgroundColor: isPlaying ? t.palette.fragmenta.navSelected : 'transparent',
            transition: 'background-color 160ms ease',
        },
        '&.MuiTableRow-hover:hover td': {
            backgroundColor: isPlaying ? t.palette.fragmenta.navSelected : `rgba(${t.palette.fragmenta.bgRgb}, 0.8)`,
        },
        '&.Mui-selected td, &.Mui-selected:hover td': { backgroundColor: t.palette.fragmenta.control },
    }),
    playButton: (isPlaying) => (t) => ({
        width: 36,
        height: 36,
        flexShrink: 0,
        color: isPlaying ? onFill(t.palette.fragmenta.accents.cool) : t.palette.text.primary,
        backgroundColor: isPlaying ? t.palette.fragmenta.accents.cool : t.palette.fragmenta.onAccent,
        boxShadow: `0 1px 3px rgba(${t.palette.fragmenta.inkRgb}, 0.12)`,
        '&:hover': { backgroundColor: isPlaying ? t.palette.fragmenta.accents.cool : t.palette.fragmenta.surface },
    }),
    fileName: {
        fontSize: '0.75rem',
        fontWeight: 600,
        minWidth: 0,
        overflowWrap: 'anywhere',
        lineHeight: 1.35,
    },
    waveCapsule: (t) => ({
        width: '100%',
        height: 40,
        flexShrink: 0,
        boxSizing: 'border-box',
        px: '9px',
        borderRadius: RADIUS.pill,
        backgroundColor: `rgba(${t.palette.fragmenta.bgRgb}, 0.8)`,
    }),
    // A clip's prompt: tonal; rose-tinted while empty.
    promptField: (empty) => (t) => ({
        '& .MuiOutlinedInput-root': {
            fontSize: '0.75rem',
            lineHeight: 1.45,
            py: 1,
            px: 1.5,
            backgroundColor: empty ? t.palette.perf.laneTints[1] : t.palette.fragmenta.control,
            '&:hover': { backgroundColor: empty ? t.palette.perf.laneTints[1] : t.palette.fragmenta.controlHover },
        },
        '& textarea::placeholder': { color: t.palette.perf.channelsDeep[1], opacity: 1 },
    }),
    rowAction: {
        width: 30,
        height: 30,
        color: 'text.secondary',
        '&:hover': { color: 'text.primary', backgroundColor: (t) => t.palette.fragmenta.control },
    },
    // Auto-annotate this clip: golden, like the other auto actions.
    rowActionWarm: (t) => ({
        width: 30,
        height: 30,
        color: onFill(t.palette.fragmenta.accents.warm),
        backgroundColor: t.palette.fragmenta.accents.warm,
        '&:hover': { backgroundColor: t.palette.fragmenta.warmHi, filter: 'brightness(0.96)' },
        '&.Mui-disabled': { backgroundColor: t.palette.fragmenta.control },
    }),
    // Rail: each tool is its own soft card.
    railCard: (t) => ({
        borderRadius: '22px',
        backgroundColor: t.palette.fragmenta.surface,
        boxShadow: t.palette.fragmenta.shadowSoft,
        overflow: 'hidden',
    }),
    railHeader: {
        width: '100%',
        px: 2.25,
        py: 1.75,
        gap: 1,
        justifyContent: 'flex-start',
        color: 'text.primary',
        '&:hover': { backgroundColor: (t) => `rgba(${t.palette.fragmenta.bgRgb}, 0.5)` },
    },
    railTitle: {
        flex: 1,
        textAlign: 'left',
        fontFamily: FONT_DISPLAY,
        fontSize: '1.0625rem',
        fontWeight: 600,
        letterSpacing: '-0.01em',
    },
};

export const tabPanelStyles = {
    root: {
        px: 0,
        pt: { xs: 1, md: 1.5 },
        pb: { xs: 2, md: 3 },
        background: 'transparent',
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        overflow: 'visible',
    },
};

export const audioUploadRowStyles = {
    // Plain row — no surrounding card background/border. Just spacing
    // between rows so the dropzone and TextField stand on their own.
    card: {
        mb: { xs: 1, sm: 1.25 },
    },
    cardContent: {
        p: 0,
    },
    gridSpacing: { xs: 1.5, sm: 2 },
    uploadDropZone: (isDragActive) => ({
        border: '1.5px dashed',
        borderColor: isDragActive ? 'primary.main' : 'divider',
        borderRadius: 2,
        p: { xs: 1.5, sm: 2 },
        textAlign: 'center',
        cursor: 'pointer',
        bgcolor: isDragActive ? 'action.selected' : 'action.hover',
        transition: 'border-color 160ms ease, background-color 160ms ease',
        '&:hover': {
            borderColor: 'primary.main',
            bgcolor: 'action.selected',
        },
    }),
    hiddenInput: {
        display: 'none',
    },
    audioPreview: {
        width: '100%',
        marginTop: 8,
    },
    deleteGridItem: {
        display: 'flex',
        justifyContent: { xs: 'flex-end', sm: 'center' },
    },
    deleteIconButton: {
        alignSelf: { xs: 'center', sm: 'flex-start' },
    },
};

export const generatedFragmentsWindowStyles = {
    rootPaper: {
        p: 3,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: RADIUS.card,
        // Min keeps the empty state from collapsing the layout.
        minHeight: 240,
        maxHeight: 640,
    },
    headerRow: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        mb: 1.5,
    },
    titleRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        minWidth: 0,
    },
    titleIcon: { display: 'none' },
    titleText: {},  // typography handled by the h6 variant in theme
    countText: (t) => ({
        fontWeight: 600,
        minWidth: 20,
        height: 26,
        px: 1.25,
        display: 'inline-flex',
        alignItems: 'center',
        borderRadius: RADIUS.pill,
        backgroundColor: t.palette.fragmenta.control,
        color: 'text.primary',
        fontVariantNumeric: 'tabular-nums',
    }),
    emptyState: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flex: 1,
        color: 'text.secondary',
    },
    listRoot: {
        flex: 1,
        overflow: 'auto',
        p: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
    },
    // [▶]  prompt · 10 s · 4 min ago        [i] [⬇] [📁] [🗑]
    //      ⟨ capsule waveform ⟩
    listItem: (isPlaying) => (t) => ({
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        p: '10px 12px 10px 10px',
        borderRadius: '20px',
        backgroundColor: isPlaying ? t.palette.fragmenta.navSelected : 'transparent',
        transition: 'background-color 160ms ease',
        '&:hover': { backgroundColor: isPlaying ? t.palette.fragmenta.navSelected : `rgba(${t.palette.fragmenta.bgRgb}, 0.7)` },
    }),
    rowBody: {
        flex: 1,
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
    },
    rowTop: {
        display: 'flex',
        alignItems: 'baseline',
        gap: 1,
        minWidth: 0,
    },
    rowWave: {
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        minWidth: 0,
    },
    fragmentMeta: {
        flex: 1,
        minWidth: 0,
    },
    fragmentPrompt: {
        fontSize: '0.8125rem',
        fontWeight: 500,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
    },
    metaText: {
        flexShrink: 0,
        fontSize: '0.6875rem',
        color: 'text.secondary',
        fontVariantNumeric: 'tabular-nums',
        whiteSpace: 'nowrap',
    },
    batchTag: (t) => ({
        display: 'inline-block',
        fontWeight: 700,
        fontSize: '0.7rem',
        px: 0.75,
        mr: 0.75,
        borderRadius: RADIUS.pill,
        bgcolor: t.palette.fragmenta.control,
        color: 'text.secondary',
        fontVariantNumeric: 'tabular-nums',
    }),
    fragmentMetaInline: {
        flexShrink: 0,
        whiteSpace: 'nowrap',
        fontVariantNumeric: 'tabular-nums',
    },
    fragmentInfoIcon: {
        display: 'inline-flex',
        alignItems: 'center',
        color: 'text.secondary',
        opacity: 0.6,
        cursor: 'help',
        transition: 'opacity 150ms ease',
        '&:hover': { opacity: 1 },
    },
    // Play: a 40 px circle, ivory with a soft shadow; the cool pastel while playing.
    playPauseButton: (isPlaying) => (t) => ({
        flexShrink: 0,
        width: 40,
        height: 40,
        color: isPlaying ? onFill(t.palette.fragmenta.accents.cool) : t.palette.text.primary,
        backgroundColor: isPlaying ? t.palette.fragmenta.accents.cool : t.palette.fragmenta.onAccent,
        boxShadow: `0 1px 3px rgba(${t.palette.fragmenta.inkRgb}, 0.12)`,
        '&:hover': { backgroundColor: isPlaying ? t.palette.fragmenta.accents.cool : t.palette.fragmenta.surface },
    }),
    downloadButton: {
        flexShrink: 0,
        color: 'text.secondary',
        '&:hover': { color: 'text.primary' },
    },
    hiddenAudio: {
        display: 'none',
    },
};

export const trainingMonitorStyles = {
    rootPaper: {
        p: 3,
        mb: 2,
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: 2.25,
        borderRadius: RADIUS.card,
    },
    headerRow: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 1,
    },
    headerTitleWrap: {
        display: 'flex',
        alignItems: 'center',
        gap: 1,
    },
    headerIcon: { display: 'none' },
    headerTitle: {},  // typography handled by the h6 variant in theme
    // Status pill: tinted by state (training green, done blue, error red).
    statusInline: (status) => (t) => {
      const bg = status === 'live'
          ? t.palette.fragmenta.accents.fresh
          : status === 'complete'
              ? t.palette.fragmenta.accents.cool
              : status === 'error'
                  ? t.palette.fragmenta.accents.hot
                  : t.palette.fragmenta.control;
      return {
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        height: 30,
        px: 1.5,
        borderRadius: RADIUS.pill,
        backgroundColor: bg,
        color: onFill(bg),
        '& .MuiTypography-root': { color: 'inherit' },
      };
    },
    statusDot: (status, animate = false) => ({
        width: 8,
        height: 8,
        borderRadius: '50%',
        bgcolor: 'currentColor',
        opacity: status === 'idle' ? 0.45 : 1,
        animation: animate ? 'fragmenta-status-pulse 2s infinite' : 'none',
        '@keyframes fragmenta-status-pulse': {
            '0%': { opacity: 1 },
            '50%': { opacity: 0.35 },
            '100%': { opacity: 1 },
        },
    }),
    statusText: () => ({
        color: 'text.primary',
        fontWeight: 600,
        fontSize: '0.75rem',
        lineHeight: 1,
    }),
    progressSection: {
        display: 'flex',
        flexDirection: 'column',
        gap: 1.25,
    },
    progressHeader: {
        display: 'flex',
        alignItems: 'baseline',
        flexWrap: 'wrap',
        gap: 1.5,
    },
    bigPercent: {
        fontFamily: FONT_DISPLAY,
        fontSize: '3rem',
        fontWeight: 600,
        letterSpacing: '-0.03em',
        lineHeight: 1,
        fontVariantNumeric: 'tabular-nums',
    },
    stepText: {
        fontSize: '0.8125rem',
        color: 'text.secondary',
        fontVariantNumeric: 'tabular-nums',
    },
    progressBar: (t) => ({
        height: 10,
        borderRadius: RADIUS.pill,
        backgroundColor: t.palette.fragmenta.control,
        boxShadow: `inset 0 1px 2px rgba(${t.palette.fragmenta.inkRgb}, 0.12)`,
        '& .MuiLinearProgress-bar': {
            borderRadius: RADIUS.pill,
            backgroundImage: `linear-gradient(90deg, ${t.palette.perf.channelsSoft[3]}, ${t.palette.fragmenta.accents.warm} 60%, ${t.palette.perf.channelsDeep[3]})`,
        },
    }),
    deviceSection: {
        mb: 2,
    },
    deviceInfo: {
        fontSize: { xs: '0.74rem', sm: '0.8rem' },
        mt: 0.5,
    },
    // Four tonal tiles: Loss, Checkpoints, Phase, Seed.
    metricsGrid: {
        display: 'grid',
        gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', sm: 'repeat(4, minmax(0, 1fr))' },
        gap: 1,
    },
    metricTile: (t) => ({
        display: 'flex',
        flexDirection: 'column',
        gap: 0.5,
        p: 1.5,
        borderRadius: '16px',
        backgroundColor: t.palette.fragmenta.section,
    }),
    metricLabel: {
        fontSize: '0.625rem',
        fontWeight: 600,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        color: 'text.secondary',
    },
    metricValue: {
        fontFamily: FONT_DISPLAY,
        fontSize: '1.25rem',
        fontWeight: 600,
        fontVariantNumeric: 'tabular-nums',
        lineHeight: 1.2,
    },
    lossSection: (t) => ({
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
        p: 2,
        pb: 1.25,
        borderRadius: '20px',
        backgroundColor: t.palette.fragmenta.section,
    }),
    lossChartBox: {
        height: 200,
        width: '100%',
    },
    lossDisclaimer: {
        display: 'block',
        lineHeight: 1.4,
        color: 'text.secondary',
    },
    errorAlert: {
        mb: 2,
    },
};

export const welcomePageStyles = {
    backdrop: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        const L = muiTheme.palette.fragmenta;
        // Match the main app body's three-layer treatment so the
        // welcome page reads as part of the same surface, not a
        // separate dialog with its own palette.
        return {
            zIndex: 9999,
            background: isDark
                ? `radial-gradient(900px 700px at -5% 50%, rgba(39, 159, 187, 0.18), transparent 60%), ` +
                  `radial-gradient(1100px 700px at 95% 108%, rgba(253, 162, 43, 0.14), transparent 55%), ` +
                  `linear-gradient(165deg, #181A1B 0%, ${DARK.bg} 42%, #1A1B1C 100%)`
                : `radial-gradient(900px 700px at -5% 50%, rgba(${L.amberRgb}, 0.09), transparent 60%), ` +
                  `radial-gradient(1100px 700px at 95% 108%, rgba(${L.warmRgb}, 0.08), transparent 55%), ` +
                  `linear-gradient(165deg, ${L.bgTop} 0%, ${L.bg} 42%, ${L.bgBottom} 100%)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            p: { xs: 2, md: 4 },
            cursor: 'pointer',
        };
    },
    panel: () => {
        // Transparent card so the backdrop gradient shows through;
        // only the inner content (logo, title, button) is visible.
        return {
            textAlign: 'center',
            width: 'min(920px, 100%)',
            border: 'none',
            borderRadius: 4,
            background: 'transparent',
            boxShadow: 'none',
            backdropFilter: 'none',
            px: { xs: 3, md: 7 },
            py: { xs: 5, md: 6 },
        };
    },
    logo: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            width: { xs: 96, sm: 122 },
            height: { xs: 96, sm: 122 },
            backgroundImage: 'url(/fragmenta_icon_1024.png)',
            backgroundSize: 'contain',
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'center',
            // No border, no surface — only the icon and its drop shadow.
            backgroundColor: 'transparent',
            border: 'none',
            boxShadow: 'none',
            filter: isDark
                ? 'drop-shadow(0 8px 16px rgba(0, 0, 0, 0.4))'
                : 'drop-shadow(0 6px 12px rgba(43, 31, 18, 0.18))',
            mx: 'auto',
            mb: { xs: 3.5, sm: 5 },
        };
    },
    title: {
        fontFamily: FONT_DISPLAY,
        fontWeight: 600,
        color: 'text.primary',
        mb: 1,
        fontSize: { xs: '2.5rem', sm: '3.25rem', md: '3.75rem' },
        letterSpacing: '-0.03em',
        fontVariationSettings: '"opsz" 96',
    },
};

// --- Demo notice (Hugging Face Space only) ----------------------------------
// Same vocabulary as AboutDialog: centred logo + display title inside the
// glass DialogTitle, quiet body copy, one accent action. The only extra note
// is the warm "Demo Space" badge — the app already uses `warm` to mean
// "temporary / in progress state", so it reads as a status, not as an alert.
export const demoNoticeStyles = {
    titleStack: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 1,
    },
    badge: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        // The warm accent as rgba, so the badge tints the glass beneath it
        // instead of sitting on an opaque chip.
        const warmRgb = isDark ? '253, 162, 43' : muiTheme.palette.fragmenta.warmRgb;
        return {
            mt: 0.25,
            px: 1.1,
            py: 0.3,
            borderRadius: 999,
            border: `1px solid rgba(${warmRgb}, 0.45)`,
            backgroundColor: `rgba(${warmRgb}, 0.12)`,
            color: isDark ? DARK.warm : muiTheme.palette.fragmenta.warm,
            fontFamily: FONT_MONO,
            fontSize: '0.62rem',
            fontWeight: 500,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            lineHeight: 1.6,
        };
    },
    body: {
        mt: 2.5,
        mb: 1.25,
        color: 'text.primary',
        textAlign: 'center',
        fontSize: '0.9rem',
        lineHeight: 1.6,
    },
    subBody: {
        mb: 2,
        color: 'text.secondary',
        textAlign: 'center',
        fontSize: '0.8rem',
    },
    linkButton: {
        fontFamily: FONT_MONO,
        fontSize: '0.78rem',
        letterSpacing: '0.01em',
        textTransform: 'none',
    },
    actions: {
        justifyContent: 'center',
    },
};

export const lossChartStyles = {
    padding: { top: 10, right: 16, bottom: 28, left: 44 },
    svg: {
        width: '100%',
        height: '100%',
    },
    axisFontSize: 10,
    tooltip: {
        width: 100,
        height: 34,
        rx: 4,
        textX: 6,
        timeY: 14,
        lossY: 28,
    },
};

// Shared visual tokens for the Performance page (panel, channels, MIDI menu).
// One source of truth for size / spacing / height across the Performance
// surface. Strict 3-size type ladder + case rules — every text choice has
// to fit a known cell, no more 4 sizes × 4 case styles soup.
//
// CASE RULES (enforced in code review):
//   • ALL-CAPS    only for scientific / control nomenclature: knob labels
//                 (GAIN/LPF/DLY/REV), PAN, dB units, BPM, MIDI, DBFS.
//                 Always paired with `caps` mixin (tracking + weight).
//   • Sentence    every interactive button text, dropdown text, section
//                 title, value readout. e.g. "Play all", "1 bar", "Save".
//   • lowercase   banned. No more "prompt…" / "empty" / "installing…".
//
// SIZE LADDER (3 sizes, no more):
export const perfTokens = {
    fontSize: {
        // 0.62rem — smallest sentence-case body text (LoRA subtitle, dropdown
        // section dividers, mapping monospace, footnotes, masterPeakValue).
        xs: '0.62rem',
        // 0.72rem — body: buttons, dropdowns, prompt, value readouts, take
        // history entries, sentence-case labels, mapping rows.
        sm: '0.72rem',
        // 0.84rem — section badges (channel "01", "Master", tab headers).
        md: '0.84rem',
    },
    letterSpacing: {
        // ALL-CAPS labels get extra tracking so they read as labels not text.
        wide: '0.08em',
        // Mild tracking on numeric readouts (dB, BPM display) for clarity.
        snug: '0.04em',
    },
    weight: {
        regular: 500,
        bold: 600,
        heavy: 700,    // single-letter glyph buttons (M, S, L)
    },
    height: {
        iconSm: 18,    // smallest tap targets (take-history clear)
        iconMd: 20,    // take-row icon buttons (audition, star, delete, commit)
        sub: 28,       // small square toggles (mute, solo, sidechain)
        compact: 34,   // bar controls (Link, MIDI, Q, BPM, selects)
        cta: 40,       // Generate
    },
    // Icon glyph sizes paired with their host control. Inside a 26px button
    // use `md`; inside 18–22px buttons use `sm`; inside the channel transport
    // play/stop (26px but the glyph reads as the primary action) use `lg`.
    icon: {
        sm: 12,
        md: 14,
        lg: 16,
    },
    // Radius ladder. `sharp` (2px) is the deliberate Ableton-style accent on
    // Link/MIDI rectangles. Everything else uses the MUI scale (1 / 1.5 / 2.5).
    radius: {
        sharp: '2px',
        sm: 1,
        md: 1.5,
        lg: 2.5,
    },
    // Composable mixin for any ALL-CAPS label/badge — applies size, case,
    // weight, and tracking in one go. Use directly in sx:
    //   sx={{ ...perfTokens.caps }}
    // Scope: scientific-control labels only (GAIN, LPF, PAN, BPM, DBFS, etc.)
    // Sits one notch under the xs body ladder (0.58rem vs 0.62rem) so caps
    // text reads as a label / annotation, not as body content.
    caps: {
        fontSize: '0.58rem',
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        fontWeight: 600,
    },
    // Mixin for sentence-case sub-labels next to a control widget
    // (Steps, Seed, Auto BPM, Main out, Cue out). Muted color + snug tracking,
    // body-size text — distinct from `caps` (scientific) and plain body text.
    labelMuted: {
        fontSize: '0.72rem',
        color: 'text.disabled',
        letterSpacing: '0.04em',
    },
    // Mixin for numeric value readouts (e.g. dB, BPM, durations).
    num: {
        fontSize: '0.72rem',
        fontVariantNumeric: 'tabular-nums',
        letterSpacing: '0.04em',
    },
};

export const performancePanelStyles = {
    root: {
        display: 'flex',
        flexDirection: 'column',
        gap: 1.5,
        width: '100%',
        minHeight: 0,
    },
    // Stateless icon-button shape. Pairs an outer size (xs/sm/md/lg → 18/20/22/26)
    // with one of two intents: 'menu' (resting color text.secondary, hover
    // text.primary — used for buttons that open menus) or 'danger' (resting
    // text.disabled, hover error.main — used for deletes). Stateful color-bound
    // toggles (fragmentIconBtn, transportBtn, loopBtn, mute/solo) stay parameterized
    // — they're not just icon buttons.
    compactIconBtn: (size = 'md', variant = 'menu') => {
        const sz = { xs: 18, sm: 20, md: 22, lg: 34 }[size];
        return {
            width: sz,
            height: sz,
            borderRadius: RADIUS.pill,
            color: variant === 'danger' ? 'text.disabled' : 'text.secondary',
            '&:hover': {
                color: variant === 'danger' ? 'error.main' : 'text.primary',
                bgcolor: 'action.hover',
            },
        };
    },
    // Shared shell for the top transport bar and the bottom generation bar.
    // Holds everything visually unifying (border, radius, gradient, alignment)
    // — gap and mt stay inline so the dense transport row (gap 1) can stay
    // compact while the grouped generation row (gap 2.5) breathes.
    barCard: {
        display: 'flex',
        alignItems: 'center',
        px: 1.5,
        py: 0.75,
        borderRadius: RADIUS.panel,
        backgroundColor: (t) => t.palette.fragmenta.surface,
        boxShadow: (t) => t.palette.fragmenta.shadowSoft,
        flexWrap: { xs: 'wrap', md: 'nowrap' },
    },
    // Single source of truth for every "pill" control in the master + bottom
    // bar (Q select, BPM input, Steps select, Seed input, Model picker, LoRA
    // picker, etc.). Spread this into the control's `sx` to lock height,
    // radius, font size, padding, and ellipsis behavior — no per-instance
    // overrides needed.
    pillControl: {
        // Force fontFamily inheritance — the global MuiSelect theme override
        // sets FONT_DISPLAY ("Bricolage Grotesque") which would otherwise
        // make the Q / Model / Steps Select triggers render in a different
        // font than the surrounding ButtonBase / TextField text on the bar
        // (which inherits FONT_BODY from the body). Inheriting at the slot
        // level wins over the theme override and keeps every pill matched.
        // Tonal control surface, no outline.
        '& .MuiOutlinedInput-root': {
            fontFamily: 'inherit',
            height: perfTokens.height.compact,
            borderRadius: RADIUS.control,
            backgroundColor: (t) => t.palette.fragmenta.control,
            transition: 'background-color 160ms ease',
            '&:hover': { backgroundColor: (t) => t.palette.fragmenta.controlHover },
        },
        '& fieldset': { border: 'none' },
        '& .MuiSelect-select': {
            fontFamily: 'inherit',
            // Explicit padding on all four sides — top/bottom 0 keeps the
            // text in the 26px row, RIGHT must reserve space for the
            // chevron (MUI's chevron is absolutely positioned ~7px from
            // right with ~24px width = needs 32px clearance). MUI ships
            // padding-right: 32px !important by default but our partial
            // override (paddingTop/Bottom only) wasn't preserving the
            // shorthand correctly across all themes — explicit is safer.
            // Line-height matched to parent height vertical-centers the
            // text without `display: flex` (which would kill
            // text-overflow: ellipsis).
            padding: '0 32px 0 14px !important',
            lineHeight: `${perfTokens.height.compact}px`,
            fontSize: perfTokens.fontSize.sm,
            fontWeight: perfTokens.weight.bold,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
        },
        // The TextField input needs the same font on the rendered <input>; we
        // put it on the wrapper here, but anywhere a TextField is used, also
        // set `inputProps={{ style: { fontSize: perfTokens.fontSize.sm } }}`
        // to win against MUI's .MuiInputBase-inputSizeSmall 14px default.
        '& .MuiOutlinedInput-input': {
            fontFamily: 'inherit',
            fontSize: perfTokens.fontSize.sm,
            fontWeight: perfTokens.weight.bold,
        },
    },
    headerCard: (theme) => ({
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        p: { xs: 1.25, sm: 1.75 },
        borderRadius: 2.5,
        border: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.paper',
        backgroundImage:
            `linear-gradient(135deg, ${theme.palette.primary.main}14 0%, ${theme.palette.background.paper} 100%)`,
        flexWrap: { xs: 'wrap', md: 'nowrap' },
    }),
    headerLeft: {
        display: 'flex',
        flexDirection: 'column',
        gap: 0.25,
        minWidth: 200,
        flex: '0 0 auto',
    },
    titleRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        color: 'primary.main',
    },
    title: {
        fontWeight: 400,
        letterSpacing: '0.02em',
    },
    subtitle: {
        color: 'text.secondary',
        fontSize: perfTokens.fontSize.sm,
    },
    headerPickers: {
        display: 'flex',
        gap: 1,
        flex: 1,
        minWidth: 0,
        flexWrap: { xs: 'wrap', sm: 'nowrap' },
    },
    headerModelPicker: {
        flex: 1,
        minWidth: 200,
        '& .MuiOutlinedInput-root': { borderRadius: 2 },
    },
    headerCheckpointPicker: {
        flex: 1,
        minWidth: 180,
        '& .MuiOutlinedInput-root': { borderRadius: 2 },
    },
    errorAlert: {
        borderRadius: 2,
    },
    channelsRow: {
        display: 'flex',
        gap: 1.5,
        alignItems: 'stretch',
        flexWrap: { xs: 'wrap', md: 'nowrap' },
    },
    channelsGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(4, minmax(200px, 1fr))',
        gap: 1.25,
        flex: 1,
        minWidth: 0,
    },
    // Master: a compact, lightly elevated panel.
    masterStrip: () => (theme) => ({
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2,
        px: 2,
        py: 2.5,
        borderRadius: RADIUS.card,
        backgroundColor: theme.palette.fragmenta.surface,
        boxShadow: theme.palette.fragmenta.shadowRaised,
        width: { xs: '100%', md: 176 },
        flex: { xs: '1 1 100%', md: '0 0 176px' },
        minHeight: 0,
        overflow: 'hidden',
        boxSizing: 'border-box',
    }),
    masterHeader: () => ({
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 20,
    }),
    masterBadge: () => ({
        ...perfTokens.caps,
        fontSize: '0.625rem',
        letterSpacing: '0.1em',
        color: 'text.secondary',
    }),
    masterReadouts: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '2px',
        mt: -1,
    },
    masterValue: {
        fontFamily: FONT_DISPLAY,
        fontSize: '1.25rem',
        fontWeight: 600,
        fontVariantNumeric: 'tabular-nums',
        color: 'text.primary',
        lineHeight: 1.1,
    },
    masterCaption: {
        ...perfTokens.caps,
        fontSize: '0.625rem',
        color: 'text.disabled',
    },
    // Tall, slim, rounded L/R meters with a dB scale beside them.
    masterMeters: {
        display: 'flex',
        alignItems: 'flex-start',
        gap: '10px',
        height: 222,
    },
    masterScale: {
        position: 'relative',
        width: 22,
        height: 200,
    },
    masterScaleLabel: (bottomPct) => ({
        position: 'absolute',
        right: 0,
        bottom: `calc(${bottomPct}% - 6px)`,
        fontSize: '0.625rem',
        lineHeight: '12px',
        color: 'text.disabled',
        fontVariantNumeric: 'tabular-nums',
    }),
    masterMeterCol: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 1,
    },
    masterMeterTrack: (theme) => ({
        position: 'relative',
        width: 12,
        height: 200,
        borderRadius: RADIUS.pill,
        backgroundColor: theme.palette.fragmenta.control,
        boxShadow: `inset 0 1px 2px rgba(${theme.palette.fragmenta.inkRgb}, 0.14)`,
        overflow: 'hidden',
    }),
    // Full height; the meter loop reveals it with a clip-path, so the warm
    // top only shows when the level gets there.
    masterMeterFill: (color) => (theme) => ({
        position: 'absolute',
        inset: 0,
        clipPath: 'inset(100% 0 0 0)',
        borderRadius: RADIUS.pill,
        background: `linear-gradient(0deg, ${theme.palette.primary.main} 0%, ${color} 55%, ${theme.palette.fragmenta.accents.warm} 82%, ${theme.palette.fragmenta.accents.hot} 100%)`,
    }),
    // Peak hold: a short ink line that holds the loudest moment briefly.
    masterPeakMarker: {
        position: 'absolute',
        left: 1,
        right: 1,
        bottom: '0%',
        height: 2,
        borderRadius: RADIUS.pill,
        backgroundColor: 'text.primary',
        opacity: 0,
        pointerEvents: 'none',
    },
    masterMeterLabel: {
        fontSize: '0.625rem',
        fontWeight: 600,
        color: 'text.disabled',
    },
    masterPeakValue: {
        fontSize: '0.6875rem',
        color: 'text.secondary',
        fontVariantNumeric: 'tabular-nums',
    },
    masterTransport: {
        display: 'flex',
        flexDirection: 'column',
        gap: 0.5,
        pt: 0.75,
    },
    masterBtn: (color, variant) => (theme) => {
        const fill = variant === 'stop' ? theme.palette.fragmenta.accents.hot : color;
        return {
            textTransform: 'none',
            borderRadius: RADIUS.control,
            fontSize: perfTokens.fontSize.sm,
            fontWeight: 500,
            height: perfTokens.height.compact,
            py: 0,
            px: 1.25,
            minWidth: 0,
            lineHeight: 1,
            border: 'none',
            color: onFill(fill),
            backgroundColor: fill,
            transition: 'background-color 160ms ease, filter 160ms ease',
            '&:hover': { border: 'none', backgroundColor: fill, filter: 'brightness(0.97)' },
            '&.Mui-disabled': {
                border: 'none',
                color: `rgba(${theme.palette.fragmenta.inkRgb}, 0.5)`,
                backgroundColor: theme.palette.fragmenta.control,
            },
        };
    },
    // Transport (Play / Stop / Record): circular, filled with their colour.
    masterIconBtn: (color) => (theme) => ({
        width: 42,
        height: 42,
        borderRadius: RADIUS.pill,
        color: onFill(color),
        backgroundColor: color,
        transition: 'background-color 160ms ease, filter 160ms ease, transform 160ms ease',
        '&:hover': { color: onFill(color), backgroundColor: color, filter: 'brightness(0.96)' },
        '&:active': { transform: 'scale(0.95)' },
        '&.Mui-focusVisible': {
            color: onFill(color),
            backgroundColor: color,
            outline: 'none',
            boxShadow: `0 0 0 2px ${onFill(color)}55`,
        },
        '&.Mui-disabled': {
            color: `rgba(${theme.palette.fragmenta.inkRgb}, 0.35)`,
            backgroundColor: theme.palette.fragmenta.control,
        },
    }),
};

export const performanceChannelStyles = {
    // Lane card: a pale tint of the lane colour on the ivory canvas, softly
    // rounded, no border or glass. Playing deepens the tint (and fills the
    // lane indicator), so the state reads without relying on motion.
    strip: (index, playing) => (theme) => {
        const perf = theme.palette.perf;
        const tints = playing ? perf.laneTintsPlaying : perf.laneTints;
        return {
            display: 'flex',
            flexDirection: 'column',
            gap: '14px',
            p: 2,
            borderRadius: RADIUS.card,
            backgroundColor: tints[index % tints.length],
            boxShadow: playing ? theme.palette.fragmenta.shadowRaised : theme.palette.fragmenta.shadowSoft,
            transition: 'background-color 220ms ease, box-shadow 220ms ease',
            height: '100%',
            minWidth: 150,
        };
    },
    stripHeader: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        minHeight: perfTokens.height.compact,
    },
    // Circular lane number in the lane colour; solid ink ring while playing.
    laneIndicator: (color, playing) => (theme) => ({
        width: 28,
        height: 28,
        flexShrink: 0,
        borderRadius: RADIUS.pill,
        display: 'grid',
        placeItems: 'center',
        fontSize: '0.75rem',
        fontWeight: perfTokens.weight.heavy,
        fontVariantNumeric: 'tabular-nums',
        color: onFill(color),
        backgroundColor: color,
        boxShadow: playing ? `0 0 0 2px ${theme.palette.perf.label}` : 'none',
        transition: 'box-shadow 160ms ease',
        userSelect: 'none',
    }),
    muteSoloRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 1,
    },
    // M / S / SC: compact rounded rectangles on the control tone; their
    // active fills come from the palette's state colours.
    muteBtn: (active) => (theme) => {
        const c = theme.palette.fragmenta.accents.hot;
        return smallToggle(theme, active, c);
    },
    soloBtn: (active) => (theme) => smallToggle(theme, active, theme.palette.fragmenta.accents.warm),
    sidechainBtn: (active) => (theme) => ({
        ...smallToggle(theme, active, theme.palette.fragmenta.accents.fresh),
        fontSize: perfTokens.fontSize.xs,
        '&.Mui-disabled': { color: 'text.disabled', backgroundColor: 'transparent', opacity: 0.45 },
    }),
    promptBox: {
        display: 'flex',
        flexDirection: 'column',
        gap: 0.75,
    },
    // Prompt sits directly on a slightly darker tone; no outline.
    promptField: (theme) => ({
        '& .MuiOutlinedInput-root': {
            fontSize: '0.8125rem',
            padding: '11px 14px',
            backgroundColor: ON_TINT,
            borderRadius: '16px',
            transition: 'background-color 160ms ease',
            '&:hover': { backgroundColor: ON_TINT_HOVER },
            '&.Mui-focused': { backgroundColor: theme.palette.fragmenta.surface },
            '& textarea': {
                lineHeight: 1.4,
                overflowY: 'auto !important',
            },
        },
        '& fieldset': { border: 'none' },
    }),
    durationRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 0.75,
    },
    durationLabel: {
        fontFamily: 'inherit',
        fontSize: perfTokens.fontSize.sm,
        color: 'text.secondary',
        minWidth: 22,
    },
    // `&&` outranks the theme's colorPrimary override, so the lane colour wins.
    durationSlider: (color) => ({
        flex: 1,
        '&&': { color },
    }),
    // Sec / Bars: a segmented control on the control tone; the selected
    // segment is the one place a fully rounded end is used.
    segmented: {
        display: 'inline-flex',
        alignItems: 'center',
        p: '3px',
        gap: '2px',
        height: 34,
        boxSizing: 'border-box',
        flexShrink: 0,
        borderRadius: RADIUS.control,
        backgroundColor: ON_TINT,
    },
    segment: (color, active) => (theme) => ({
        fontSize: '0.75rem',
        px: 1.4,
        height: 28,
        borderRadius: RADIUS.pill,
        bgcolor: active ? color : 'transparent',
        color: active ? onFill(color) : 'text.secondary',
        fontWeight: active ? perfTokens.weight.bold : perfTokens.weight.regular,
        transition: 'background-color 160ms ease, color 160ms ease',
        '&:hover': {
            bgcolor: active ? color : ON_TINT_HOVER,
            color: active ? onFill(color) : 'text.primary',
        },
    }),
    // The lane's length select, on the tint.
    laneSelect: {
        flex: 1,
        '& .MuiOutlinedInput-root': {
            height: 34,
            borderRadius: '10px',
            backgroundColor: ON_TINT,
            '&:hover': { backgroundColor: ON_TINT_HOVER },
        },
        '& .MuiSelect-select': { lineHeight: '34px', fontSize: '0.75rem' },
    },
    // The ×N batch select next to Generate: tonal, no outline.
    channelPillControl: (theme) => ({
        '& .MuiOutlinedInput-root': {
            fontFamily: 'inherit',
            height: 40,
            borderRadius: RADIUS.control,
            backgroundColor: ON_TINT,
            '&:hover': { backgroundColor: ON_TINT_HOVER },
        },
        '& fieldset': { border: 'none' },
        '& .MuiSelect-select': {
            fontFamily: 'inherit',
            padding: '0 26px 0 10px !important',
            lineHeight: '40px',
            fontSize: perfTokens.fontSize.sm,
            fontWeight: perfTokens.weight.bold,
            fontVariantNumeric: 'tabular-nums',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
        },
    }),
    // Generate: the lane's primary action. A filled rounded rectangle in the
    // lane colour; it gives a brief expansion when generation starts, and a
    // progress fill runs inside while it works.
    generatePill: (color, { generating, disabled }) => (theme) => {
        const base = {
            position: 'relative',
            overflow: 'hidden',
            boxSizing: 'border-box',
            width: '100%',
            minWidth: 96,
            height: 40,
            px: 1.5,
            borderRadius: '14px',
            border: 'none',
            fontSize: '0.8125rem',
            fontWeight: 600,
            textTransform: 'none',
            cursor: disabled ? 'not-allowed' : 'pointer',
            transition: 'background-color 160ms ease, color 160ms ease, transform 160ms ease',
        };
        if (disabled) {
            return {
                ...base,
                backgroundColor: ON_TINT,
                color: `rgba(${theme.palette.fragmenta.inkRgb}, 0.45)`,
            };
        }
        if (generating) {
            return {
                ...base,
                backgroundColor: ON_TINT,
                color: theme.palette.perf.label,
                animation: 'fragmenta-generate-pop 260ms cubic-bezier(0.16, 1, 0.3, 1)',
            };
        }
        return {
            ...base,
            backgroundColor: color,
            color: onFill(color),
            boxShadow: `0 1px 2px rgba(${theme.palette.fragmenta.inkRgb}, 0.10)`,
            '&:hover': { filter: 'brightness(0.97)' },
            '&:active': { transform: 'scale(0.98)' },
        };
    },
    generatePillFill: (color, progressPct) => ({
        position: 'absolute',
        left: 0,
        top: 0,
        bottom: 0,
        width: `${Math.max(0, Math.min(100, progressPct))}%`,
        backgroundColor: color,
        borderRadius: '14px',
        transition: 'width 250ms ease-out',
        pointerEvents: 'none',
        zIndex: 0,
    }),
    generatePillLabel: {
        position: 'relative',
        zIndex: 1,
        display: 'inline-flex',
        alignItems: 'center',
        gap: 0.5,
        fontVariantNumeric: 'tabular-nums',
        lineHeight: 1,
    },
    // Waveform: open and borderless on a faint veil, softly rounded. Not
    // clipped, so the loop brackets can sit on the very edges.
    // Waveform: a tall capsule. The waveform and its overlays sit in an
    // inner box inset from the round ends (waveformInner), so nothing is
    // clipped by the caps and the trim maths stays in one box.
    waveformWrap: {
        position: 'relative',
        height: 96,
        borderRadius: RADIUS.pill,
        backgroundColor: 'rgba(255, 255, 255, 0.42)',
        px: '18px',
        boxSizing: 'border-box',
    },
    waveformInner: {
        position: 'relative',
        height: '100%',
    },
    // Loop in/out handle: a slim pill inset from the capsule's caps, with a
    // wider invisible hit area.
    trimHandle: (pos, deep) => ({
        position: 'absolute',
        top: 0,
        bottom: 0,
        left: `calc(${pos * 100}% - 7px)`,
        width: 14,
        cursor: 'ew-resize',
        touchAction: 'none',
        zIndex: 2,
        '&::before': {
            content: '""',
            position: 'absolute',
            top: 20,
            bottom: 20,
            left: 5,
            width: 4,
            borderRadius: RADIUS.pill,
            backgroundColor: deep,
        },
    }),
    playhead: (t) => ({
        position: 'absolute',
        top: 12,
        bottom: 12,
        left: 0,
        // '2px', not 2: sx reads numbers <= 1 as a percentage.
        width: '2px',
        ml: '-1px',
        borderRadius: RADIUS.pill,
        backgroundColor: t.palette.perf.label,
        opacity: 0,
        pointerEvents: 'none',
        zIndex: 4,
        '&::before': {
            content: '""',
            position: 'absolute',
            top: -6,
            left: -4,
            width: 10,
            height: 10,
            borderRadius: RADIUS.pill,
            backgroundColor: t.palette.perf.label,
        },
    }),
    // Outside the loop: the canvas tone laid over the waveform, so those
    // parts fade rather than go dark.
    trimVeil: {
        backgroundColor: (t) => `rgba(${t.palette.fragmenta.bgRgb}, 0.72)`,
        pointerEvents: 'none',
    },
    // While a lane generates, a soft band of its colour sweeps across.
    waveformGenerating: (color) => ({
        '&::after': {
            content: '""',
            position: 'absolute',
            inset: 0,
            borderRadius: RADIUS.pill,
            pointerEvents: 'none',
            background: `linear-gradient(90deg, transparent 0%, ${color}99 50%, transparent 100%)`,
            backgroundSize: '50% 100%',
            backgroundRepeat: 'no-repeat',
            animation: 'fragmenta-wave-sweep 1.6s ease-in-out infinite',
        },
    }),
    // Fragment history: no box or rule lines, just a small label and rows;
    // the loaded fragment gets a soft tint.
    // Four rows tall at most; the rest scroll, so lanes stay one height.
    fragmentHistoryPanel: {
        overflow: 'hidden',
        flex: '0 0 auto',
        display: 'flex',
        flexDirection: 'column',
    },
    fragmentHistoryHeader: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        px: 0.5,
        minHeight: 20,
    },
    fragmentHistoryHeaderText: {
        ...perfTokens.caps,
        color: 'text.secondary',
    },
    fragmentHistoryHeaderBtn: {
        width: 18,
        height: 18,
        borderRadius: RADIUS.small,
        color: 'text.disabled',
        '&:hover': {
            color: 'error.main',
            bgcolor: (t) => t.palette.fragmenta.control,
        },
    },
    // Rows are 36 px + 2 px gap; four show before the list scrolls, so the
    // lane never resizes as fragments accumulate.
    fragmentHistoryList: {
        height: 4 * 38,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '2px',
    },
    fragmentHistoryEmpty: {
        ...perfTokens.caps,
        textAlign: 'center',
        color: 'text.disabled',
        height: 4 * 38,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '14px',
        backgroundColor: 'rgba(255, 255, 255, 0.3)',
    },
    fragmentRow: (soft, isCommitted, isAuditioning) => () => ({
        display: 'flex',
        alignItems: 'center',
        gap: 0.75,
        pl: '5px',
        pr: 1.25,
        minHeight: 36,
        borderRadius: '14px',
        bgcolor: isCommitted || isAuditioning ? soft : 'transparent',
        '&:hover': { bgcolor: isCommitted || isAuditioning ? soft : ON_TINT },
        '& .frag-reorder': { opacity: 0, transition: 'opacity 120ms ease' },
        '&:hover .frag-reorder, &:focus-within .frag-reorder': { opacity: 1 },
        transition: 'background-color 160ms ease',
    }),
    fragmentReorderCol: {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        flexShrink: 0,
        width: 14,
        lineHeight: 0,
        '& > span': { display: 'flex' },
    },
    fragmentReorderBtn: {
        width: 14,
        height: 10,
        minWidth: 0,
        p: 0,
        borderRadius: 0.5,
        color: 'text.disabled',
        '&:hover': { color: 'text.primary', bgcolor: 'action.hover' },
        '&.Mui-disabled': { color: 'text.disabled', opacity: 0.25 },
    },
    // Row buttons: audition is a small circle (the play indicator), commit
    // and star are compact squares. `filled` = the solid active state.
    fragmentIconBtn: (color, active, filled = false) => (theme) => ({
        width: 20,
        height: 20,
        borderRadius: RADIUS.pill,
        color: active ? theme.palette.perf.label : 'text.secondary',
        backgroundColor: filled && active ? theme.palette.fragmenta.surface : 'transparent',
        boxShadow: filled && active ? theme.palette.fragmenta.shadowSoft : 'none',
        transition: 'background-color 160ms ease, color 160ms ease',
        '&:hover': { color: 'text.primary', bgcolor: theme.palette.fragmenta.surface },
        '&.Mui-disabled': filled && active
            ? { color: theme.palette.perf.label, backgroundColor: theme.palette.fragmenta.surface, opacity: 1 }
            : { color: 'text.secondary', opacity: 0.85 },
    }),
    // Audition: the row's small circular play indicator.
    fragmentPlayBtn: (active, loaded) => (theme) => ({
        width: 26,
        height: 26,
        borderRadius: RADIUS.pill,
        color: theme.palette.perf.label,
        backgroundColor: active || loaded ? theme.palette.fragmenta.surface : ON_TINT,
        boxShadow: active || loaded ? `0 1px 3px rgba(${theme.palette.fragmenta.inkRgb}, 0.14)` : 'none',
        transition: 'background-color 160ms ease',
        '&:hover': { bgcolor: theme.palette.fragmenta.surface },
    }),
    fragmentDeleteBtn: {
        width: 20,
        height: 20,
        borderRadius: RADIUS.pill,
        color: 'text.disabled',
        '&:hover': { color: 'error.main', bgcolor: (t) => t.palette.fragmenta.surface },
    },
    fragmentMeta: {
        flex: 1,
        minWidth: 0,
        display: 'flex',
        alignItems: 'baseline',
        gap: 0.5,
        overflow: 'hidden',
    },
    fragmentOrdinal: {
        fontSize: '0.75rem',
        fontWeight: 600,
        fontVariantNumeric: 'tabular-nums',
        color: 'text.primary',
        flexShrink: 0,
    },
    fragmentLoaded: {
        fontSize: '0.625rem',
        fontWeight: 600,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        color: 'text.primary',
        flexShrink: 0,
    },
    fragmentLabelText: {
        ...perfTokens.num,
        fontWeight: perfTokens.weight.bold,
        color: 'text.primary',
        minWidth: 0,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        cursor: 'text',
    },
    fragmentLabelInput: {
        ...perfTokens.num,
        fontWeight: perfTokens.weight.bold,
        flex: 1,
        minWidth: 0,
        '& input': {
            p: 0,
            height: 16,
            font: 'inherit',
            color: 'text.primary',
        },
    },
    fragmentDuration: {
        fontSize: '0.6875rem',
        fontVariantNumeric: 'tabular-nums',
        color: 'text.secondary',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
    },
    waveformPlaceholder: {
        ...perfTokens.caps,
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'text.disabled',
        pointerEvents: 'none',
    },
    knobsGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
        gap: 0.5,
    },
    // "-12 st" / "+12 st" either side of the pitch knob.
    pitchRangeLabel: {
        fontSize: perfTokens.fontSize.xs,
        color: 'text.secondary',
        fontVariantNumeric: 'tabular-nums',
        whiteSpace: 'nowrap',
        userSelect: 'none',
    },
    transportRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 0.5,
        mt: 'auto',
    },
    // Play: a circle on the control tone; filled with the lane colour while
    // playing.
    transportBtn: (color, playing) => (theme) => ({
        width: 36,
        height: 36,
        borderRadius: RADIUS.pill,
        color: playing ? onFill(color) : theme.palette.perf.label,
        backgroundColor: playing ? color : theme.palette.fragmenta.surface,
        boxShadow: `0 1px 2px rgba(${theme.palette.fragmenta.inkRgb}, 0.08), 0 4px 10px rgba(${theme.palette.fragmenta.inkRgb}, 0.08)`,
        transition: 'background-color 160ms ease',
        '&:hover': { backgroundColor: playing ? color : theme.palette.fragmenta.paperHi },
        '&.Mui-disabled': { color: 'text.disabled', boxShadow: 'none', backgroundColor: 'transparent' },
    }),
    loopBtn: (soft, active) => (theme) => ({
        width: 30,
        height: 30,
        borderRadius: RADIUS.pill,
        color: active ? theme.palette.perf.label : 'text.secondary',
        backgroundColor: active ? soft : 'transparent',
        transition: 'background-color 160ms ease, color 160ms ease',
        '&:hover': {
            backgroundColor: active ? soft : ON_TINT,
            color: 'text.primary',
        },
    }),
    // Lane level: a slim rounded bar.
    meterTrack: (theme) => ({
        flex: 1,
        height: 6,
        mt: 0.75,
        backgroundColor: ON_TINT,
        boxShadow: `inset 0 1px 1px rgba(${theme.palette.fragmenta.inkRgb}, 0.10)`,
        borderRadius: RADIUS.pill,
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
    }),
    // Full width; the lane reveals it with a clip-path (see the meter tick).
    meterFill: (color, deep) => (theme) => ({
        width: '100%',
        height: '100%',
        clipPath: 'inset(0 100% 0 0)',
        borderRadius: RADIUS.pill,
        background: `linear-gradient(90deg, ${color} 0%, ${deep} 82%, ${theme.palette.error.main} 100%)`,
    }),
};

// M / S / SC button: compact rounded rectangle, filled with `fill` when on.
function smallToggle(theme, active, fill) {
    return {
        width: 28,
        height: 28,
        fontSize: '0.6875rem',
        fontWeight: 700,
        borderRadius: RADIUS.small,
        color: active ? onFill(fill) : 'text.secondary',
        backgroundColor: active ? fill : ON_TINT,
        transition: 'background-color 160ms ease, color 160ms ease',
        '&:hover': {
            backgroundColor: active ? fill : ON_TINT_HOVER,
            color: active ? onFill(fill) : 'text.primary',
        },
    };
}

export const paletteThemes = Object.fromEntries(
    Object.entries(PALETTES).map(([key, P]) => [key, makePaletteTheme(paletteTokens(P))]),
);
export const lightTheme = paletteThemes[DEFAULT_PALETTE];

export default theme;
