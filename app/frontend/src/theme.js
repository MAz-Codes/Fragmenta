import { createTheme, responsiveFontSizes } from '@mui/material/styles';

/* =====================================================================
 * Fragmenta 0.2.0 theme — Arcade (dark) × Paper (light)
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
    warm:      '#FDA22B',
    warmHi:    '#FFB855',
    warmLo:    '#D17F1A',
    // Deep blue used for moon-icon / night cues on dark mode.
    night:     '#3D6FA8',
    blue:      '#5BA9E8',      // secondary — selected-file cue
    blueDim:   '#84BFEE',
    success:   '#7AC795',
    error:     '#E26B5E',
    warning:   '#E3A34B',
};

// --- Paper (light) palette --------------------------------------------------
const LIGHT = {
    bg:        '#F2EDE3',      // lighter cream — less amber, closer to warm white
    bgElev:    '#F6F1E8',
    paper:     '#F8F3EA',
    paperHi:   '#FBF6EE',
    divider:   'rgba(43, 31, 18, 0.16)',
    text:      '#2B1F12',      // warm dark brown
    textDim:   '#4D3F2A',      // darker for ~7:1 contrast on cream
    textFaint: '#7A6A50',      // bumped from #9C8B70 for ~4.5:1
    // "amber" is a legacy token name — the accent is actually a deep
    // cyan that mirrors dark mode. Cream + gold read as muddy yellow-on-
    // amber; cyan gives the accents real separation from the warm paper.
    amber:     '#1F7E94',      // deep cyan accent for legibility on cream
    amberHi:   '#2DA0BC',
    amberLo:   '#155F71',
    // Warm complement on cream — same hue family as dark mode but deeper
    // for legibility against the warm-paper background.
    warm:      '#C97A1A',
    warmHi:    '#E59334',
    warmLo:    '#9C5C0F',
    // Deep navy for moon-icon / night cues on light mode — needs strong
    // contrast against the warm cream paper.
    night:     '#1F3A5F',
    blue:      '#3B6E9B',
    blueDim:   '#5E8FB8',
    success:   '#2E8A52',
    error:     '#B84E45',
    warning:   '#B47318',
};

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
export const PERF_BTN_SHADOW = '0 3px 8px rgba(0,0,0,0.45), 0 12px 24px rgba(0,0,0,0.55)';
export const PERF_BTN_SHADOW_HOVER = '0 5px 12px rgba(0,0,0,0.5), 0 16px 32px rgba(0,0,0,0.7)';

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
            shortest: 200,
            shorter: 280,
            short: 330,
            standard: 420,
            complex: 500,
            enteringScreen: 320,
            leavingScreen: 280,
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
            fontWeight: 400,
            letterSpacing: '-0.015em',
            fontSize: '1.2rem',
            fontVariationSettings: '"opsz" 96',
        },
        // Tier-2 section/accordion labels — same family, slightly smaller.
        // "Annotator Labels", "Advanced Settings", "Edit existing audio".
        subtitle1: {
            fontFamily: FONT_DISPLAY,
            fontWeight: 400,
            letterSpacing: '-0.01em',
            fontSize: '1.05rem',
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
                root: { color: DARK.amber, height: 4 },
                rail: { backgroundColor: 'rgba(240, 237, 229, 0.12)', opacity: 1 },
                track: { backgroundColor: DARK.amber, border: 0 },
                thumb: {
                    backgroundColor: DARK.amberHi,
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
                markActive: { backgroundColor: DARK.amber },
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

export const lightTheme = createTheme(theme, {
    palette: {
        mode: 'light',
        primary: { main: LIGHT.amber, light: LIGHT.amberHi, dark: LIGHT.amberLo, contrastText: '#FFFBF1' },
        secondary: { main: LIGHT.blue, light: LIGHT.blueDim, dark: '#2B547A', contrastText: '#FFFBF1' },
        background: { default: LIGHT.bg, paper: LIGHT.paper },
        text: { primary: LIGHT.text, secondary: LIGHT.textDim, disabled: LIGHT.textFaint },
        action: {
            active: 'rgba(43, 31, 18, 0.6)',
            hover: 'rgba(43, 31, 18, 0.04)',
            selected: 'rgba(43, 31, 18, 0.08)',
            disabled: 'rgba(43, 31, 18, 0.26)',
            disabledBackground: 'rgba(43, 31, 18, 0.10)',
        },
        divider: LIGHT.divider,
        error: { main: LIGHT.error },
        warning: { main: LIGHT.warning },
        success: { main: LIGHT.success },
        info: { main: LIGHT.blue },
        warm: { main: LIGHT.warm, light: LIGHT.warmHi, dark: LIGHT.warmLo },
        night: { main: LIGHT.night },
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
                    backgroundImage:
                        `radial-gradient(900px 700px at -5% 50%, rgba(31, 126, 148, 0.06), transparent 60%), ` +
                        `radial-gradient(1100px 700px at 95% 108%, rgba(201, 122, 26, 0.05), transparent 55%), ` +
                        `linear-gradient(165deg, #F7F2E8 0%, ${LIGHT.bg} 42%, #ECE5D5 100%)`,
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
                    background: 'rgba(43, 31, 18, 0.18)',
                    borderRadius: '999px',
                    border: '2px solid rgba(0, 0, 0, 0)',
                    backgroundClip: 'padding-box',
                    '&:hover': { background: 'rgba(43, 31, 18, 0.32)' },
                },
                '*::-webkit-scrollbar-corner': { background: 'transparent' },
                '*': {
                    scrollbarWidth: 'thin',
                    scrollbarColor: 'rgba(43, 31, 18, 0.18) transparent',
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    // Liquid Glass on cream: translucent warm paper with
                    // softer rim highlights so the surface reads as glass
                    // on paper rather than glass on charcoal.
                    backgroundColor: 'rgba(248, 241, 224, 0.72)',
                    backgroundImage: 'none',
                    backdropFilter: 'blur(22px) saturate(160%)',
                    WebkitBackdropFilter: 'blur(22px) saturate(160%)',
                    border: 'none',
                    boxShadow:
                        '0 18px 36px rgba(43, 31, 18, 0.10), ' +
                        '0 3px 10px rgba(43, 31, 18, 0.06), ' +
                        'inset 0 1px 0 rgba(255, 255, 255, 0.7), ' +
                        'inset 0 -1px 0 rgba(43, 31, 18, 0.08), ' +
                        'inset 1px 0 0 rgba(255, 255, 255, 0.3), ' +
                        'inset -1px 0 0 rgba(43, 31, 18, 0.04)',
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(248, 241, 224, 0.72)',
                    backgroundImage: 'none',
                    backdropFilter: 'blur(22px) saturate(160%)',
                    WebkitBackdropFilter: 'blur(22px) saturate(160%)',
                    border: 'none',
                    boxShadow:
                        '0 18px 36px rgba(43, 31, 18, 0.10), ' +
                        '0 3px 10px rgba(43, 31, 18, 0.06), ' +
                        'inset 0 1px 0 rgba(255, 255, 255, 0.7), ' +
                        'inset 0 -1px 0 rgba(43, 31, 18, 0.08)',
                    transition: 'box-shadow 220ms ease, background-color 220ms ease',
                    '&:hover': {
                        backgroundColor: 'rgba(252, 246, 232, 0.82)',
                        boxShadow:
                            '0 24px 48px rgba(43, 31, 18, 0.14), ' +
                            '0 5px 14px rgba(43, 31, 18, 0.08), ' +
                            'inset 0 1px 0 rgba(255, 255, 255, 0.85), ' +
                            'inset 0 -1px 0 rgba(43, 31, 18, 0.10), ' +
                            '0 0 0 1px rgba(31, 126, 148, 0.25)',
                    },
                },
            },
        },
        MuiButton: {
            variants: [
                {
                    props: { color: 'warm', variant: 'contained' },
                    style: {
                        backgroundImage: `linear-gradient(135deg, ${LIGHT.warmHi} 0%, ${LIGHT.warm} 55%, ${LIGHT.warmLo} 100%)`,
                        color: '#FFFBF1',
                        '&:hover': {
                            backgroundImage: `linear-gradient(135deg, ${LIGHT.warmHi} 0%, ${LIGHT.warmHi} 55%, ${LIGHT.warm} 100%)`,
                        },
                    },
                },
            ],
            styleOverrides: {
                contained: {
                    boxShadow: '0 6px 14px rgba(31, 126, 148, 0.18)',
                    '&:hover': { boxShadow: '0 10px 20px rgba(31, 126, 148, 0.26)' },
                    '&.Mui-disabled': {
                        backgroundImage: 'none',
                        backgroundColor: 'rgba(43, 31, 18, 0.08)',
                        color: 'rgba(43, 31, 18, 0.30)',
                        boxShadow: 'none',
                    },
                },
                containedPrimary: {
                    backgroundImage: `linear-gradient(135deg, ${LIGHT.amberHi} 0%, ${LIGHT.amber} 55%, ${LIGHT.amberLo} 100%)`,
                    color: '#FFFBF1',
                },
                containedSecondary: {
                    backgroundImage: `linear-gradient(135deg, ${LIGHT.blueDim} 0%, ${LIGHT.blue} 100%)`,
                },
                containedError: {
                    backgroundImage: 'linear-gradient(135deg, #C95A50 0%, #A03B33 100%)',
                },
                outlined: {
                    // Cyan accent by default, mirroring dark-mode outlined
                    // buttons. Keeps "Browse / Choose CSV" etc. visually
                    // active without going to a fully filled (contained)
                    // treatment.
                    borderColor: 'rgba(31, 126, 148, 0.50)',
                    color: LIGHT.amber,
                    '&:hover': {
                        borderColor: LIGHT.amberLo,
                        backgroundColor: 'rgba(31, 126, 148, 0.10)',
                        color: LIGHT.amberLo,
                    },
                    '&.Mui-disabled': {
                        borderColor: 'rgba(43, 31, 18, 0.12)',
                        color: 'rgba(43, 31, 18, 0.30)',
                    },
                },
                outlinedError: {
                    borderColor: 'rgba(184, 78, 69, 0.55)',
                    color: LIGHT.error,
                    '&:hover': {
                        borderColor: LIGHT.error,
                        backgroundColor: 'rgba(184, 78, 69, 0.10)',
                        color: LIGHT.error,
                    },
                },
                text: {
                    color: LIGHT.text,
                    '&:hover': {
                        backgroundColor: 'rgba(31, 126, 148, 0.08)',
                        color: LIGHT.amberLo,
                    },
                },
            },
        },
        MuiTextField: {
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        backgroundColor: 'rgba(255, 251, 241, 0.6)',
                        borderRadius: 8,
                        '& fieldset': { borderColor: 'rgba(43, 31, 18, 0.18)' },
                        '&:hover fieldset': { borderColor: 'rgba(43, 31, 18, 0.36)' },
                        '&.Mui-focused fieldset': {
                            borderColor: LIGHT.amber,
                            boxShadow: '0 0 0 3px rgba(31, 126, 148, 0.14)',
                        },
                    },
                },
            },
        },
        MuiSelect: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(255, 251, 241, 0.6)',
                    borderRadius: 8,
                    '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(43, 31, 18, 0.18)' },
                    '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(43, 31, 18, 0.36)' },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: LIGHT.amber,
                        boxShadow: '0 0 0 3px rgba(31, 126, 148, 0.14)',
                    },
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
                paper: {
                    border: `1px solid ${LIGHT.divider}`,
                    borderRadius: 20,
                    overflow: 'hidden',
                },
            },
        },
        MuiMenu: {
            defaultProps: { disableScrollLock: true },
            styleOverrides: {
                paper: {
                    border: `1px solid ${LIGHT.divider}`,
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
                    backgroundColor: LIGHT.paper,
                    '&:hover': { backgroundColor: LIGHT.paperHi },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(31, 126, 148, 0.14)',
                        color: LIGHT.amberLo,
                        '&:hover': { backgroundColor: 'rgba(31, 126, 148, 0.20)' },
                    },
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(43, 31, 18, 0.06)',
                    color: LIGHT.text,
                    border: `1px solid rgba(43, 31, 18, 0.14)`,
                    borderRadius: 999,
                    '&.MuiChip-filled.MuiChip-colorPrimary': {
                        backgroundColor: LIGHT.amber,
                        backgroundImage: `linear-gradient(135deg, ${LIGHT.amberHi} 0%, ${LIGHT.amber} 55%, ${LIGHT.amberLo} 100%)`,
                        color: '#FFFBF1',
                        borderColor: LIGHT.amber,
                        boxShadow: RAISE_LIGHT,
                    },
                    '&.MuiChip-filled.MuiChip-colorSuccess': {
                        backgroundColor: LIGHT.success,
                        backgroundImage: `linear-gradient(135deg, #3FA86A 0%, ${LIGHT.success} 55%, #1F6038 100%)`,
                        color: '#FFFBF1',
                        borderColor: LIGHT.success,
                        boxShadow: RAISE_LIGHT,
                    },
                },
                outlined: {
                    borderColor: 'rgba(43, 31, 18, 0.20)',
                    '&.MuiChip-colorPrimary': { borderColor: LIGHT.amber, color: LIGHT.amberLo },
                },
            },
        },
        MuiAccordion: {
            styleOverrides: {
                root: {
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
                    '&.Mui-expanded:last-of-type': { marginBottom: 16 },
                },
            },
        },
        MuiAccordionSummary: {
            styleOverrides: {
                root: {
                    backgroundColor: 'transparent',
                    '&:hover': { backgroundColor: 'rgba(31, 126, 148, 0.06)' },
                },
            },
        },
        MuiDialog: {
            defaultProps: { disableScrollLock: true },
            styleOverrides: {
                paper: {
                    // Liquid Glass on cream — same shape as the global
                    // MuiPaper override so dialogs match the rest of the
                    // surfaces (translucent warm + soft rim highlights).
                    backgroundColor: 'rgba(248, 241, 224, 0.72)',
                    backgroundImage: 'none',
                    backdropFilter: 'blur(22px) saturate(160%)',
                    WebkitBackdropFilter: 'blur(22px) saturate(160%)',
                    border: 'none',
                    borderRadius: 14,
                    boxShadow:
                        '0 24px 48px rgba(43, 31, 18, 0.18), ' +
                        '0 3px 10px rgba(43, 31, 18, 0.06), ' +
                        'inset 0 1px 0 rgba(255, 255, 255, 0.7), ' +
                        'inset 0 -1px 0 rgba(43, 31, 18, 0.08), ' +
                        'inset 1px 0 0 rgba(255, 255, 255, 0.3), ' +
                        'inset -1px 0 0 rgba(43, 31, 18, 0.04)',
                },
            },
        },
        MuiDialogTitle: {
            styleOverrides: {
                root: {
                    backgroundColor: 'transparent',
                    borderBottom: `1px solid ${LIGHT.divider}`,
                    color: LIGHT.text,
                    fontWeight: 600,
                    fontSize: '1.1rem',
                    letterSpacing: '-0.01em',
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
                    '&:hover': { backgroundColor: 'rgba(31, 126, 148, 0.06)' },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(31, 126, 148, 0.14)',
                        '&:hover': { backgroundColor: 'rgba(31, 126, 148, 0.20)' },
                    },
                },
            },
        },
        MuiCheckbox: {
            styleOverrides: {
                root: {
                    color: LIGHT.textDim,
                    '&.Mui-checked': { color: LIGHT.amber },
                    '&:hover': { backgroundColor: 'rgba(31, 126, 148, 0.08)' },
                },
            },
        },
        MuiFormControlLabel: { styleOverrides: { label: { color: LIGHT.text } } },
        MuiSlider: {
            styleOverrides: {
                root: { color: LIGHT.amber, height: 4 },
                rail: { backgroundColor: 'rgba(43, 31, 18, 0.18)', opacity: 1 },
                track: { backgroundColor: LIGHT.amber, border: 0 },
                thumb: {
                    backgroundColor: LIGHT.amber,
                    width: 16,
                    height: 16,
                    border: `2px solid ${LIGHT.paper}`,
                    boxShadow: '0 2px 6px rgba(43, 31, 18, 0.20)',
                    '&:hover, &.Mui-focusVisible': {
                        boxShadow: '0 0 0 8px rgba(31, 126, 148, 0.18)',
                    },
                },
                valueLabel: {
                    backgroundColor: LIGHT.text,
                    color: '#FFFBF1',
                    borderRadius: 6,
                    fontWeight: 400,
                },
                mark: { backgroundColor: 'rgba(43, 31, 18, 0.28)' },
                markActive: { backgroundColor: LIGHT.amber },
            },
        },
        MuiLinearProgress: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(43, 31, 18, 0.10)',
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
                    backgroundColor: 'rgba(43, 31, 18, 0.32)',
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
                        backgroundColor: 'rgba(31, 126, 148, 0.10)',
                        color: LIGHT.amberLo,
                    },
                },
            },
        },
        MuiToggleButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    border: `1px solid ${LIGHT.divider}`,
                    color: LIGHT.textDim,
                    fontWeight: 400,
                    '&:hover': {
                        backgroundColor: 'rgba(31, 126, 148, 0.06)',
                        color: LIGHT.text,
                    },
                    '&.Mui-selected': {
                        backgroundColor: LIGHT.amber,
                        backgroundImage: `linear-gradient(135deg, ${LIGHT.amberHi} 0%, ${LIGHT.amber} 55%, ${LIGHT.amberLo} 100%)`,
                        color: '#FFFBF1',
                        borderColor: LIGHT.amber,
                        boxShadow: RAISE_LIGHT,
                        '&:hover': {
                            backgroundImage: `linear-gradient(135deg, ${LIGHT.amberHi} 0%, ${LIGHT.amberHi} 55%, ${LIGHT.amber} 100%)`,
                            backgroundColor: LIGHT.amberHi,
                        },
                    },
                },
            },
        },
        MuiSwitch: {
            styleOverrides: {
                switchBase: {
                    '&.Mui-checked': {
                        color: '#FFFBF1',
                        '& + .MuiSwitch-track': {
                            backgroundColor: LIGHT.amber,
                            opacity: 1,
                        },
                    },
                },
                track: { backgroundColor: 'rgba(43, 31, 18, 0.30)' },
            },
        },
        MuiTooltip: {
            styleOverrides: {
                tooltip: {
                    backgroundColor: LIGHT.text,
                    color: '#FFFBF1',
                    fontWeight: 400,
                    fontSize: '0.75rem',
                    borderRadius: 6,
                    boxShadow: '0 8px 18px rgba(43, 31, 18, 0.24)',
                },
                arrow: { color: LIGHT.text },
            },
        },
    },
});

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
    headerRowScrolled: (theme) => ({
        backdropFilter: 'blur(14px)',
        WebkitBackdropFilter: 'blur(14px)',
        backgroundColor: theme.palette.mode === 'dark'
            ? 'rgba(31, 32, 33, 0.55)'
            : 'rgba(242, 237, 227, 0.65)',
        borderBottom: theme.palette.mode === 'dark'
            ? '1px solid rgba(255, 255, 255, 0.04)'
            : '1px solid rgba(43, 31, 18, 0.08)',
    }),
    headerBrand: {
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        gap: { xs: 1, sm: 1.5 },
        py: 0,
    },
    logo: {
        width: 44,
        height: 44,
        backgroundImage: 'url(/fragmenta_icon_1024.png)',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        borderRadius: 1.5,
        filter: 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3))',
    },
    title: {
        color: 'text.primary',
        fontFamily: '"Bitcount Single", "IBM Plex Mono", "JetBrains Mono", "Space Mono", "Courier New", monospace',
        fontWeight: 400,
        fontSize: { xs: '1.5rem', sm: '1.65rem' },
        letterSpacing: '0.02em',
        textShadow: '0 2px 10px rgba(0, 0, 0, 0.6)',
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
    gpuCard: (isCompactLayout) => ({
        // Layout-only — Paper component's MuiPaper.root override owns
        // bg/border/shadow so this card inherits the same Liquid Glass
        // treatment as every other Tier-1 card.
        px: 1.75,
        py: 1.25,
        borderRadius: 2,
        minWidth: isCompactLayout ? '100%' : 240,
        flexShrink: 0,
        position: 'relative',
        overflow: 'hidden',
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
        borderRadius: 2.5,
        overflow: 'hidden',
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
        gap: { xs: 0.25, sm: 0.5 },
        '& .MuiTabs-indicator': {
            display: 'none',
        },
        '& .MuiTab-root': {
            alignItems: 'center',
            justifyContent: (isCompactLayout || isIconOnly) ? 'center' : 'flex-start',
            textAlign: (isCompactLayout || isIconOnly) ? 'center' : 'left',
            minHeight: { xs: 40, sm: 46 },
            // Match the dropdown / Select font size so the nav rail labels
            // read at the same scale as in-page form chrome.
            fontSize: '0.8rem',
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
            '&.Mui-selected': {
                color: 'primary.main',
                fontWeight: 600,
                backgroundColor: 'rgba(53, 194, 212, 0.16)',
                boxShadow: 'inset 0 0 0 1px rgba(115, 215, 227, 0.35)',
            },
            '&:hover': {
                color: 'text.primary',
                backgroundColor: 'rgba(53, 194, 212, 0.08)',
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
        '& .MuiTypography-body1, & .MuiTypography-body2, & .MuiTypography-caption, & .MuiTypography-subtitle2': {
            fontSize: muiTheme.typography.body2.fontSize,
            lineHeight: muiTheme.typography.body2.lineHeight,
        },
        '& .MuiButton-startIcon svg, & .MuiButton-endIcon svg, & .MuiIconButton-root svg, & .MuiAccordionSummary-expandIconWrapper svg': {
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
        gap: { xs: 1, sm: 2 },
        mb: 2,
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
    sectionCardIcon: {
        display: 'inline-flex',
        color: 'text.primary',
        lineHeight: 0,
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
        borderRadius: 2.5,
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
            backgroundColor: 'rgba(53, 194, 212, 0.08)',
        },
        '&.Mui-disabled': {
            opacity: 0.45,
        },
    },
    dockIconButtonAccent: {
        color: 'primary.main',
        '&:hover': {
            color: 'primary.light',
            backgroundColor: 'rgba(53, 194, 212, 0.10)',
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
    // Theme-driven coloring — MuiPaper handles bg/border/shadow.
    // Card grows with content up to a sensible cap, then scrolls.
    rootPaper: {
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 2.5,
        // Tall enough to show ~9 rows before scrolling. Min keeps the empty
        // state from collapsing the layout when no fragments exist yet.
        minHeight: 240,
        maxHeight: 520,
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
    titleIcon: {
        display: 'inline-flex',
        color: 'text.primary',
        lineHeight: 0,
    },
    titleText: {},  // typography handled by the h6 variant in theme
    countText: {
        fontWeight: 600,
        minWidth: 20,
        textAlign: 'right',
    },
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
        // Each row is ~40px (single-line layout); cap so the card height
        // never exceeds rootPaper.maxHeight regardless of fragment count.
        p: 0,
        '& .MuiListItem-root': {
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 1.5,
            mb: 0.75,
            bgcolor: 'background.default',
            '&:last-child': { mb: 0 },
        },
    },
    // One-row, Spotify-style layout:
    //   [▶]   prompt text (ellipsis)    Xs · ago    (i)   [⬇]
    listItem: {
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        py: 0.5,
        px: 1,
        minHeight: 40,
    },
    fragmentMeta: {
        flex: 1,
        minWidth: 0,
    },
    fragmentPrompt: {
        fontWeight: 500,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
    },
    batchTag: {
        display: 'inline-block',
        fontWeight: 700,
        fontSize: '0.7rem',
        px: 0.5,
        py: 0,
        mr: 0.75,
        borderRadius: 0.75,
        bgcolor: 'action.selected',
        color: 'text.secondary',
        fontVariantNumeric: 'tabular-nums',
    },
    fragmentMetaInline: {
        flexShrink: 0,
        whiteSpace: 'nowrap',
        fontVariantNumeric: 'tabular-nums',
    },
    fragmentInfoIcon: {
        display: 'inline-flex',
        alignItems: 'center',
        color: 'text.secondary',
        opacity: 0.5,
        cursor: 'help',
        transition: 'opacity 150ms ease',
        '&:hover': { opacity: 0.95 },
    },
    playPauseButton: (isPlaying) => (muiTheme) => ({
        flexShrink: 0,
        border: '1px solid',
        borderColor: isPlaying
            ? 'primary.main'
            : (muiTheme.palette.mode === 'dark' ? 'rgba(194, 207, 228, 0.22)' : 'rgba(15, 23, 42, 0.2)'),
        color: isPlaying ? 'primary.main' : 'inherit',
    }),
    downloadButton: {
        flexShrink: 0,
        color: 'text.secondary',
        '&:hover': { color: 'primary.main' },
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
        borderRadius: 2.5,
        transition: 'all 0.3s ease',
    },
    headerRow: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 1,
        mb: 2,
    },
    headerTitleWrap: {
        display: 'flex',
        alignItems: 'center',
        gap: 1,
    },
    headerIcon: {
        display: 'inline-flex',
        color: 'text.primary',
        lineHeight: 0,
    },
    headerTitle: {},  // typography handled by the h6 variant in theme
    statusInline: {
        display: 'flex',
        alignItems: 'center',
        gap: 0.75,
        px: 1,
        py: 0.35,
        borderRadius: 999,
        border: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.default',
    },
    statusDot: (status, animate = false) => ({
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
    statusText: (status) => ({
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
    progressSection: {
        mb: 2,
    },
    progressHeader: {
        display: 'flex',
        justifyContent: 'space-between',
        mb: 1,
    },
    progressBar: {
        height: 8,
        borderRadius: 4,
    },
    deviceSection: {
        mb: 2,
    },
    deviceInfo: {
        fontSize: { xs: '0.74rem', sm: '0.8rem' },
        mt: 0.5,
    },
    metricsGrid: {
        mb: 2,
    },
    lossSection: {
        mb: 2,
    },
    lossChartBox: {
        height: 200,
        width: '100%',
    },
    lossDisclaimer: {
        display: 'block',
        mt: 0.75,
        fontStyle: 'italic',
        lineHeight: 1.4,
        opacity: 0.75,
    },
    errorAlert: {
        mb: 2,
    },
};

export const welcomePageStyles = {
    backdrop: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        // Match the main app body's three-layer treatment so the
        // welcome page reads as part of the same surface, not a
        // separate dialog with its own palette.
        return {
            zIndex: 9999,
            background: isDark
                ? `radial-gradient(900px 700px at -5% 50%, rgba(39, 159, 187, 0.18), transparent 60%), ` +
                  `radial-gradient(1100px 700px at 95% 108%, rgba(253, 162, 43, 0.14), transparent 55%), ` +
                  `linear-gradient(165deg, #181A1B 0%, ${DARK.bg} 42%, #1A1B1C 100%)`
                : `radial-gradient(900px 700px at -5% 50%, rgba(31, 126, 148, 0.09), transparent 60%), ` +
                  `radial-gradient(1100px 700px at 95% 108%, rgba(201, 122, 26, 0.08), transparent 55%), ` +
                  `linear-gradient(165deg, #F7F2E8 0%, ${LIGHT.bg} 42%, #ECE5D5 100%)`,
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
        fontFamily: '"Bitcount Single", "IBM Plex Mono", "JetBrains Mono", "Space Mono", "Courier New", monospace',
        fontWeight: 400,
        color: 'text.primary',
        mb: 1,
        fontSize: { xs: '2.5rem', sm: '3.5rem', md: '4rem' },
        letterSpacing: '0.02em',
    },
};

export const lossChartStyles = {
    padding: { top: 10, right: 16, bottom: 28, left: 44 },
    colors: {
        grid: '#2B3446',
        axis: '#9DA9BC',
        line: '#35C2D4',
        point: '#73D7E3',
        tooltipBg: '#111826',
        tooltipBorder: 'rgba(194, 207, 228, 0.2)',
        tooltipText: '#E8EDF5',
    },
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
        sub: 22,       // small subordinate square toggles (mute, solo, loop)
        compact: 26,   // primary compact controls (Link, MIDI, Q, BPM, transport)
        cta: 28,       // primary CTA pill (Generate)
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
        const sz = { xs: 18, sm: 20, md: 22, lg: 26 }[size];
        return {
            width: sz,
            height: sz,
            borderRadius: 1,
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
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
        background: 'linear-gradient(135deg, rgba(53, 194, 212, 0.05) 0%, rgba(159, 138, 230, 0.04) 100%)',
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
        '& .MuiOutlinedInput-root': {
            fontFamily: 'inherit',
            height: perfTokens.height.compact,
            borderRadius: 1.5,
        },
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
        gridTemplateColumns: 'repeat(4, minmax(200px, 300px))',
        gap: 1.25,
        flex: 1,
        minWidth: 0,
    },
    masterStrip: (color) => (theme) => {
        const dark = theme.palette.mode === 'dark';
        // Matches the channel-strip at-rest treatment so master sits in the
        // same visual family — same liquid glass (translucent + backdrop blur
        // + inset rim) rather than a higher-elevation solid surface.
        const glassBg = dark ? 'rgba(38, 41, 44, 0.42)' : 'rgba(248, 243, 234, 0.45)';
        const rim = dark
            ? 'inset 0 1px 0 rgba(255,255,255,0.22), inset 0 -1px 0 rgba(0,0,0,0.32)'
            : 'inset 0 1px 0 rgba(255,255,255,0.55), inset 0 -1px 0 rgba(0,0,0,0.10)';
        return {
            display: 'flex',
            flexDirection: 'column',
            gap: 1.5,
            p: 1.25,
            borderRadius: 2.5,
            border: '1px solid',
            borderColor: dark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.10)',
            background: glassBg,
            backdropFilter: 'blur(28px) saturate(200%)',
            WebkitBackdropFilter: 'blur(28px) saturate(200%)',
            boxShadow: `0 2px 10px ${dark ? 'rgba(4, 8, 14, 0.34)' : 'rgba(0,0,0,0.08)'}, ${rim}`,
            width: { xs: '100%', md: 150 },
            flex: { xs: '1 1 100%', md: '0 0 150px' },
            minHeight: 0,
            overflow: 'hidden',
            boxSizing: 'border-box',
        };
    },
    masterHeader: (color) => ({
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 0.5,
        // Match the channel header's content height (driven there by the
        // badge chip) so the divider lands at the same vertical position —
        // the plain caps label is shorter and otherwise pulls it higher.
        minHeight: perfTokens.height.compact,
        borderBottom: `1px solid ${color}33`,
        pb: 0.75,
        color,
    }),
    masterBadge: () => ({
        // Plain caps label — same family as TAKES, MIDI SETTINGS, etc.
        // Dropped the chip (tinted bg + colored ring) so master reads as a
        // section label rather than a special UI element.
        ...perfTokens.caps,
        color: 'text.secondary',
    }),
    masterFaderWrap: {
        // Fixed-height lane for the meter + vertical fader. Was `flex: 1`
        // (filled all remaining vertical space) — that made the fader feel
        // disproportionately tall. The JSX now wraps this + the readouts in
        // a flex-column with justify-content: center, so this fixed height
        // sits vertically centered between the header and the FX pickers.
        // 320 px ≈ 80% of the previously-stretched height.
        height: 320,
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'stretch',
        justifyContent: 'center',
        gap: 1,
        py: 1.25,
        minHeight: 0,
        flexShrink: 0,
    },
    masterMeterTrack: {
        width: 10,
        bgcolor: 'background.default',
        borderRadius: 0.75,
        border: '1px solid',
        borderColor: 'divider',
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'flex-end',
    },
    masterMeterFill: (color) => (theme) => ({
        width: '100%',
        height: '0%',
        // Peaks light up in warning (amber) then error (red) — uses theme
        // tokens instead of hardcoded #E3A34B / #E36C61 so the meter shifts
        // automatically when the palette changes.
        background: `linear-gradient(0deg, ${color} 0%, ${color}DD 55%, ${theme.palette.warning.main} 78%, ${theme.palette.error.main} 100%)`,
        transition: 'height 0.05s linear',
    }),
    masterFader: (color) => ({
        height: '100%',
        color,
        '& .MuiSlider-thumb': { width: 16, height: 16 },
        '& .MuiSlider-rail': { opacity: 0.3, width: 4 },
        // Filled track from the bottom (min dB) up to the thumb — gives a
        // quick visual read of how much gain you've dialed in. Width matches
        // the rail so the fill sits flush inside it.
        '& .MuiSlider-track': {
            width: 4,
            border: 'none',
            opacity: 0.9,
        },
        '& .MuiSlider-mark': {
            width: 6,
            height: 1,
            bgcolor: 'text.disabled',
            opacity: 1,
        },
        '& .MuiSlider-markActive': {
            bgcolor: 'text.disabled',
        },
    }),
    masterReadouts: {
        display: 'flex',
        flexDirection: 'column',
        gap: 0.25,
        alignItems: 'center',
    },
    masterValue: {
        ...perfTokens.num,
        textAlign: 'center',
        color: 'primary.main',
    },
    masterPeakValue: {
        ...perfTokens.num,
        textAlign: 'center',
        color: 'text.disabled',
        // Visually subordinate to the main dB readout — kept xs intentionally
        // even though `num` sets sm.
        fontSize: perfTokens.fontSize.xs,
    },
    masterTransport: {
        display: 'flex',
        flexDirection: 'column',
        gap: 0.5,
        pt: 0.75,
        borderTop: '1px solid',
        borderTopColor: 'divider',
    },
    masterBtn: (color, variant) => (theme) => {
        // Gradient fill + drop shadow, NO border (matches the Suggest-
        // hyperparameters contained button). 'stop' uses the error palette;
        // 'play' uses the passed color.
        const fill = variant === 'stop' ? theme.palette.error.main : color;
        return {
            textTransform: 'none',
            borderRadius: 1.5,
            fontSize: perfTokens.fontSize.sm,
            // Match the app's buttons (MuiButton root is 400) — not bold.
            fontWeight: 400,
            height: perfTokens.height.compact,
            // Override MUI's vertical padding so the button lands at the
            // shared 26px row height (matches Q select + BPM input).
            py: 0,
            px: 1.25,
            minWidth: 0,
            lineHeight: 1,
            border: 'none',
            color: 'rgba(0,0,0,0.88)',
            background: `${PERF_BTN_SHEEN}, ${fill}`,
            boxShadow: PERF_BTN_SHADOW,
            transition: 'box-shadow 200ms cubic-bezier(0.16,1,0.3,1), transform 200ms cubic-bezier(0.16,1,0.3,1), filter 120ms',
            '&:hover': {
                border: 'none',
                filter: 'brightness(1.05)',
                transform: 'translateY(-1px)',
                boxShadow: PERF_BTN_SHADOW_HOVER,
            },
            // Disabled (e.g. Play All with nothing loaded) → neutral gray chip,
            // matching the contained button's disabled look.
            '&.Mui-disabled': {
                border: 'none',
                color: 'rgba(255,255,255,0.26)',
                background: 'rgba(255,255,255,0.06)',
                boxShadow: 'none',
            },
        };
    },
};

export const performanceChannelStyles = {
    strip: (color, playing) => (theme) => {
        const dark = theme.palette.mode === 'dark';
        // Liquid glass: translucent surface + backdrop blur so the panel's
        // glow bleeds through, matching the app's Paper/Card treatment.
        // Playing channels add a channel-colored tint + accent rim on top.
        const glassBg = dark ? 'rgba(38, 41, 44, 0.42)' : 'rgba(248, 243, 234, 0.45)';
        const rim = dark
            ? 'inset 0 1px 0 rgba(255,255,255,0.22), inset 0 -1px 0 rgba(0,0,0,0.32)'
            : 'inset 0 1px 0 rgba(255,255,255,0.55), inset 0 -1px 0 rgba(0,0,0,0.10)';
        return {
            display: 'flex',
            flexDirection: 'column',
            gap: 1,
            p: 1.25,
            borderRadius: 2.5,
            border: '1px solid',
            borderColor: playing ? color : (dark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.10)'),
            background: playing
                ? `linear-gradient(160deg, ${color}2E 0%, ${glassBg} 60%)`
                : glassBg,
            backdropFilter: 'blur(28px) saturate(200%)',
            WebkitBackdropFilter: 'blur(28px) saturate(200%)',
            boxShadow: playing
                ? `0 0 0 1px ${color}66, 0 8px 22px ${dark ? 'rgba(4, 8, 14, 0.5)' : 'rgba(0,0,0,0.15)'}, ${rim}`
                : `0 2px 10px ${dark ? 'rgba(4, 8, 14, 0.34)' : 'rgba(0,0,0,0.08)'}, ${rim}`,
            transition: 'box-shadow 0.2s ease, border-color 0.2s ease, background 0.3s ease',
            height: '100%',
            minWidth: 150,
        };
    },
    stripHeader: (color) => ({
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        // Pinned to the same height the master header uses, so both dividers
        // align. ≈ the badge's natural height, so channels don't shift.
        minHeight: perfTokens.height.compact,
        borderBottom: `1px solid ${color}33`,
        pb: 0.75,
    }),
    channelBadge: (color) => ({
        fontFamily: 'inherit',
        fontSize: perfTokens.fontSize.md,
        fontWeight: 700,
        // Solid channel color with a sheen overlay + raised shadow so the
        // badge reads as a filled gradient pad, not a flat tinted box.
        color: 'rgba(0,0,0,0.85)',
        letterSpacing: perfTokens.letterSpacing.wide,
        px: 0.85,
        py: 0.25,
        borderRadius: 1,
        backgroundColor: color,
        backgroundImage: SHEEN_DARK,
        border: `1px solid ${color}`,
        boxShadow: RAISE_DARK,
    }),
    muteSoloRow: {
        display: 'flex',
        gap: 0.5,
    },
    muteBtn: (active) => (theme) => {
        const c = theme.palette.error.main;
        return {
            width: perfTokens.height.sub,
            height: perfTokens.height.sub,
            fontSize: perfTokens.fontSize.sm,
            fontWeight: 700,
            borderRadius: 1,
            color: active ? '#fff' : 'text.secondary',
            backgroundColor: active ? `${c}D9` : 'transparent',   // D9 ≈ 85%
            backgroundImage: active ? SHEEN_DARK : 'none',
            border: '1px solid',
            borderColor: active ? `${c}D9` : 'divider',
            boxShadow: active ? RAISE_DARK : 'none',
            '&:hover': {
                backgroundColor: active ? c : `${c}2E`,            // 2E ≈ 18%
            },
        };
    },
    soloBtn: (active) => (theme) => {
        const c = theme.palette.warning.main;
        return {
            width: perfTokens.height.sub,
            height: perfTokens.height.sub,
            fontSize: perfTokens.fontSize.sm,
            fontWeight: 700,
            borderRadius: 1,
            color: active ? '#0c1018' : 'text.secondary',
            backgroundColor: active ? `${c}F2` : 'transparent',   // F2 ≈ 95%
            backgroundImage: active ? SHEEN_DARK : 'none',
            border: '1px solid',
            borderColor: active ? `${c}F2` : 'divider',
            boxShadow: active ? RAISE_DARK : 'none',
            '&:hover': {
                backgroundColor: active ? c : `${c}33`,            // 33 ≈ 20%
            },
        };
    },
    promptBox: {
        display: 'flex',
        flexDirection: 'column',
        gap: 0.5,
    },
    promptField: (theme) => ({
        '& .MuiOutlinedInput-root': {
            // xs — smaller than the bar pills so the prompt doesn't visually
            // dominate the channel strip when the user types a long line.
            fontSize: perfTokens.fontSize.xs,
            backgroundColor: theme.palette.mode === 'dark' ? 'rgba(9, 12, 18, 0.5)' : 'rgba(0, 0, 0, 0.04)',
            borderRadius: 1.5,
            // Capped at 2 visible lines (set via maxRows on the TextField);
            // !important wins against MUI's textarea inline `overflow: hidden`
            // so additional lines scroll inside the field instead of getting
            // clipped silently.
            '& textarea': {
                lineHeight: 1.4,
                overflowY: 'auto !important',
            },
        },
        '& fieldset': { borderColor: theme.palette.divider },
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
    durationSlider: (color) => ({
        flex: 1,
        color,
        '& .MuiSlider-thumb': { width: 10, height: 10 },
        '& .MuiSlider-rail': { opacity: 0.3 },
    }),
    // Channel CTA-row variant of pillControl — same as the bar pills but
    // taller (matches `cta` height so it sits flush with the Generate
    // pill next to it). Use for the Takes select.
    channelPillControl: {
        '& .MuiOutlinedInput-root': {
            fontFamily: 'inherit',
            height: perfTokens.height.cta,
            borderRadius: 1.5,
        },
        '& .MuiSelect-select': {
            fontFamily: 'inherit',
            padding: '0 28px 0 10px !important',
            lineHeight: `${perfTokens.height.cta}px`,
            fontSize: perfTokens.fontSize.sm,
            fontWeight: perfTokens.weight.bold,
            fontVariantNumeric: 'tabular-nums',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
        },
    },
    // Wide Generate pill — primary CTA on the channel. While generating,
    // a sibling Box fills left-to-right based on /api/generation-progress
    // (rendered by the parent component so this style stays static). The
    // label (`Generate ➝` / `Generating · 47%`) sits flush against the
    // right edge of the pill via `justifyContent: 'flex-end'` so the arrow
    // visually punctuates the action's destination.
    generatePill: (color, { generating, disabled }) => {
        const base = {
            position: 'relative',
            overflow: 'hidden',
            // Fills whatever space the wrapping span gives it. Sizing lives
            // on the parent row (the pill is a flex child via the wrapper
            // span), so this stays free of magic widths and the right edge
            // tracks the row's right edge instead of poking past it.
            boxSizing: 'border-box',
            width: '100%',
            height: perfTokens.height.cta,
            px: 1.5,
            borderRadius: 1.5,
            border: 'none',
            fontSize: perfTokens.fontSize.sm,
            // Match the app's buttons (MuiButton root is 400) — not bold.
            fontWeight: 400,
            textTransform: 'none',
            cursor: disabled ? 'not-allowed' : 'pointer',
            transition: 'box-shadow 200ms cubic-bezier(0.16,1,0.3,1), transform 200ms cubic-bezier(0.16,1,0.3,1), filter 120ms, background-color 120ms, color 120ms',
        };
        if (disabled) {
            // Neutral gray chip — matches the contained button's disabled look.
            return {
                ...base,
                background: 'rgba(255,255,255,0.06)',
                color: 'rgba(255,255,255,0.26)',
                boxShadow: 'none',
            };
        }
        if (generating) {
            // Translucent track so the progress fill inside still reads.
            return {
                ...base,
                backgroundColor: `${color}2E`,
                color,
                boxShadow: PERF_BTN_SHADOW,
            };
        }
        // Idle + enabled — gradient fill + drop shadow, no border, matching the
        // "Suggest hyperparameters" contained button.
        return {
            ...base,
            background: `${PERF_BTN_SHEEN}, ${color}`,
            color: 'rgba(0,0,0,0.88)',
            boxShadow: PERF_BTN_SHADOW,
            '&:hover': {
                filter: 'brightness(1.05)',
                transform: 'translateY(-1px)',
                boxShadow: PERF_BTN_SHADOW_HOVER,
            },
        };
    },
    // The animated fill that lives INSIDE generatePill while generating.
    // Width is the only dynamic field — parent passes a 0-100 number.
    generatePillFill: (color, progressPct) => ({
        position: 'absolute',
        left: 0,
        top: 0,
        bottom: 0,
        width: `${Math.max(0, Math.min(100, progressPct))}%`,
        backgroundColor: `${color}33`,
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
    waveformWrap: (theme) => ({
        position: 'relative',
        height: 42,
        backgroundColor: theme.palette.mode === 'dark' ? 'rgba(9, 12, 18, 0.6)' : 'rgba(0, 0, 0, 0.06)',
        borderRadius: 1.5,
        border: '1px solid',
        borderColor: theme.palette.divider,
        overflow: 'hidden',
    }),
    // Channel fragment history strip — persistent per-channel rolling list
    // of the last N fragments (default 50). Always visible (empty-state
    // included) so the user knows it's there. Oldest at top, newest at the
    // bottom; scrolls vertically when the list grows past ~4 visible rows.
    fragmentHistoryPanel: {
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1.25,
        overflow: 'hidden',
        bgcolor: 'background.paper',
    },
    fragmentHistoryHeader: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        px: 0.75,
        py: 0.25,
        minHeight: 22,
        borderBottom: '1px solid',
        borderColor: 'divider',
    },
    fragmentHistoryHeaderText: {
        ...perfTokens.caps,
        color: 'text.secondary',
    },
    fragmentHistoryHeaderBtn: {
        width: 18,
        height: 18,
        color: 'text.disabled',
        '&:hover': {
            color: 'error.main',
            bgcolor: 'action.hover',
        },
    },
    // Each fragment row is 28px tall (minHeight 24 + 2+2 padding); 4 rows +
    // 3 interior 1px borders = ~115px. Fix the panel body to that height
    // so the channel strip never resizes as fragments accumulate — past 4
    // fragments the strip scrolls internally instead of growing.
    fragmentHistoryList: {
        height: 3 * 28 + 2,
        overflowY: 'auto',
    },
    fragmentHistoryEmpty: {
        ...perfTokens.caps,
        textAlign: 'center',
        color: 'text.disabled',
        // Match the 3-row list height so the panel doesn't resize when
        // fragments appear / disappear.
        height: 3 * 28 + 2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    fragmentRow: (color, isCommitted, isAuditioning) => ({
        display: 'flex',
        alignItems: 'center',
        gap: 0.25,
        px: 0.5,
        py: 0.25,
        minHeight: 24,
        bgcolor: isCommitted ? `${color}14` : 'transparent',
        borderLeft: isCommitted ? `2px solid ${color}` : '2px solid transparent',
        '&:not(:last-child)': {
            borderBottom: '1px solid',
            borderColor: 'divider',
        },
        '&:hover': {
            bgcolor: isAuditioning
                ? `${color}26`
                : isCommitted
                    ? `${color}1F`
                    : 'action.hover',
        },
        transition: 'background-color 120ms',
    }),
    // fragmentIconBtn — used by audition, star, and commit buttons in the
    // fragment history rows. The optional `filled` flag flips the active
    // state from a ghost (color + transparent bg) to a chip (white glyph on
    // channel-color bg + 1px ring). The commit button uses `filled` so the
    // "loaded" fragment reads as a definitively selected pill, not just a
    // tinted icon.
    fragmentIconBtn: (color, active, filled = false) => ({
        width: 20,
        height: 20,
        borderRadius: 0.75,
        color: filled && active ? '#0c1018' : (active ? color : 'text.disabled'),
        backgroundColor: filled && active ? color : 'transparent',
        border: filled && active ? `1px solid ${color}` : '1px solid transparent',
        '&:hover': filled && active
            ? { backgroundColor: color, color: '#0c1018' }
            : { color, bgcolor: 'action.hover' },
        '&.Mui-disabled': filled && active
            ? { color: '#0c1018', backgroundColor: color, opacity: 1 }
            : { color, opacity: 0.85 },
    }),
    fragmentDeleteBtn: {
        width: 20,
        height: 20,
        borderRadius: 0.75,
        color: 'text.disabled',
        '&:hover': { color: 'error.main', bgcolor: 'action.hover' },
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
        ...perfTokens.num,
        fontWeight: perfTokens.weight.bold,
        color: 'text.primary',
        flexShrink: 0,
    },
    fragmentDuration: {
        ...perfTokens.num,
        color: 'text.disabled',
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
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 0.5,
        py: 0.5,
    },
    knobCell: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 0.25,
        height: 70,
    },
    knobSlider: (color, fat = false) => ({
        height: 50,
        color,
        // Gain is visually fattened (wider track + bigger thumb) so it stands
        // out from LPF/DLY/REV — it's the dBFS-scaled "fader" of the four.
        '& .MuiSlider-thumb': { width: fat ? 12 : 10, height: fat ? 12 : 10 },
        '& .MuiSlider-rail': { opacity: 0.3, width: fat ? 4 : 2 },
        '& .MuiSlider-track': { width: fat ? 4 : 2, border: 'none' },
    }),
    knobLabel: {
        ...perfTokens.caps,
        display: 'block',
        color: 'text.secondary',
        mt: 0.75,
    },
    transportRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 0.5,
        mt: 'auto',
        pt: 0.75,
        borderTop: '1px solid',
        borderTopColor: 'divider',
    },
    transportBtn: (color, playing) => (theme) => ({
        width: perfTokens.height.compact,
        height: perfTokens.height.compact,
        borderRadius: 1.5,
        color: playing ? '#0c1018' : color,
        backgroundColor: playing ? color : `${color}14`,
        backgroundImage: playing ? SHEEN_DARK : 'none',
        border: `1px solid ${color}${playing ? '' : '55'}`,
        boxShadow: playing ? RAISE_DARK : 'none',
        '&:hover': { backgroundColor: playing ? color : `${color}28` },
        '&.Mui-disabled': theme.palette.mode === 'dark' ? { opacity: 0.3 } : {},
    }),
    // Channel Loop toggle — literal "L" glyph at weight heavy. State cue
    // is two-axis so color-blind users always have a non-chromatic signal:
    //   Off → 1px divider border, transparent bg, dim "L"
    //   On  → 1px channel-color border + 1px inset shadow (= 2px ring
    //         without layout shift), tinted fill, full-color bold "L",
    //         AND a 4×4px LED dot in the bottom-right corner.
    // Filled chip when active, gray ghost when off — same pattern as the
    // committed-take ✓ button in the take history list. The filled bg now
    // carries the active signal on its own, so the old inset ring + LED dot
    // are gone (was visually noisy on top of the filled state).
    loopBtn: (color, active) => ({
        width: perfTokens.height.sub,
        height: perfTokens.height.sub,
        borderRadius: 1,
        fontSize: perfTokens.fontSize.sm,
        fontWeight: perfTokens.weight.heavy,
        lineHeight: 1,
        color: active ? '#0c1018' : 'text.disabled',
        backgroundColor: active ? color : 'transparent',
        backgroundImage: active ? SHEEN_DARK : 'none',
        border: '1px solid',
        borderColor: active ? color : 'divider',
        boxShadow: active ? RAISE_DARK : 'none',
        transition: 'background-color 120ms, color 120ms, border-color 120ms, box-shadow 120ms',
        '&:hover': {
            backgroundColor: active ? color : 'action.hover',
            color: active ? '#0c1018' : 'text.secondary',
        },
    }),
    meterTrack: (theme) => ({
        flex: 1,
        height: 12,
        mt: 1,
        backgroundColor: theme.palette.mode === 'dark' ? 'rgba(9, 12, 18, 0.7)' : 'rgba(0, 0, 0, 0.08)',
        borderRadius: 0.75,
        border: '1px solid',
        borderColor: theme.palette.divider,
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
    }),
    meterFill: (color) => ({
        width: '0%',
        height: '100%',
        background: `linear-gradient(90deg, ${color} 0%, ${color}AA 70%, #E36C61 100%)`,
        transition: 'width 0.05s linear',
    }),
};

export default theme;
