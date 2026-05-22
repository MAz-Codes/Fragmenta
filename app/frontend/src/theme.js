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

const FONT_BODY  = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif';
const FONT_MONO  = '"JetBrains Mono", "IBM Plex Mono", ui-monospace, Menlo, monospace';
// Display face used for Tier-1 section titles + Tier-2 accordion labels —
// gives the cards a strong, distinctive header voice that doesn't compete
// with body Inter. Variable weights 400–800 available; choose per-variant.
const FONT_DISPLAY = '"Archivo", "Inter", system-ui, sans-serif';

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
    amber:     '#D4A24A',      // primary accent — Arcade's signature
    amberHi:   '#E2B559',      // hover / lighter
    amberLo:   '#A37C30',      // pressed / darker
    blue:      '#5BA9E8',      // secondary — selected-file cue
    blueDim:   '#84BFEE',
    success:   '#7AC795',
    error:     '#E26B5E',
    warning:   '#E3A34B',
};

// --- Paper (light) palette --------------------------------------------------
const LIGHT = {
    bg:        '#EAE3D2',      // the user's exact cream
    bgElev:    '#EFE9D9',
    paper:     '#F2EBD9',
    paperHi:   '#F7F1E2',
    divider:   'rgba(43, 31, 18, 0.16)',
    text:      '#2B1F12',      // warm dark brown
    textDim:   '#6E5E48',
    textFaint: '#9C8B70',
    amber:     '#9E7228',      // deeper gold for contrast on cream
    amberHi:   '#C49350',
    amberLo:   '#6E4F17',
    blue:      '#3B6E9B',
    blueDim:   '#5E8FB8',
    success:   '#2E8A52',
    error:     '#B84E45',
    warning:   '#B47318',
};

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
    },
    shape: {
        borderRadius: 10,
    },
    typography: {
        fontFamily: FONT_BODY,
        h1: { fontWeight: 700, letterSpacing: '-0.02em' },
        h2: { fontWeight: 700, letterSpacing: '-0.02em' },
        h3: { fontWeight: 600, letterSpacing: '-0.015em' },
        h4: { fontWeight: 600, letterSpacing: '-0.01em' },
        h5: { fontWeight: 600, letterSpacing: '-0.005em' },
        // Tier-1 section card titles — Archivo at heavy weight for display
        // presence without the extreme density of Archivo Black.
        h6: { fontFamily: FONT_DISPLAY, fontWeight: 700, letterSpacing: 0, fontSize: '1.05rem' },
        // Tier-2 section/accordion labels — same Archivo face, same heavy
        // weight, slightly smaller. "Annotator Labels", "Advanced Settings",
        // "Edit existing audio" all flow through here.
        subtitle1: { fontFamily: FONT_DISPLAY, fontWeight: 700, letterSpacing: 0, fontSize: '0.95rem', textTransform: 'none' },
        subtitle2: { fontWeight: 500, letterSpacing: 0,         fontSize: '0.825rem', textTransform: 'uppercase' },
        body1: { fontWeight: 400, letterSpacing: '-0.005em',    fontSize: '0.925rem' },
        body2: { fontWeight: 400, letterSpacing: '-0.005em',    fontSize: '0.825rem' },
        button: { fontWeight: 500, letterSpacing: '0.01em',     textTransform: 'none' },
        caption: { fontWeight: 400, letterSpacing: '0.005em',   fontSize: '0.75rem' },
        overline: { fontWeight: 600, letterSpacing: '0.12em',   textTransform: 'uppercase', fontSize: '0.7rem' },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                ':root': { colorScheme: 'dark' },
                body: {
                    margin: 0,
                    minHeight: '100vh',
                    // Neutral charcoal — the amber accent does the warmth.
                    // A faint diagonal sheen for life; no coloured glows.
                    backgroundColor: DARK.bg,
                    backgroundImage:
                        `linear-gradient(170deg, ${DARK.bg} 0%, #1B1C1D 45%, ${DARK.bg} 100%)`,
                    color: DARK.text,
                    fontFeatureSettings: '"cv11", "ss01", "ss03"',  // Inter stylistic alts
                },
                '#root': { minHeight: '100vh' },
                '*::-webkit-scrollbar': { width: '10px', height: '10px' },
                '*::-webkit-scrollbar-track': {
                    background: 'rgba(212, 162, 74, 0.06)',
                    borderRadius: '999px',
                },
                '*::-webkit-scrollbar-thumb': {
                    background: 'rgba(212, 162, 74, 0.32)',
                    borderRadius: '999px',
                    border: '2px solid rgba(0, 0, 0, 0)',
                    backgroundClip: 'padding-box',
                    '&:hover': { background: 'rgba(212, 162, 74, 0.52)' },
                },
                '*::-webkit-scrollbar-corner': { background: 'transparent' },
                '*': {
                    scrollbarWidth: 'thin',
                    scrollbarColor: 'rgba(212, 162, 74, 0.32) rgba(212, 162, 74, 0.06)',
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundColor: DARK.paper,
                    backgroundImage:
                        `linear-gradient(180deg, ${DARK.paper} 0%, ${DARK.bgElev} 100%)`,
                    border: `1px solid ${DARK.divider}`,
                    boxShadow: '0 20px 38px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.02)',
                    backdropFilter: 'blur(10px)',
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: DARK.paper,
                    backgroundImage:
                        `linear-gradient(180deg, ${DARK.paper} 0%, ${DARK.bgElev} 100%)`,
                    border: `1px solid ${DARK.divider}`,
                    boxShadow: '0 10px 22px rgba(0, 0, 0, 0.45)',
                    transition: 'border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease',
                    '&:hover': {
                        borderColor: 'rgba(212, 162, 74, 0.4)',
                        boxShadow: '0 16px 32px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(212, 162, 74, 0.18)',
                        transform: 'translateY(-1px)',
                    },
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    borderRadius: 999,                 // pill — instrument-like
                    fontWeight: 500,
                    paddingInline: 18,
                    lineHeight: 1.2,
                    letterSpacing: '0.01em',
                    transition: 'transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, background-color 160ms ease, color 160ms ease',
                },
                contained: {
                    boxShadow: '0 6px 16px rgba(0, 0, 0, 0.4)',
                    '&:hover': {
                        boxShadow: '0 10px 22px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(212, 162, 74, 0.4)',
                        transform: 'translateY(-1px)',
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
                    borderColor: 'rgba(240, 237, 229, 0.18)',
                    color: DARK.text,
                    '&:hover': {
                        borderColor: DARK.amber,
                        backgroundColor: 'rgba(212, 162, 74, 0.08)',
                        color: DARK.amberHi,
                    },
                },
                text: {
                    color: DARK.text,
                    '&:hover': {
                        backgroundColor: 'rgba(212, 162, 74, 0.08)',
                        color: DARK.amberHi,
                    },
                },
            },
        },
        MuiInputBase: {
            styleOverrides: {
                root: {
                    '&:not(.MuiInputBase-multiline)': { alignItems: 'center' },
                },
                input: { lineHeight: 1.4 },
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
                            boxShadow: '0 0 0 3px rgba(212, 162, 74, 0.12)',
                        },
                    },
                },
            },
        },
        MuiSelect: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(10, 8, 6, 0.5)',
                    borderRadius: 8,
                    '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(240, 237, 229, 0.14)' },
                    '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(240, 237, 229, 0.32)' },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: DARK.amber,
                        boxShadow: '0 0 0 3px rgba(212, 162, 74, 0.12)',
                    },
                },
                select: { display: 'flex', alignItems: 'center' },
            },
        },
        MuiMenuItem: {
            styleOverrides: {
                root: {
                    backgroundColor: DARK.paper,
                    '&:hover': { backgroundColor: DARK.paperHi },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(212, 162, 74, 0.14)',
                        color: DARK.amberHi,
                        '&:hover': { backgroundColor: 'rgba(212, 162, 74, 0.20)' },
                    },
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(240, 237, 229, 0.06)',
                    color: DARK.text,
                    border: `1px solid rgba(240, 237, 229, 0.12)`,
                    borderRadius: 999,            // pill, matches button language
                    '&.MuiChip-colorPrimary': {
                        backgroundColor: 'rgba(212, 162, 74, 0.16)',
                        color: DARK.amberHi,
                        borderColor: 'rgba(212, 162, 74, 0.4)',
                    },
                    '&.MuiChip-colorSuccess': {
                        backgroundColor: 'rgba(122, 199, 149, 0.16)',
                        color: '#9FDDB5',
                        borderColor: 'rgba(122, 199, 149, 0.4)',
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
                    backgroundColor: DARK.paper,
                    border: `1px solid ${DARK.divider}`,
                    borderRadius: 10,
                    overflow: 'hidden',
                    '&:before': { display: 'none' },
                    '&.Mui-expanded': { margin: 0 },
                },
            },
        },
        MuiAccordionSummary: {
            styleOverrides: {
                root: {
                    backgroundColor: DARK.paperHi,
                    borderRadius: 10,
                    minHeight: 44,
                    '& .MuiAccordionSummary-content': { margin: '10px 0', alignItems: 'center' },
                    '&.Mui-expanded': { minHeight: 44 },
                    '&.Mui-expanded .MuiAccordionSummary-content': { margin: '10px 0' },
                    '&:hover': { backgroundColor: '#2B2519' },
                },
            },
        },
        MuiDialog: {
            styleOverrides: {
                paper: {
                    backgroundColor: DARK.paper,
                    backgroundImage:
                        `linear-gradient(180deg, ${DARK.paper} 0%, ${DARK.bgElev} 100%)`,
                    border: `1px solid ${DARK.divider}`,
                    borderRadius: 14,
                    boxShadow: '0 32px 60px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(212, 162, 74, 0.04)',
                    backdropFilter: 'blur(14px)',
                },
            },
        },
        MuiDialogTitle: {
            styleOverrides: {
                root: {
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
                    '&:hover': { backgroundColor: 'rgba(212, 162, 74, 0.06)' },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(212, 162, 74, 0.14)',
                        '&:hover': { backgroundColor: 'rgba(212, 162, 74, 0.20)' },
                    },
                },
            },
        },
        MuiCheckbox: {
            styleOverrides: {
                root: {
                    color: DARK.textDim,
                    '&.Mui-checked': { color: DARK.amber },
                    '&:hover': { backgroundColor: 'rgba(212, 162, 74, 0.08)' },
                },
            },
        },
        MuiFormControlLabel: {
            styleOverrides: { label: { color: DARK.text, fontSize: '0.875rem' } },
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
                        boxShadow: '0 0 0 8px rgba(212, 162, 74, 0.18)',
                    },
                    '&.Mui-active': {
                        boxShadow: '0 0 0 12px rgba(212, 162, 74, 0.24)',
                    },
                },
                valueLabel: {
                    backgroundColor: DARK.paper,
                    color: DARK.text,
                    border: `1px solid ${DARK.divider}`,
                    borderRadius: 6,
                    fontWeight: 500,
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
                bar: {
                    backgroundImage: `linear-gradient(90deg, ${DARK.amberLo} 0%, ${DARK.amber} 100%)`,
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
                    color: DARK.textDim,
                    textTransform: 'none',
                    fontWeight: 500,
                    letterSpacing: '-0.005em',
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
                    transition: 'all 160ms ease',
                    '&:hover': {
                        backgroundColor: 'rgba(212, 162, 74, 0.10)',
                        color: DARK.amberHi,
                    },
                },
            },
        },
        MuiToggleButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    border: `1px solid ${DARK.divider}`,
                    color: DARK.textDim,
                    fontWeight: 500,
                    letterSpacing: '-0.005em',
                    '&:hover': {
                        backgroundColor: 'rgba(212, 162, 74, 0.06)',
                        color: DARK.text,
                    },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(212, 162, 74, 0.14)',
                        color: DARK.amberHi,
                        borderColor: DARK.amber,
                        '&:hover': { backgroundColor: 'rgba(212, 162, 74, 0.20)' },
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
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                ':root': { colorScheme: 'light' },
                body: {
                    margin: 0,
                    minHeight: '100vh',
                    // Solid warm cream with a very subtle radial warm spot
                    // top-left for life. Paper apps don't need gradients.
                    backgroundColor: LIGHT.bg,
                    backgroundImage:
                        `radial-gradient(1200px 600px at 6% -10%, rgba(158, 114, 40, 0.06), transparent 55%), ` +
                        `linear-gradient(180deg, ${LIGHT.bg} 0%, #E6DECB 100%)`,
                    color: LIGHT.text,
                    fontFeatureSettings: '"cv11", "ss01", "ss03"',
                },
                '#root': { minHeight: '100vh' },
                '*::-webkit-scrollbar-track': {
                    background: 'rgba(158, 114, 40, 0.06)',
                    borderRadius: '999px',
                },
                '*::-webkit-scrollbar-thumb': {
                    background: 'rgba(158, 114, 40, 0.30)',
                    borderRadius: '999px',
                    '&:hover': { background: 'rgba(158, 114, 40, 0.48)' },
                },
                '*::-webkit-scrollbar-corner': { background: 'transparent' },
                '*': {
                    scrollbarWidth: 'thin',
                    scrollbarColor: 'rgba(158, 114, 40, 0.30) rgba(158, 114, 40, 0.06)',
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundColor: LIGHT.paper,
                    backgroundImage:
                        `linear-gradient(180deg, ${LIGHT.paper} 0%, ${LIGHT.bgElev} 100%)`,
                    border: `1px solid ${LIGHT.divider}`,
                    boxShadow: '0 8px 22px rgba(43, 31, 18, 0.08)',
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: LIGHT.paper,
                    backgroundImage:
                        `linear-gradient(180deg, ${LIGHT.paper} 0%, ${LIGHT.bgElev} 100%)`,
                    border: `1px solid ${LIGHT.divider}`,
                    boxShadow: '0 6px 14px rgba(43, 31, 18, 0.08)',
                    '&:hover': {
                        borderColor: 'rgba(158, 114, 40, 0.4)',
                        boxShadow: '0 12px 22px rgba(43, 31, 18, 0.12)',
                    },
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                contained: {
                    boxShadow: '0 6px 14px rgba(158, 114, 40, 0.18)',
                    '&:hover': { boxShadow: '0 10px 20px rgba(158, 114, 40, 0.26)' },
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
                    borderColor: 'rgba(43, 31, 18, 0.18)',
                    color: LIGHT.text,
                    '&:hover': {
                        borderColor: LIGHT.amber,
                        backgroundColor: 'rgba(158, 114, 40, 0.08)',
                        color: LIGHT.amberLo,
                    },
                },
                text: {
                    color: LIGHT.text,
                    '&:hover': {
                        backgroundColor: 'rgba(158, 114, 40, 0.08)',
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
                            boxShadow: '0 0 0 3px rgba(158, 114, 40, 0.14)',
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
                        boxShadow: '0 0 0 3px rgba(158, 114, 40, 0.14)',
                    },
                },
            },
        },
        MuiMenuItem: {
            styleOverrides: {
                root: {
                    backgroundColor: LIGHT.paper,
                    '&:hover': { backgroundColor: LIGHT.paperHi },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(158, 114, 40, 0.14)',
                        color: LIGHT.amberLo,
                        '&:hover': { backgroundColor: 'rgba(158, 114, 40, 0.20)' },
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
                    '&.MuiChip-colorPrimary': {
                        backgroundColor: 'rgba(158, 114, 40, 0.14)',
                        color: LIGHT.amberLo,
                        borderColor: 'rgba(158, 114, 40, 0.36)',
                    },
                    '&.MuiChip-colorSuccess': {
                        backgroundColor: 'rgba(46, 138, 82, 0.14)',
                        color: '#1F6038',
                        borderColor: 'rgba(46, 138, 82, 0.36)',
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
                    backgroundColor: LIGHT.paper,
                    border: `1px solid ${LIGHT.divider}`,
                    borderRadius: 10,
                    overflow: 'hidden',
                    '&:before': { display: 'none' },
                    '&.Mui-expanded': { margin: 0 },
                },
            },
        },
        MuiAccordionSummary: {
            styleOverrides: {
                root: {
                    backgroundColor: LIGHT.paperHi,
                    '&:hover': { backgroundColor: '#FAF4E4' },
                },
            },
        },
        MuiDialog: {
            styleOverrides: {
                paper: {
                    backgroundColor: LIGHT.paper,
                    backgroundImage: `linear-gradient(180deg, ${LIGHT.paper} 0%, ${LIGHT.bgElev} 100%)`,
                    border: `1px solid ${LIGHT.divider}`,
                    borderRadius: 14,
                    boxShadow: '0 24px 48px rgba(43, 31, 18, 0.18)',
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
                    '&:hover': { backgroundColor: 'rgba(158, 114, 40, 0.06)' },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(158, 114, 40, 0.14)',
                        '&:hover': { backgroundColor: 'rgba(158, 114, 40, 0.20)' },
                    },
                },
            },
        },
        MuiCheckbox: {
            styleOverrides: {
                root: {
                    color: LIGHT.textDim,
                    '&.Mui-checked': { color: LIGHT.amber },
                    '&:hover': { backgroundColor: 'rgba(158, 114, 40, 0.08)' },
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
                        boxShadow: '0 0 0 8px rgba(158, 114, 40, 0.18)',
                    },
                },
                valueLabel: {
                    backgroundColor: LIGHT.text,
                    color: '#FFFBF1',
                    borderRadius: 6,
                    fontWeight: 500,
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
                    backgroundImage: `linear-gradient(90deg, ${LIGHT.amber} 0%, ${LIGHT.amberHi} 100%)`,
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
                    fontWeight: 500,
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
                        backgroundColor: 'rgba(158, 114, 40, 0.10)',
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
                    fontWeight: 500,
                    '&:hover': {
                        backgroundColor: 'rgba(158, 114, 40, 0.06)',
                        color: LIGHT.text,
                    },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(158, 114, 40, 0.14)',
                        color: LIGHT.amberLo,
                        borderColor: LIGHT.amber,
                        '&:hover': { backgroundColor: 'rgba(158, 114, 40, 0.20)' },
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
        py: { xs: 1, sm: 1.5, md: 2.5 },
        px: { xs: 0.75, sm: 1.25, md: 2.5 },
        minHeight: '100vh',
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
        alignItems: { xs: 'stretch', md: 'flex-start' },
        flexDirection: { xs: 'column', md: 'row' },
        gap: { xs: 1.25, sm: 1.75, md: 2 },
        mb: { xs: 1, sm: 1.5 },
    },
    headerBrand: {
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        gap: { xs: 1.25, sm: 2 },
        py: { xs: 0.25, sm: 0.5 },
    },
    logo: {
        width: 60,
        height: 60,
        backgroundImage: 'url(/fragmenta_icon_1024.png)',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        borderRadius: 2,
        border: '1px solid rgba(194, 207, 228, 0.22)',
        boxShadow: '0 10px 20px rgba(4, 8, 14, 0.36)',
        filter: 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3))',
    },
    title: {
        color: 'text.primary',
        fontFamily: '"Bitcount Single", "IBM Plex Mono", "JetBrains Mono", "Space Mono", "Courier New", monospace',
        fontWeight: 400,
        letterSpacing: '0.02em',
        textShadow: '0 2px 10px rgba(0, 0, 0, 0.6)',
    },
    headerActionsContainer: (isCompactLayout) => ({
        display: 'flex',
        alignItems: 'stretch',
        justifyContent: { xs: 'flex-start', md: 'flex-end' },
        gap: { xs: 1, sm: 1.5 },
        flexDirection: isCompactLayout ? 'column' : 'row',
        flexWrap: isCompactLayout ? 'wrap' : 'nowrap',
        width: { xs: '100%', md: 'auto' },
    }),
    headerActionsGrid: (isCompactLayout) => ({
        display: 'grid',
        gridTemplateColumns: isCompactLayout
            ? 'repeat(2, minmax(0, 1fr))'
            : 'repeat(2, 122px)',
        gap: { xs: 0.75, sm: 1 },
        justifyContent: 'flex-end',
        flex: isCompactLayout ? '1 1 auto' : '0 1 auto',
        width: isCompactLayout ? '100%' : 'auto',
    }),
    headerActionButton: {
        fontSize: { xs: '0.70rem', sm: '0.72rem' },
        height: { xs: 34, sm: 36 },
        minWidth: 0,
        width: '100%',
        px: { xs: 1, sm: 1.5 },
        '& .MuiButton-startIcon svg': {
            width: { xs: 14, sm: 15 },
            height: { xs: 14, sm: 15 },
        },
    },
    gpuCard: (isCompactLayout) => ({
        p: { xs: 1.25, sm: 1.75 },
        bgcolor: 'background.paper',
        borderRadius: 2.5,
        border: '1px solid',
        borderColor: 'divider',
        minWidth: isCompactLayout ? '100%' : 270,
        flexShrink: 0,
        position: 'relative',
        overflow: 'hidden',
        boxShadow: '0 16px 32px rgba(4, 8, 14, 0.44)',
    }),
    gpuUsageTrack: {
        position: 'relative',
        width: '100%',
        height: 6,
        bgcolor: 'rgba(157, 169, 188, 0.2)',
        borderRadius: 3,
        overflow: 'hidden',
    },
    gpuUsageFill: (width, color) => ({
        position: 'absolute',
        top: 0,
        left: 0,
        height: '100%',
        width,
        bgcolor: color,
        borderRadius: 3,
        transition: 'width 0.3s ease-in-out',
    }),
    emphasizedPrimaryBody2: {
        fontWeight: 'bold',
        color: 'primary.main',
    },
    advancedSettingsDetails: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            backgroundColor: isDark
                ? 'rgba(10, 15, 23, 0.46)'
                : 'rgba(255, 255, 255, 0.82)',
            borderTop: isDark
                ? '1px solid rgba(194, 207, 228, 0.12)'
                : '1px solid rgba(15, 23, 42, 0.08)',
            borderBottomLeftRadius: 12,
            borderBottomRightRadius: 12,
            maxHeight: { xs: 'none', md: '400px' },
            overflowY: { xs: 'visible', md: 'auto' },
            overflowX: 'hidden',
            '&::-webkit-scrollbar': {
                width: '8px',
            },
            '&::-webkit-scrollbar-track': {
                background: isDark
                    ? 'rgba(157, 169, 188, 0.14)'
                    : 'rgba(100, 116, 139, 0.14)',
                borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb': {
                background: isDark
                    ? 'rgba(157, 169, 188, 0.45)'
                    : 'rgba(100, 116, 139, 0.42)',
                borderRadius: '4px',
                '&:hover': {
                    background: isDark
                        ? 'rgba(157, 169, 188, 0.62)'
                        : 'rgba(100, 116, 139, 0.56)',
                },
            },
        };
    },
    mainLayout: {
        display: 'flex',
        flexDirection: { xs: 'column', md: 'row' },
        width: '100%',
        flex: 1,
        gap: { xs: 1, sm: 1.25, md: 1.5 },
        borderRadius: 3,
        minHeight: 0,
    },
    navPaper: {
        width: { xs: '100%', md: 64, lg: 220 },
        backgroundColor: 'background.paper',
        borderRadius: 2.5,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
    },
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
            fontSize: { xs: '0.78rem', sm: '0.86rem' },
            fontWeight: 500,
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
    mainContentPaper: (muiTheme) => ({
        flex: 1,
        backgroundColor: 'background.paper',
        borderRadius: 2.5,
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
        p: { xs: 1.5, sm: 2 },
        mb: 2,
        borderRadius: 2.5,
        transition: 'all 0.3s ease',
        '&:hover': { transform: 'translateY(-1px)' },
    },
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
    headerActionButtonWithOpacity: (isEnabled) => ({
        fontSize: { xs: '0.70rem', sm: '0.72rem' },
        height: { xs: 34, sm: 36 },
        minWidth: 0,
        width: '100%',
        px: { xs: 1, sm: 1.5 },
        opacity: isEnabled ? 1 : 0.5,
        '& .MuiButton-startIcon svg': {
            width: { xs: 14, sm: 15 },
            height: { xs: 14, sm: 15 },
        },
    }),
    gpuHeaderRow: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        mb: 1,
    },
    gpuLabel: {
        fontWeight: 500,
    },
    gpuStatusGroup: {
        display: 'flex',
        alignItems: 'center',
        gap: 0.5,
    },
    gpuStatusDot: (status, animate = true) => ({
        width: 6,
        height: 6,
        borderRadius: '50%',
        bgcolor: status === 'good'
            ? 'success.main'
            : status === 'low'
                ? 'warning.main'
                : 'error.main',
        animation: animate ? 'pulse 2s infinite' : 'none',
        '@keyframes pulse': {
            '0%': { opacity: 1 },
            '50%': { opacity: 0.5 },
            '100%': { opacity: 1 },
        },
    }),
    gpuUsageWrap: {
        mb: 1.25,
    },
    gpuFooterRow: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    gpuFreeText: {
        fontWeight: 'bold',
    },
    centeredCaption: {
        display: 'block',
        textAlign: 'center',
    },
    centeredCaptionWithMargin: {
        display: 'block',
        textAlign: 'center',
        mt: 0.5,
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
    accordionMarginBottom: {
        mb: 2,
    },
    sliderRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 2,
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
        height: 8,
        borderRadius: 4,
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
        p: { xs: 1.5, sm: 2 },
        mb: 2,
        borderRadius: 2.5,
        transition: 'all 0.3s ease',
        '&:hover': { transform: 'translateY(-1px)' },
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
    modeToggleButton: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            position: 'fixed',
            left: { xs: 12, sm: 16 },
            bottom: { xs: 58, sm: 66 },
            width: { xs: 38, sm: 42 },
            height: { xs: 38, sm: 42 },
            zIndex: 1350,
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
                background: isDark
                    ? 'linear-gradient(145deg, rgba(20, 28, 42, 1) 0%, rgba(14, 22, 34, 1) 100%)'
                    : 'linear-gradient(145deg, rgba(244, 250, 255, 1) 0%, rgba(236, 245, 252, 1) 100%)',
                transform: 'translateY(-1px)',
                boxShadow: isDark
                    ? '0 18px 28px rgba(4, 8, 14, 0.6)'
                    : '0 18px 28px rgba(15, 23, 42, 0.18)',
            },
        };
    },
    infoButton: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            position: 'fixed',
            left: { xs: 12, sm: 16 },
            bottom: { xs: 12, sm: 16 },
            width: { xs: 38, sm: 42 },
            height: { xs: 38, sm: 42 },
            zIndex: 1350,
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
                background: isDark
                    ? 'linear-gradient(145deg, rgba(20, 28, 42, 1) 0%, rgba(14, 22, 34, 1) 100%)'
                    : 'linear-gradient(145deg, rgba(244, 250, 255, 1) 0%, rgba(236, 245, 252, 1) 100%)',
                transform: 'translateY(-1px)',
                boxShadow: isDark
                    ? '0 18px 28px rgba(4, 8, 14, 0.6)'
                    : '0 18px 28px rgba(15, 23, 42, 0.18)',
            },
        };
    },
    infoDialogTitleRow: {
        display: 'inline-flex',
        alignItems: 'center',
        gap: 1,
    },
    infoDialogIntro: {
        mt: 2.5,
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
        p: { xs: 1.25, md: 2 },
        background: 'transparent',
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        overflow: 'visible',
    },
};

export const audioUploadRowStyles = {
    // Layout / motion only — MuiPaper theme override owns background, border,
    // and shadow. Avoids the old palette leaking through inline gradients.
    card: {
        mb: { xs: 1.5, sm: 2 },
        borderRadius: 2.2,
        transition: 'all 0.3s ease',
        '&:hover': {
            transform: 'translateY(-1px)',
        },
    },
    cardContent: {
        p: { xs: 1.5, sm: 2 },
        '&:last-child': {
            pb: { xs: 1.5, sm: 2 },
        },
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
    rootPaper: {
        p: 2,
        height: 240,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 2.5,
    },
    headerRow: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        mb: 2,
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
        height: '100%',
        color: 'text.secondary',
    },
    listRoot: {
        flex: 1,
        overflow: 'auto',
        maxHeight: 180,
        '& .MuiListItem-root': {
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 1.5,
            mb: 1,
            bgcolor: 'background.default',
            '&:last-child': { mb: 0 },
        },
    },
    listItem: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'stretch',
        py: 1,
    },
    fragmentRow: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        mb: 1,
    },
    fragmentMeta: {
        flex: 1,
        minWidth: 0,
    },
    fragmentPrompt: {
        fontWeight: 'bold',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        display: '-webkit-box',
        WebkitLineClamp: 2,
        WebkitBoxOrient: 'vertical',
    },
    fragmentActions: {
        display: 'flex',
        gap: 1,
        flexShrink: 0,
    },
    playPauseButton: (isPlaying) => (muiTheme) => ({
        border: '1px solid',
        borderColor: isPlaying
            ? 'primary.main'
            : (muiTheme.palette.mode === 'dark' ? 'rgba(194, 207, 228, 0.22)' : 'rgba(15, 23, 42, 0.2)'),
    }),
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
        '&:hover': { transform: 'translateY(-1px)' },
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
    errorAlert: {
        mb: 2,
    },
};

export const welcomePageStyles = {
    backdrop: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            zIndex: 9999,
            background: isDark
                ? 'radial-gradient(1200px 600px at 8% -15%, rgba(53, 194, 212, 0.2), transparent 55%), radial-gradient(800px 500px at 92% 110%, rgba(83, 193, 138, 0.16), transparent 65%), linear-gradient(160deg, #090C12 0%, #0C1119 45%, #090D13 100%)'
                : 'radial-gradient(1200px 600px at 8% -15%, rgba(20, 151, 168, 0.16), transparent 55%), radial-gradient(800px 500px at 92% 110%, rgba(72, 171, 118, 0.14), transparent 65%), linear-gradient(160deg, #F4FAFF 0%, #EAF4FF 45%, #F8FCFF 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            p: { xs: 2, md: 4 },
            cursor: 'pointer',
        };
    },
    panel: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            textAlign: 'center',
            width: 'min(920px, 100%)',
            border: isDark
                ? '1px solid rgba(194, 207, 228, 0.2)'
                : '1px solid rgba(15, 23, 42, 0.12)',
            borderRadius: 4,
            background: isDark
                ? 'linear-gradient(170deg, rgba(19, 27, 41, 0.95) 0%, rgba(12, 19, 31, 0.96) 100%)'
                : 'linear-gradient(170deg, rgba(255, 255, 255, 0.98) 0%, rgba(242, 248, 255, 0.98) 100%)',
            boxShadow: isDark
                ? '0 32px 56px rgba(4, 8, 14, 0.64)'
                : '0 24px 46px rgba(15, 23, 42, 0.2)',
            backdropFilter: 'blur(10px)',
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
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            borderRadius: 3,
            border: isDark
                ? '1px solid rgba(194, 207, 228, 0.24)'
                : '1px solid rgba(15, 23, 42, 0.12)',
            boxShadow: isDark
                ? '0 16px 28px rgba(4, 8, 14, 0.45)'
                : '0 12px 24px rgba(15, 23, 42, 0.22)',
            filter: isDark
                ? 'drop-shadow(0 8px 16px rgba(0, 0, 0, 0.4))'
                : 'drop-shadow(0 6px 12px rgba(15, 23, 42, 0.22))',
            mx: 'auto',
            mb: 1.5,
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
    overline: {
        color: 'primary.main',
        letterSpacing: { xs: '0.12em', md: '0.18em' },
        fontWeight: 700,
        fontSize: { xs: '0.62rem', sm: '0.7rem' },
    },
    footer: {
        color: 'text.secondary',
        opacity: 0.6,
        fontSize: { xs: '0.64rem', sm: '0.7rem' },
        marginTop: 5,
    },
    version: {
        color: 'text.secondary',
        opacity: 0.6,
        fontSize: { xs: '0.64rem', sm: '0.7rem' },
        fontStyle: 'italic',
    },
    ctaButton: {
        mt: 3,
        mb: 2,
        px: { xs: 3.25, md: 4.5 },
        py: { xs: 1.2, md: 1.5 },
        borderRadius: 2,
        textTransform: 'none',
        fontSize: { xs: '0.98rem', sm: '1.05rem', md: '1.1rem' },
        fontWeight: 500,
    },
};

export const checkpointManagerStyles = {
    root: {
        mt: 2,
        pt: 2,
        borderTop: '1px solid',
        borderColor: 'divider',
    },
    panelPaper: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            p: 2,
            mb: 2,
            boxShadow: isDark
                ? '0 16px 30px rgba(4, 8, 14, 0.48)'
                : '0 16px 30px rgba(15, 23, 42, 0.1)',
            borderRadius: 2.5,
            border: isDark
                ? '1px solid rgba(194, 207, 228, 0.16)'
                : '1px solid rgba(15, 23, 42, 0.12)',
            background: isDark
                ? 'linear-gradient(160deg, rgba(17, 24, 37, 0.96) 0%, rgba(13, 20, 31, 0.92) 100%)'
                : 'linear-gradient(160deg, rgba(255, 255, 255, 0.99) 0%, rgba(245, 250, 255, 0.98) 100%)',
            '&:hover': {
                boxShadow: isDark
                    ? '0 22px 38px rgba(4, 8, 14, 0.58)'
                    : '0 22px 38px rgba(15, 23, 42, 0.14)',
                transform: 'translateY(-1px)',
                transition: 'all 0.3s ease',
            },
            transition: 'all 0.3s ease',
        };
    },
    checkpointsList: {
        mt: 1,
        display: 'grid',
        gap: 1,
    },
    checkpointCard: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            mb: 1,
            p: 1.25,
            boxShadow: isDark
                ? '0 10px 20px rgba(4, 8, 14, 0.34)'
                : '0 10px 18px rgba(15, 23, 42, 0.08)',
            borderRadius: 1.75,
            border: isDark
                ? '1px solid rgba(194, 207, 228, 0.16)'
                : '1px solid rgba(15, 23, 42, 0.12)',
            background: isDark
                ? 'linear-gradient(180deg, rgba(18, 25, 38, 0.98) 0%, rgba(15, 22, 34, 0.96) 100%)'
                : 'linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(248, 251, 255, 0.99) 100%)',
            '&:hover': {
                boxShadow: isDark
                    ? '0 14px 26px rgba(4, 8, 14, 0.44)'
                    : '0 16px 28px rgba(15, 23, 42, 0.12)',
                transform: 'translateY(-1px)',
                transition: 'all 0.2s ease',
            },
            transition: 'all 0.2s ease',
        };
    },
    checkpointRow: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 1,
    },
    checkpointInfo: {
        flex: 1,
    },
    checkpointName: {
        fontWeight: 600,
        display: 'flex',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 0.75,
    },
    unwrappedChip: {
        fontSize: '0.7rem',
    },
    emptyText: {
        display: 'block',
        mt: 0.5,
    },
    metaNext: {
        ml: 1,
    },
    actions: {
        display: 'flex',
        gap: 1,
        flexWrap: 'wrap',
        justifyContent: 'flex-end',
    },
    errorAlert: {
        mt: 2,
    },
    deleteDialogText: {
        mt: 1.5,
    },
    snackbarAlert: {
        width: '100%',
    },
};

export const hfAuthDialogStyles = {
    checkingBox: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        my: 4,
    },
    checkingProgress: {
        mb: 2,
    },
    authStepBox: {
        mt: 2,
    },
    downloadStepBox: {
        mt: 2,
        textAlign: 'center',
    },
    downloadProgress: {
        mb: 2,
        height: 10,
        borderRadius: 5,
    },
    successStepBox: {
        mt: 2,
        textAlign: 'center',
    },
    stepper: {
        mb: 4,
    },
    errorAlert: {
        mb: 2,
    },
    loginSpinnerSize: 24,
};

export const modelUnwrapButtonStyles = {
    root: {
        mt: 1,
    },
    result: {
        mt: 0.5,
    },
    error: {
        color: '#DB5044',
        mt: 0.5,
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
// One source of truth for the size/spacing/height scale so similar elements
// match. The previous code carried 9+ distinct font sizes and 5+ letter
// spacings across these surfaces; anything new should pick from this set.
export const perfTokens = {
    fontSize: {
        knob: '0.58rem',   // knob labels, pan label, master peak readout
        small: '0.66rem',  // small labels: BPM unit, mute/solo, durationLabel, footer notes
        body: '0.72rem',   // primary text: buttons, dropdowns, prompt field, mapping rows
        badge: '0.78rem',  // section badges (MASTER, channel numbers)
    },
    letterSpacing: {
        wide: '0.08em',    // uppercase labels and badges
    },
    height: {
        compact: 26,       // primary compact controls (Link, MIDI, Q, BPM, transport, generate)
        sub: 22,           // small subordinate square buttons (mute, solo, loop)
    },
    // Sharp 2px radius is the deliberate Ableton-style accent on Link/MIDI.
    // Everything else lives on the MUI scale via shape.borderRadius (= 1.5).
};

export const performancePanelStyles = {
    root: {
        display: 'flex',
        flexDirection: 'column',
        gap: 1.5,
        width: '100%',
        minHeight: 0,
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
        fontSize: perfTokens.fontSize.body,
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
        gridTemplateColumns: 'repeat(4, minmax(150px, 1fr))',
        gap: 1.25,
        flex: 1,
        minWidth: 0,
    },
    masterStrip: (color) => (theme) => ({
        display: 'flex',
        flexDirection: 'column',
        gap: 1.5,
        p: 1.25,
        borderRadius: 2.5,
        border: `1px solid ${color}55`,
        bgcolor: 'background.paper',
        backgroundImage: `linear-gradient(160deg, ${color}14 0%, ${theme.palette.background.paper} 70%)`,
        boxShadow: theme.palette.mode === 'dark'
            ? `0 8px 22px rgba(0, 0, 0, 0.55), inset 0 0 0 1px ${color}22`
            : `0 6px 14px rgba(43, 31, 18, 0.10), inset 0 0 0 1px ${color}22`,
        width: { xs: '100%', md: 160 },
        flex: { xs: '1 1 100%', md: '0 0 160px' },
        minHeight: 0,
        overflow: 'hidden',
        boxSizing: 'border-box',
    }),
    masterHeader: (color) => ({
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 0.5,
        borderBottom: `1px solid ${color}33`,
        pb: 0.75,
        color,
    }),
    masterBadge: (color) => ({
        fontSize: perfTokens.fontSize.badge,
        fontWeight: 600,
        color,
        letterSpacing: perfTokens.letterSpacing.wide,
        px: 0.75,
        py: 0.25,
        borderRadius: 1,
        backgroundColor: `${color}1F`,
        border: `1px solid ${color}55`,
    }),
    masterFaderWrap: {
        flex: 1,
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'stretch',
        justifyContent: 'center',
        gap: 1,
        py: 1.25,
        minHeight: 0,
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
        '& .MuiSlider-track': { display: 'none' },
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
        textAlign: 'center',
        color: 'primary.main',
        fontSize: perfTokens.fontSize.small,
        letterSpacing: '0.04em',
    },
    masterPeakValue: {
        textAlign: 'center',
        color: 'text.disabled',
        fontSize: perfTokens.fontSize.knob,
        letterSpacing: '0.04em',
    },
    masterTransport: {
        display: 'flex',
        flexDirection: 'column',
        gap: 0.5,
        pt: 0.75,
        borderTop: '1px solid',
        borderTopColor: 'divider',
    },
    masterBtn: (color, variant) => (theme) => ({
        textTransform: 'none',
        borderRadius: 1.5,
        fontSize: perfTokens.fontSize.body,
        py: 0.5,
        ...(variant === 'play'
            ? {
                color,
                borderColor: theme.palette.mode === 'dark' ? `${color}66` : `${color}BB`,
                backgroundColor: `${color}14`,
                '&:hover': { backgroundColor: `${color}26`, borderColor: color },
            }
            : {
                color: 'error.main',
                borderColor: 'error.main',
                '&:hover': {
                    bgcolor: theme.palette.mode === 'dark'
                        ? `${theme.palette.error.main}1F`
                        : `${theme.palette.error.main}14`,
                    borderColor: 'error.main',
                },
            }),
    }),
};

export const performanceChannelStyles = {
    strip: (color, playing) => (theme) => ({
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
        p: 1.25,
        borderRadius: 2.5,
        border: '1px solid',
        borderColor: playing ? color : theme.palette.divider,
        background: playing
            ? `linear-gradient(160deg, ${color}1F 0%, ${theme.palette.background.paper} 60%)`
            : theme.palette.background.paper,
        boxShadow: playing
            ? `0 0 0 1px ${color}66, 0 8px 22px ${theme.palette.mode === 'dark' ? 'rgba(4, 8, 14, 0.5)' : 'rgba(0,0,0,0.15)'}`
            : `0 2px 8px ${theme.palette.mode === 'dark' ? 'rgba(4, 8, 14, 0.36)' : 'rgba(0,0,0,0.08)'}`,
        transition: 'box-shadow 0.2s ease, border-color 0.2s ease, background 0.3s ease',
        height: '100%',
        minWidth: 150,
    }),
    stripHeader: (color) => ({
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: `1px solid ${color}33`,
        pb: 0.75,
    }),
    channelBadge: (color) => ({
        fontFamily: 'inherit',
        fontSize: perfTokens.fontSize.badge,
        fontWeight: 600,
        color,
        letterSpacing: perfTokens.letterSpacing.wide,
        px: 0.75,
        py: 0.25,
        borderRadius: 1,
        backgroundColor: `${color}1F`,
        border: `1px solid ${color}55`,
    }),
    muteSoloRow: {
        display: 'flex',
        gap: 0.5,
    },
    muteBtn: (active) => ({
        width: perfTokens.height.sub,
        height: perfTokens.height.sub,
        fontSize: perfTokens.fontSize.small,
        fontWeight: 700,
        borderRadius: 1,
        color: active ? '#fff' : 'text.secondary',
        backgroundColor: active ? 'rgba(227, 108, 97, 0.85)' : 'transparent',
        border: '1px solid',
        borderColor: active ? 'rgba(227, 108, 97, 0.85)' : 'divider',
        '&:hover': {
            backgroundColor: active ? 'rgba(227, 108, 97, 0.95)' : 'rgba(227, 108, 97, 0.18)',
        },
    }),
    soloBtn: (active) => ({
        width: perfTokens.height.sub,
        height: perfTokens.height.sub,
        fontSize: perfTokens.fontSize.small,
        fontWeight: 700,
        borderRadius: 1,
        color: active ? '#0c1018' : 'text.secondary',
        backgroundColor: active ? 'rgba(227, 163, 75, 0.95)' : 'transparent',
        border: '1px solid',
        borderColor: active ? 'rgba(227, 163, 75, 0.95)' : 'divider',
        '&:hover': {
            backgroundColor: active ? 'rgba(227, 163, 75, 1)' : 'rgba(227, 163, 75, 0.2)',
        },
    }),
    promptBox: {
        display: 'flex',
        flexDirection: 'column',
        gap: 0.5,
    },
    promptField: (theme) => ({
        '& .MuiOutlinedInput-root': {
            fontSize: perfTokens.fontSize.body,
            backgroundColor: theme.palette.mode === 'dark' ? 'rgba(9, 12, 18, 0.5)' : 'rgba(0, 0, 0, 0.04)',
            borderRadius: 1.5,
            '& textarea': { lineHeight: 1.3 },
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
        fontSize: perfTokens.fontSize.small,
        color: 'text.secondary',
        minWidth: 22,
    },
    durationSlider: (color) => ({
        flex: 1,
        color,
        '& .MuiSlider-thumb': { width: 10, height: 10 },
        '& .MuiSlider-rail': { opacity: 0.3 },
    }),
    generateBtn: (color) => (theme) => ({
        alignSelf: 'flex-end',
        width: perfTokens.height.compact,
        height: perfTokens.height.compact,
        borderRadius: 1.5,
        color,
        border: `1px solid ${color}55`,
        backgroundColor: `${color}14`,
        '&:hover': { backgroundColor: `${color}26` },
        '&.Mui-disabled': theme.palette.mode === 'dark' ? { opacity: 0.35 } : {},
    }),
    waveformWrap: (theme) => ({
        position: 'relative',
        height: 42,
        backgroundColor: theme.palette.mode === 'dark' ? 'rgba(9, 12, 18, 0.6)' : 'rgba(0, 0, 0, 0.06)',
        borderRadius: 1.5,
        border: '1px solid',
        borderColor: theme.palette.divider,
        overflow: 'hidden',
    }),
    waveformPlaceholder: {
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'text.disabled',
        fontFamily: 'inherit',
        fontSize: perfTokens.fontSize.small,
        letterSpacing: perfTokens.letterSpacing.wide,
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
        display: 'block',
        fontFamily: 'inherit',
        fontSize: perfTokens.fontSize.knob,
        color: 'text.secondary',
        letterSpacing: perfTokens.letterSpacing.wide,
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
        border: `1px solid ${color}55`,
        '&:hover': { backgroundColor: playing ? color : `${color}28` },
        '&.Mui-disabled': theme.palette.mode === 'dark' ? { opacity: 0.3 } : {},
    }),
    loopBtn: (color, active) => ({
        width: perfTokens.height.sub,
        height: perfTokens.height.sub,
        borderRadius: 1,
        color: active ? color : 'text.secondary',
        backgroundColor: active ? `${color}1F` : 'transparent',
        border: '1px solid',
        borderColor: active ? `${color}55` : 'divider',
        '&:hover': { backgroundColor: `${color}1F` },
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
