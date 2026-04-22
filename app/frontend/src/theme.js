import { createTheme, responsiveFontSizes } from '@mui/material/styles';

let theme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#35C2D4',
            light: '#73D7E3',
            dark: '#1B98A8',
            contrastText: '#061017',
        },
        secondary: {
            main: '#9AA7BA',
            light: '#C4CEDB',
            dark: '#6E7C92',
            contrastText: '#09101A',
        },
        background: {
            default: '#090C12',
            paper: '#121926',
        },
        text: {
            primary: '#E8EDF5',
            secondary: '#9DA9BC',
        },
        divider: 'rgba(194, 207, 228, 0.16)',
        error: {
            main: '#E36C61',
        },
        warning: {
            main: '#E3A34B',
        },
        success: {
            main: '#53C18A',
        },
        info: {
            main: '#35C2D4',
        },
    },
    shape: {
        borderRadius: 12,
    },
    typography: {
        fontFamily: [
            'Helvetica Neue',
            'Helvetica',
            'Arial',
            'sans-serif'
        ].join(','),
        h1: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 1500,
        },
        h2: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 200,
        },
        h3: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 250,
        },
        h4: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 300,
            letterSpacing: '0.01em',
        },
        h5: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 350,
            letterSpacing: '0.01em',
        },
        h6: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 400,
        },
        body1: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 300,
        },
        body2: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 300,
        },
        button: {
            fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
            fontWeight: 400,
            letterSpacing: '0.01em',
        },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                ':root': {
                    colorScheme: 'dark',
                },
                body: {
                    margin: 0,
                    minHeight: '100vh',
                    backgroundColor: '#090C12',
                    backgroundImage: 'radial-gradient(1400px 700px at 8% -10%, rgba(53, 194, 212, 0.16), transparent 55%), radial-gradient(900px 500px at 92% -20%, rgba(83, 193, 138, 0.12), transparent 60%), linear-gradient(160deg, #090C12 0%, #0D121B 45%, #0A0F16 100%)',
                    color: '#E8EDF5',
                },
                '#root': {
                    minHeight: '100vh',
                },
                '*::-webkit-scrollbar': {
                    width: '10px',
                    height: '10px',
                },
                '*::-webkit-scrollbar-track': {
                    background: 'rgba(157, 169, 188, 0.14)',
                    borderRadius: '999px',
                },
                '*::-webkit-scrollbar-thumb': {
                    background: 'rgba(157, 169, 188, 0.45)',
                    borderRadius: '999px',
                    border: '2px solid rgba(0, 0, 0, 0)',
                    backgroundClip: 'padding-box',
                    '&:hover': {
                        background: 'rgba(157, 169, 188, 0.62)',
                    },
                },
                '*::-webkit-scrollbar-corner': {
                    background: 'rgba(157, 169, 188, 0.12)',
                },
                '*': {
                    scrollbarWidth: 'thin',
                    scrollbarColor: 'rgba(157, 169, 188, 0.45) rgba(157, 169, 188, 0.14)',
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundColor: '#121926',
                    backgroundImage: 'linear-gradient(180deg, rgba(20, 27, 40, 0.92) 0%, rgba(15, 22, 34, 0.95) 100%)',
                    border: '1px solid rgba(194, 207, 228, 0.14)',
                    boxShadow: '0 18px 32px rgba(4, 8, 14, 0.42)',
                    backdropFilter: 'blur(8px)',
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: '#111826',
                    backgroundImage: 'linear-gradient(180deg, rgba(18, 25, 38, 0.98) 0%, rgba(15, 22, 34, 0.96) 100%)',
                    border: '1px solid rgba(194, 207, 228, 0.14)',
                    boxShadow: '0 10px 18px rgba(4, 8, 14, 0.3)',
                    transition: 'border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease',
                    '&:hover': {
                        borderColor: 'rgba(115, 215, 227, 0.3)',
                        boxShadow: '0 16px 30px rgba(4, 8, 14, 0.46)',
                        transform: 'translateY(-1px)',
                    },
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    borderRadius: 10,
                    fontWeight: 600,
                    paddingInline: 16,
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    lineHeight: 1.2,
                    '& .MuiButton-startIcon, & .MuiButton-endIcon': {
                        display: 'inline-flex',
                        alignItems: 'center',
                    },
                    transition: 'transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, background-color 160ms ease',
                },
                contained: {
                    boxShadow: '0 8px 18px rgba(6, 10, 18, 0.46)',
                    '&:hover': {
                        boxShadow: '0 12px 22px rgba(6, 10, 18, 0.58)',
                        transform: 'translateY(-1px)',
                    },
                },
                containedPrimary: {
                    backgroundImage: 'linear-gradient(135deg, #35C2D4 0%, #2AA9B9 55%, #228E9D 100%)',
                },
                containedError: {
                    backgroundImage: 'linear-gradient(135deg, #E36C61 0%, #CF5A4E 100%)',
                },
                outlined: {
                    borderColor: 'rgba(157, 169, 188, 0.4)',
                    '&:hover': {
                        borderColor: '#35C2D4',
                        backgroundColor: 'rgba(53, 194, 212, 0.08)',
                    },
                },
            },
        },
        MuiInputBase: {
            styleOverrides: {
                root: {
                    '&:not(.MuiInputBase-multiline)': {
                        alignItems: 'center',
                    },
                },
                input: {
                    lineHeight: 1.4,
                },
            },
        },
        MuiTextField: {
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        backgroundColor: 'rgba(10, 15, 23, 0.84)',
                        '& fieldset': {
                            borderColor: 'rgba(157, 169, 188, 0.3)',
                        },
                        '&:hover fieldset': {
                            borderColor: 'rgba(157, 169, 188, 0.55)',
                        },
                        '&.Mui-focused fieldset': {
                            borderColor: '#35C2D4',
                        },
                    },
                },
            },
        },
        MuiSelect: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(10, 15, 23, 0.84)',
                    '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(157, 169, 188, 0.3)',
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(157, 169, 188, 0.55)',
                    },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#35C2D4',
                    },
                },
                select: {
                    display: 'flex',
                    alignItems: 'center',
                },
            },
        },
        MuiMenuItem: {
            styleOverrides: {
                root: {
                    backgroundColor: '#121926',
                    '&:hover': {
                        backgroundColor: 'rgba(53, 194, 212, 0.08)',
                    },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(53, 194, 212, 0.14)',
                        '&:hover': {
                            backgroundColor: 'rgba(53, 194, 212, 0.2)',
                        },
                    },
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(157, 169, 188, 0.16)',
                    color: '#E8EDF5',
                    border: '1px solid rgba(157, 169, 188, 0.24)',
                    '&.MuiChip-colorPrimary': {
                        backgroundColor: 'rgba(53, 194, 212, 0.2)',
                        color: '#C8F3F9',
                    },
                },
                outlined: {
                    borderColor: 'rgba(157, 169, 188, 0.4)',
                    '&.MuiChip-colorPrimary': {
                        borderColor: '#35C2D4',
                        color: '#73D7E3',
                    },
                },
            },
        },
        MuiAccordion: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(12, 18, 28, 0.7)',
                    border: '1px solid rgba(194, 207, 228, 0.16)',
                    borderRadius: 12,
                    overflow: 'hidden',
                    '&:before': {
                        display: 'none',
                    },
                    '&.Mui-expanded': {
                        margin: 0,
                    },
                },
            },
        },
        MuiAccordionSummary: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(15, 22, 34, 0.8)',
                    borderRadius: 12,
                    minHeight: 44,
                    '& .MuiAccordionSummary-content': {
                        margin: '10px 0',
                        alignItems: 'center',
                    },
                    '&.Mui-expanded': {
                        minHeight: 44,
                    },
                    '&.Mui-expanded .MuiAccordionSummary-content': {
                        margin: '10px 0',
                    },
                    '&:hover': {
                        backgroundColor: 'rgba(19, 28, 42, 0.9)',
                    },
                },
            },
        },
        MuiDialog: {
            styleOverrides: {
                paper: {
                    backgroundColor: '#121926',
                    backgroundImage: 'linear-gradient(180deg, rgba(20, 27, 40, 0.98) 0%, rgba(14, 21, 33, 0.98) 100%)',
                    border: '1px solid rgba(194, 207, 228, 0.18)',
                    borderRadius: 14,
                    boxShadow: '0 28px 48px rgba(4, 8, 14, 0.6)',
                },
            },
        },
        MuiDialogTitle: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(14, 21, 33, 0.8)',
                    borderBottom: '1px solid rgba(194, 207, 228, 0.15)',
                    color: '#F4F7FC',
                    fontWeight: 600,
                    fontSize: '1.15rem',
                },
            },
        },
        MuiDialogContent: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(14, 21, 33, 0.64)',
                    color: '#CCD5E3',
                },
            },
        },
        MuiDialogActions: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(14, 21, 33, 0.72)',
                    borderTop: '1px solid rgba(194, 207, 228, 0.15)',
                    padding: '14px 20px',
                    gap: 8,
                },
            },
        },
        MuiListItem: {
            styleOverrides: {
                root: {
                    '&:hover': {
                        backgroundColor: 'rgba(53, 194, 212, 0.08)',
                    },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(53, 194, 212, 0.14)',
                        '&:hover': {
                            backgroundColor: 'rgba(53, 194, 212, 0.2)',
                        },
                    },
                },
            },
        },
        MuiCheckbox: {
            styleOverrides: {
                root: {
                    color: '#9AA7BA',
                    '&.Mui-checked': {
                        color: '#35C2D4',
                    },
                    '&:hover': {
                        backgroundColor: 'rgba(53, 194, 212, 0.08)',
                    },
                },
            },
        },
        MuiFormControlLabel: {
            styleOverrides: {
                label: {
                    color: '#CCD5E3',
                    fontSize: '0.875rem',
                },
            },
        },
        MuiSlider: {
            styleOverrides: {
                root: {
                    color: '#35C2D4',
                },
                rail: {
                    backgroundColor: 'rgba(157, 169, 188, 0.24)',
                },
                track: {
                    backgroundColor: '#35C2D4',
                    border: 0,
                },
                thumb: {
                    backgroundColor: '#74DEE9',
                    '&:hover': {
                        boxShadow: '0 0 0 8px rgba(53, 194, 212, 0.2)',
                    },
                },
            },
        },
        MuiLinearProgress: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(157, 169, 188, 0.2)',
                },
                bar: {
                    backgroundColor: '#35C2D4',
                },
            },
        },
        MuiCircularProgress: {
            styleOverrides: {
                root: {
                    color: '#35C2D4',
                },
            },
        },
        MuiTabs: {
            styleOverrides: {
                root: {
                    '& .MuiTabs-indicator': {
                        backgroundColor: '#35C2D4',
                    },
                },
            },
        },
        MuiTab: {
            styleOverrides: {
                root: {
                    color: '#9DA9BC',
                    '&.Mui-selected': {
                        color: '#35C2D4',
                    },
                    '&:hover': {
                        color: '#E8EDF5',
                    },
                },
            },
        },
        MuiBackdrop: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(5, 9, 16, 0.84)',
                    backdropFilter: 'blur(4px)',
                },
            },
        },
        MuiDivider: {
            styleOverrides: {
                root: {
                    borderColor: 'rgba(194, 207, 228, 0.16)',
                },
            },
        },
        MuiIconButton: {
            styleOverrides: {
                root: {
                    color: '#9DA9BC',
                    '&:hover': {
                        backgroundColor: 'rgba(53, 194, 212, 0.1)',
                        color: '#73D7E3',
                    },
                },
            },
        },
        MuiContainer: {
            styleOverrides: {
                root: {
                    backgroundColor: 'transparent',
                    background: 'transparent',
                },
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
        primary: {
            main: '#1497A8',
            light: '#4CBCCA',
            dark: '#0F7482',
            contrastText: '#F7FDFF',
        },
        secondary: {
            main: '#64748B',
            light: '#93A3B8',
            dark: '#475569',
            contrastText: '#F8FAFC',
        },
        background: {
            default: '#F5F9FC',
            paper: '#FFFFFF',
        },
        text: {
            primary: '#0F172A',
            secondary: '#475569',
            disabled: 'rgba(0, 0, 0, 0.38)',
        },
        action: {
            active: 'rgba(0, 0, 0, 0.54)',
            hover: 'rgba(0, 0, 0, 0.04)',
            selected: 'rgba(0, 0, 0, 0.08)',
            disabled: 'rgba(0, 0, 0, 0.26)',
            disabledBackground: 'rgba(0, 0, 0, 0.12)',
        },
        divider: 'rgba(15, 23, 42, 0.14)',
        error: {
            main: '#DC5B57',
        },
        warning: {
            main: '#D08C30',
        },
        success: {
            main: '#2E9E63',
        },
        info: {
            main: '#1497A8',
        },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                ':root': {
                    colorScheme: 'light',
                },
                body: {
                    margin: 0,
                    minHeight: '100vh',
                    backgroundColor: '#F5F9FC',
                    backgroundImage: 'radial-gradient(1400px 700px at 8% -10%, rgba(20, 151, 168, 0.14), transparent 55%), radial-gradient(900px 500px at 92% -20%, rgba(46, 158, 99, 0.1), transparent 60%), linear-gradient(160deg, #F6FAFD 0%, #EFF5FA 45%, #F8FBFE 100%)',
                    color: '#0F172A',
                },
                '#root': {
                    minHeight: '100vh',
                },
                '*::-webkit-scrollbar-track': {
                    background: 'rgba(100, 116, 139, 0.12)',
                    borderRadius: '999px',
                },
                '*::-webkit-scrollbar-thumb': {
                    background: 'rgba(100, 116, 139, 0.38)',
                    borderRadius: '999px',
                    '&:hover': {
                        background: 'rgba(100, 116, 139, 0.52)',
                    },
                },
                '*::-webkit-scrollbar-corner': {
                    background: 'rgba(100, 116, 139, 0.1)',
                },
                '*': {
                    scrollbarWidth: 'thin',
                    scrollbarColor: 'rgba(100, 116, 139, 0.38) rgba(100, 116, 139, 0.12)',
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundColor: '#FFFFFF',
                    backgroundImage: 'linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 251, 255, 0.98) 100%)',
                    border: '1px solid rgba(15, 23, 42, 0.12)',
                    boxShadow: '0 14px 26px rgba(15, 23, 42, 0.08)',
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: '#FFFFFF',
                    backgroundImage: 'linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(248, 251, 255, 0.99) 100%)',
                    border: '1px solid rgba(15, 23, 42, 0.12)',
                    boxShadow: '0 10px 18px rgba(15, 23, 42, 0.08)',
                    '&:hover': {
                        borderColor: 'rgba(20, 151, 168, 0.32)',
                        boxShadow: '0 16px 30px rgba(15, 23, 42, 0.12)',
                    },
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                contained: {
                    boxShadow: '0 8px 18px rgba(20, 151, 168, 0.22)',
                    '&:hover': {
                        boxShadow: '0 12px 24px rgba(20, 151, 168, 0.28)',
                    },
                },
                containedPrimary: {
                    backgroundImage: 'linear-gradient(135deg, #1497A8 0%, #1AAABC 55%, #107A88 100%)',
                },
                containedError: {
                    backgroundImage: 'linear-gradient(135deg, #DC5B57 0%, #CB4B45 100%)',
                },
                outlined: {
                    borderColor: 'rgba(100, 116, 139, 0.32)',
                    '&:hover': {
                        borderColor: '#1497A8',
                        backgroundColor: 'rgba(20, 151, 168, 0.08)',
                    },
                },
            },
        },
        MuiTextField: {
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        backgroundColor: 'rgba(255, 255, 255, 0.92)',
                        '& fieldset': {
                            borderColor: 'rgba(100, 116, 139, 0.28)',
                        },
                        '&:hover fieldset': {
                            borderColor: 'rgba(100, 116, 139, 0.5)',
                        },
                        '&.Mui-focused fieldset': {
                            borderColor: '#1497A8',
                        },
                    },
                },
            },
        },
        MuiSelect: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(255, 255, 255, 0.92)',
                    '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(100, 116, 139, 0.28)',
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(100, 116, 139, 0.5)',
                    },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#1497A8',
                    },
                },
            },
        },
        MuiMenuItem: {
            styleOverrides: {
                root: {
                    backgroundColor: '#FFFFFF',
                    '&:hover': {
                        backgroundColor: 'rgba(20, 151, 168, 0.08)',
                    },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(20, 151, 168, 0.14)',
                        '&:hover': {
                            backgroundColor: 'rgba(20, 151, 168, 0.2)',
                        },
                    },
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(100, 116, 139, 0.12)',
                    color: '#0F172A',
                    border: '1px solid rgba(100, 116, 139, 0.24)',
                },
            },
        },
        MuiAccordion: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(255, 255, 255, 0.82)',
                    border: '1px solid rgba(15, 23, 42, 0.12)',
                    borderRadius: 12,
                    overflow: 'hidden',
                    '&:before': {
                        display: 'none',
                    },
                    '&.Mui-expanded': {
                        margin: 0,
                    },
                },
            },
        },
        MuiAccordionSummary: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(248, 251, 255, 0.95)',
                    borderRadius: 12,
                    '&:hover': {
                        backgroundColor: 'rgba(239, 245, 250, 1)',
                    },
                },
            },
        },
        MuiDialog: {
            styleOverrides: {
                paper: {
                    backgroundColor: '#FFFFFF',
                    backgroundImage: 'linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(246, 250, 254, 0.99) 100%)',
                    border: '1px solid rgba(15, 23, 42, 0.12)',
                    boxShadow: '0 28px 48px rgba(15, 23, 42, 0.16)',
                },
            },
        },
        MuiDialogTitle: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(246, 250, 254, 0.98)',
                    borderBottom: '1px solid rgba(15, 23, 42, 0.1)',
                    color: '#0F172A',
                },
            },
        },
        MuiDialogContent: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(255, 255, 255, 0.98)',
                    color: '#334155',
                },
            },
        },
        MuiDialogActions: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(246, 250, 254, 0.98)',
                    borderTop: '1px solid rgba(15, 23, 42, 0.1)',
                },
            },
        },
        MuiListItem: {
            styleOverrides: {
                root: {
                    '&:hover': {
                        backgroundColor: 'rgba(53, 194, 212, 0.08)',
                    },
                    '&.Mui-selected': {
                        backgroundColor: 'rgba(53, 194, 212, 0.14)',
                        '&:hover': {
                            backgroundColor: 'rgba(53, 194, 212, 0.2)',
                        },
                    },
                },
            },
        },
        MuiCheckbox: {
            styleOverrides: {
                root: {
                    color: '#64748B',
                    '&.Mui-checked': {
                        color: '#1497A8',
                    },
                    '&:hover': {
                        backgroundColor: 'rgba(20, 151, 168, 0.08)',
                    },
                },
            },
        },
        MuiFormControlLabel: {
            styleOverrides: {
                label: {
                    color: '#334155',
                },
            },
        },
        MuiSlider: {
            styleOverrides: {
                root: {
                    color: '#1497A8',
                },
                rail: {
                    backgroundColor: 'rgba(100, 116, 139, 0.24)',
                },
                track: {
                    backgroundColor: '#1497A8',
                    border: 0,
                },
                thumb: {
                    backgroundColor: '#4CBCCA',
                    '&:hover': {
                        boxShadow: '0 0 0 8px rgba(20, 151, 168, 0.2)',
                    },
                },
            },
        },
        MuiLinearProgress: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(100, 116, 139, 0.2)',
                },
                bar: {
                    backgroundColor: '#1497A8',
                },
            },
        },
        MuiCircularProgress: {
            styleOverrides: {
                root: {
                    color: '#1497A8',
                },
            },
        },
        MuiTabs: {
            styleOverrides: {
                root: {
                    '& .MuiTabs-indicator': {
                        backgroundColor: '#1497A8',
                    },
                },
            },
        },
        MuiTab: {
            styleOverrides: {
                root: {
                    color: '#64748B',
                    '&.Mui-selected': {
                        color: '#1497A8',
                    },
                    '&:hover': {
                        color: '#0F172A',
                    },
                },
            },
        },
        MuiBackdrop: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(15, 23, 42, 0.38)',
                    backdropFilter: 'blur(4px)',
                },
            },
        },
        MuiDivider: {
            styleOverrides: {
                root: {
                    borderColor: 'rgba(15, 23, 42, 0.14)',
                },
            },
        },
        MuiIconButton: {
            styleOverrides: {
                root: {
                    color: '#64748B',
                    '&:hover': {
                        backgroundColor: 'rgba(20, 151, 168, 0.1)',
                        color: '#1497A8',
                    },
                },
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
    elevatedInfoCard: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            p: { xs: 1.5, sm: 2 },
            mb: 2,
            boxShadow: isDark
                ? '0 14px 28px rgba(4, 8, 14, 0.44)'
                : '0 14px 26px rgba(15, 23, 42, 0.1)',
            borderRadius: 2.5,
            border: isDark
                ? '1px solid rgba(194, 207, 228, 0.16)'
                : '1px solid rgba(15, 23, 42, 0.12)',
            background: isDark
                ? 'linear-gradient(160deg, rgba(17, 24, 37, 0.96) 0%, rgba(13, 20, 31, 0.92) 100%)'
                : 'linear-gradient(160deg, rgba(255, 255, 255, 0.98) 0%, rgba(245, 250, 255, 0.98) 100%)',
            '&:hover': {
                boxShadow: isDark
                    ? '0 20px 34px rgba(4, 8, 14, 0.56)'
                    : '0 20px 34px rgba(15, 23, 42, 0.14)',
                transform: 'translateY(-1px)',
                transition: 'all 0.3s ease',
            },
            transition: 'all 0.3s ease',
        };
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
    sectionCardTitle: {
        fontWeight: 500,
    },
    selectedModelCard: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            p: { xs: 1.5, sm: 2 },
            mb: 2,
            boxShadow: isDark
                ? '0 14px 28px rgba(4, 8, 14, 0.44)'
                : '0 14px 26px rgba(15, 23, 42, 0.1)',
            borderRadius: 2.5,
            border: isDark
                ? '1px solid rgba(194, 207, 228, 0.16)'
                : '1px solid rgba(15, 23, 42, 0.12)',
            background: isDark
                ? 'linear-gradient(160deg, rgba(17, 24, 37, 0.96) 0%, rgba(13, 20, 31, 0.92) 100%)'
                : 'linear-gradient(160deg, rgba(255, 255, 255, 0.98) 0%, rgba(245, 250, 255, 0.98) 100%)',
            '&:hover': {
                boxShadow: isDark
                    ? '0 20px 34px rgba(4, 8, 14, 0.56)'
                    : '0 20px 34px rgba(15, 23, 42, 0.14)',
                transform: 'translateY(-1px)',
                transition: 'all 0.3s ease',
            },
            transition: 'all 0.3s ease',
        };
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
        mt: 1,
        color: 'text.secondary',
    },
    infoDialogSectionTitle: {
        mt: 2.25,
        mb: 1,
        color: 'text.primary',
        fontWeight: 700,
        letterSpacing: '0.03em',
        textTransform: 'uppercase',
        fontSize: { xs: '0.7rem', sm: '0.74rem' },
    },
    infoDialogActionStack: {
        mt: 0.25,
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
    },
    infoDocButton: {
        justifyContent: 'flex-start',
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
    card: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            mb: { xs: 1.5, sm: 2 },
            boxShadow: isDark
                ? '0 12px 24px rgba(4, 8, 14, 0.34)'
                : '0 12px 24px rgba(15, 23, 42, 0.1)',
            borderRadius: 2.2,
            border: isDark
                ? '1px solid rgba(194, 207, 228, 0.15)'
                : '1px solid rgba(15, 23, 42, 0.12)',
            background: isDark
                ? 'linear-gradient(160deg, rgba(17, 24, 37, 0.96) 0%, rgba(13, 20, 31, 0.92) 100%)'
                : 'linear-gradient(160deg, rgba(255, 255, 255, 0.99) 0%, rgba(245, 250, 255, 0.98) 100%)',
            '&:hover': {
                boxShadow: isDark
                    ? '0 16px 30px rgba(4, 8, 14, 0.46)'
                    : '0 16px 30px rgba(15, 23, 42, 0.14)',
                transform: 'translateY(-1px)',
                transition: 'all 0.3s ease',
            },
            transition: 'all 0.3s ease',
        };
    },
    cardContent: {
        p: { xs: 1.5, sm: 2 },
        '&:last-child': {
            pb: { xs: 1.5, sm: 2 },
        },
    },
    gridSpacing: { xs: 1.5, sm: 2 },
    uploadDropZone: (isDragActive) => (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            border: isDark
                ? '1.5px dashed rgba(194, 207, 228, 0.35)'
                : '1.5px dashed rgba(100, 116, 139, 0.34)',
            borderRadius: 2,
            p: { xs: 1.5, sm: 2 },
            textAlign: 'center',
            cursor: 'pointer',
            '&:hover': {
                borderColor: 'primary.main',
                backgroundColor: isDark
                    ? 'rgba(53, 194, 212, 0.08)'
                    : 'rgba(20, 151, 168, 0.08)',
            },
            backgroundColor: isDragActive
                ? (isDark ? 'rgba(53, 194, 212, 0.12)' : 'rgba(20, 151, 168, 0.12)')
                : (isDark ? 'rgba(10, 15, 23, 0.8)' : 'rgba(255, 255, 255, 0.9)'),
        };
    },
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
    rootPaper: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            p: 2,
            height: 240,
            display: 'flex',
            flexDirection: 'column',
            borderRadius: 2.5,
            borderColor: isDark
                ? 'rgba(194, 207, 228, 0.16)'
                : 'rgba(15, 23, 42, 0.12)',
            background: isDark
                ? 'linear-gradient(160deg, rgba(17, 24, 37, 0.94) 0%, rgba(13, 20, 31, 0.9) 100%)'
                : 'linear-gradient(160deg, rgba(255, 255, 255, 0.99) 0%, rgba(245, 250, 255, 0.98) 100%)',
            boxShadow: isDark
                ? '0 14px 28px rgba(4, 8, 14, 0.42)'
                : '0 14px 28px rgba(15, 23, 42, 0.1)',
        };
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
    titleText: {
        fontWeight: 500,
    },
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
    listRoot: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            flex: 1,
            overflow: 'auto',
            maxHeight: 180,
            '& .MuiListItem-root': {
                border: '1px solid',
                borderColor: isDark
                    ? 'rgba(194, 207, 228, 0.16)'
                    : 'rgba(15, 23, 42, 0.12)',
                borderRadius: 1.5,
                mb: 1,
                backgroundColor: isDark
                    ? 'rgba(12, 18, 28, 0.62)'
                    : 'rgba(248, 251, 255, 0.9)',
                '&:last-child': {
                    mb: 0,
                },
            },
        };
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
    rootPaper: (muiTheme) => {
        const isDark = muiTheme.palette.mode === 'dark';
        return {
            p: 3,
            mb: 2,
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
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
    headerTitle: {
        fontWeight: 500,
    },
    statusInline: (muiTheme) => ({
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

export const performancePanelStyles = {
    root: {
        display: 'flex',
        flexDirection: 'column',
        gap: 1.5,
        width: '100%',
        minHeight: 0,
    },
    headerCard: {
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        p: { xs: 1.25, sm: 1.75 },
        borderRadius: 2.5,
        border: '1px solid',
        borderColor: 'divider',
        background: 'linear-gradient(135deg, rgba(53, 194, 212, 0.08) 0%, rgba(159, 138, 230, 0.06) 100%)',
        flexWrap: { xs: 'wrap', md: 'nowrap' },
    },
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
        fontSize: '0.72rem',
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
        background: theme.palette.mode === 'dark'
            ? `linear-gradient(160deg, ${color}14 0%, rgba(13, 20, 31, 0.94) 70%)`
            : `linear-gradient(160deg, ${color}14 0%, ${theme.palette.background.paper} 70%)`,
        boxShadow: theme.palette.mode === 'dark'
            ? `0 8px 22px rgba(4, 8, 14, 0.44), inset 0 0 0 1px ${color}22`
            : `0 2px 8px rgba(0,0,0,0.1), inset 0 0 0 1px ${color}22`,
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
        fontSize: '0.72rem',
        fontWeight: 600,
        color,
        letterSpacing: '0.14em',
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
    masterMeterTrack: (theme) => ({
        width: 10,
        backgroundColor: theme.palette.mode === 'dark' ? 'rgba(9, 12, 18, 0.7)' : 'rgba(0, 0, 0, 0.08)',
        borderRadius: 0.75,
        border: '1px solid',
        borderColor: theme.palette.divider,
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'flex-end',
    }),
    masterMeterFill: (color) => ({
        width: '100%',
        height: '0%',
        background: `linear-gradient(0deg, ${color} 0%, ${color}DD 60%, #E3A34B 80%, #E36C61 100%)`,
        transition: 'height 0.05s linear',
    }),
    masterFader: (color) => (theme) => ({
        height: '100%',
        color,
        '& .MuiSlider-thumb': { width: 16, height: 16 },
        '& .MuiSlider-rail': { opacity: 0.3, width: 4 },
        '& .MuiSlider-track': { display: 'none' },
        '& .MuiSlider-mark': {
            width: 6,
            height: 1,
            backgroundColor: theme.palette.mode === 'dark' ? 'rgba(157, 169, 188, 0.55)' : 'rgba(0, 0, 0, 0.25)',
            opacity: 1,
        },
        '& .MuiSlider-markActive': {
            backgroundColor: theme.palette.mode === 'dark' ? 'rgba(157, 169, 188, 0.55)' : 'rgba(0, 0, 0, 0.25)',
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
        fontSize: '0.68rem',
        letterSpacing: '0.04em',
    },
    masterPeakValue: {
        textAlign: 'center',
        color: 'text.disabled',
        fontSize: '0.58rem',
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
        fontSize: '0.72rem',
        py: 0.5,
        ...(variant === 'play'
            ? {
                color,
                borderColor: theme.palette.mode === 'dark' ? `${color}66` : `${color}BB`,
                backgroundColor: `${color}14`,
                '&:hover': { backgroundColor: `${color}26`, borderColor: color },
            }
            : {
                color: '#E36C61',
                borderColor: theme.palette.mode === 'dark' ? 'rgba(227, 108, 97, 0.5)' : 'rgba(227, 108, 97, 0.8)',
                '&:hover': { backgroundColor: 'rgba(227, 108, 97, 0.12)', borderColor: '#E36C61' },
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
        fontSize: '0.78rem',
        fontWeight: 600,
        color,
        letterSpacing: '0.08em',
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
        width: 22,
        height: 22,
        fontSize: '0.65rem',
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
        width: 22,
        height: 22,
        fontSize: '0.65rem',
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
            fontSize: '0.75rem',
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
        fontSize: '0.65rem',
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
        width: 28,
        height: 28,
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
        fontSize: '0.65rem',
        letterSpacing: '0.1em',
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
    knobSlider: (color) => ({
        height: 50,
        color,
        '& .MuiSlider-thumb': { width: 10, height: 10 },
        '& .MuiSlider-rail': { opacity: 0.3, width: 2 },
        '& .MuiSlider-track': { width: 2, border: 'none' },
    }),
    knobLabel: {
        display: 'block',
        fontFamily: 'inherit',
        fontSize: '0.53rem',
        color: 'text.secondary',
        letterSpacing: '0.06em',
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
        width: 26,
        height: 26,
        borderRadius: 1.5,
        color: playing ? '#0c1018' : color,
        backgroundColor: playing ? color : `${color}14`,
        border: `1px solid ${color}55`,
        '&:hover': { backgroundColor: playing ? color : `${color}28` },
        '&.Mui-disabled': theme.palette.mode === 'dark' ? { opacity: 0.3 } : {},
    }),
    loopBtn: (color, active) => ({
        width: 22,
        height: 22,
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
