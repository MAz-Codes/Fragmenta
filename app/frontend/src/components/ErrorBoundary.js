import React from 'react';

// Last-resort catch for render-time crashes. Without a boundary, a single
// "Objects are not valid as a React child" (or any throw during render)
// blanks the whole app with no way back but a reload — and in the desktop
// webview there isn't even a visible reload button.
export default class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { error: null };
    }

    static getDerivedStateFromError(error) {
        return { error };
    }

    componentDidCatch(error, info) {
        console.error('Unhandled render error:', error, info?.componentStack);
    }

    render() {
        if (!this.state.error) return this.props.children;
        return (
            <div style={{
                minHeight: '100vh',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 16,
                background: '#0D1117',
                color: '#e6edf3',
                fontFamily: 'system-ui, sans-serif',
                padding: 24,
                textAlign: 'center',
            }}>
                <h2 style={{ margin: 0 }}>Something went wrong</h2>
                <p style={{ margin: 0, opacity: 0.8, maxWidth: 520 }}>
                    {String(this.state.error?.message || this.state.error)}
                </p>
                <button
                    onClick={() => window.location.reload()}
                    style={{
                        padding: '8px 20px',
                        borderRadius: 8,
                        border: '1px solid #30363d',
                        background: '#21262d',
                        color: '#e6edf3',
                        cursor: 'pointer',
                        fontSize: 14,
                    }}
                >
                    Reload Fragmenta
                </button>
            </div>
        );
    }
}
