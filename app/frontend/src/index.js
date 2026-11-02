import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import ErrorBoundary from './components/ErrorBoundary';
import { loadPerformanceState } from './utils/performanceStore';
import { moveFragmentsOver } from './utils/fragmentStorage';

document.body.style.margin = '0';
document.body.style.padding = '0';

const rootElement = document.getElementById('root');
rootElement.style.minHeight = '100vh';

const root = ReactDOM.createRoot(rootElement);
// The Performance state comes from the data folder; its synchronous reads at
// mount need it in place first. One local request.
loadPerformanceState().finally(() => {
    root.render(
        <React.StrictMode>
            <ErrorBoundary>
                <App />
            </ErrorBoundary>
        </React.StrictMode>
    );
    // Channel audio from before 2.0 moves over in the background.
    moveFragmentsOver().catch((err) => console.warn('[performance] moving fragments failed', err));
});
