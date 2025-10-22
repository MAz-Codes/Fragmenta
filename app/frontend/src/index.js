import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

document.body.style.margin = '0';
document.body.style.padding = '0';
document.body.style.backgroundColor = '#0D1117';

document.body.style.overflow = 'auto';
document.documentElement.style.backgroundColor = '#0D1117';

document.documentElement.style.overflow = 'auto';

const rootElement = document.getElementById('root');

rootElement.style.overflow = 'auto';
rootElement.style.minHeight = '100vh';

const root = ReactDOM.createRoot(rootElement);
root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
); 