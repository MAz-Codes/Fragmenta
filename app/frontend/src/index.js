import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

document.body.style.margin = '0';
document.body.style.padding = '0';

const rootElement = document.getElementById('root');
rootElement.style.minHeight = '100vh';

const root = ReactDOM.createRoot(rootElement);
root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
); 