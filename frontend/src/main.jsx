import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/main.css';
import './styles/components/chat.css';
import './styles/components/documents.css';
import './styles/components/settings.css';
import './styles/components/modals.css';
import './styles/utils.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

