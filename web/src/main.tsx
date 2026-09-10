import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import './styles.css';
import './local-workspace.css';
import './scientific.css';
const client = new QueryClient({
  defaultOptions: { queries: { staleTime: 10000, retry: 1 }, mutations: { retry: false } },
});
ReactDOM.createRoot(document.getElementById('app')!).render(
  <React.StrictMode>
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
