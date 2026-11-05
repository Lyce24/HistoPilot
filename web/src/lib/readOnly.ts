import { createContext, useContext } from 'react';

/**
 * True inside a packaged project of a public demo (`histopilot serve --demo --demo-projects`):
 * every record reads as usual, nothing can change, and the machine-wide tools are absent.
 * The service refuses changes either way; this only keeps the page from offering them.
 */
export const ReadOnlyContext = createContext(false);
export const useReadOnly = () => useContext(ReadOnlyContext);
