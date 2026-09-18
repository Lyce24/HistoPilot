import type { ReactNode } from 'react';
import { renderToReadableStream } from 'react-dom/server';

/** Wait for actual lazy page content instead of asserting only its fallback. */
export async function renderLoadedPage(children: ReactNode, onError?: (error: unknown) => void) {
  const stream = await renderToReadableStream(children, { onError });
  await stream.allReady;
  return new Response(stream).text();
}
