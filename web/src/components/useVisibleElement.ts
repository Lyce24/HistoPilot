import { useEffect, useRef, useState } from 'react';
/** Delay protected thumbnail and preview requests until cards approach the viewport. */
export function useVisibleElement<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    if (!ref.current) return;
    if (typeof IntersectionObserver === 'undefined') { setVisible(true); return; }
    const observer = new IntersectionObserver(([entry]) => { if (entry.isIntersecting) { setVisible(true); observer.disconnect(); } }, { rootMargin: '240px' });
    observer.observe(ref.current); return () => observer.disconnect();
  }, []);
  return { ref, visible };
}
