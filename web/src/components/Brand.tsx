import type { CSSProperties } from 'react';
import './Brand.css';

interface HistoPilotMarkProps {
  size?: number;
  className?: string;
  /** Hide the symbol when adjacent text already provides its name. */
  decorative?: boolean;
  label?: string;
}

/**
 * One cell from an epithelial sheet, taken as the field, with a compass needle
 * through it. The outline and the needle's trailing half take the surrounding text
 * colour, so the mark stays legible on the light workspace and on the dark
 * navigation alike; the gold leading half is the one fixed colour.
 */
export function HistoPilotMark({ size = 36, className = '', decorative = false, label = 'HistoPilot' }: HistoPilotMarkProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className={`hp-brand__mark ${className}`.trim()}
      viewBox="0 0 48 48"
      width={size}
      height={size}
      role={decorative ? undefined : 'img'}
      aria-label={decorative ? undefined : label}
      aria-hidden={decorative ? true : undefined}
      focusable="false"
    >
      <path fill="none" stroke="currentColor" strokeWidth="3.6" strokeLinejoin="round" d="M43 24 33.5 7.6 14.5 7.6 5 24 14.5 40.4 33.5 40.4Z" />
      <path fill="currentColor" d="M17.3 30.7 26.8 26.8 21.2 21.2Z" />
      <path fill="#ffc72c" d="M32.2 15.8 26.8 26.8 21.2 21.2Z" />
    </svg>
  );
}

interface BrandProps {
  /** Symbol size in pixels. The wordmark scales with it. */
  size?: number;
  iconOnly?: boolean;
  subtitle?: string;
  tone?: 'default' | 'inverse';
  className?: string;
}

/** Noninteractive brand content; wrap in the appropriate navigation link or button. */
export function Brand({ size = 36, iconOnly = false, subtitle, tone = 'default', className = '' }: BrandProps) {
  return (
    <span
      className={`hp-brand hp-brand--${tone}${iconOnly ? ' hp-brand--icon-only' : ''} ${className}`.trim()}
      style={{ '--hp-brand-size': `${size}px` } as CSSProperties}
    >
      <HistoPilotMark size={size} decorative={!iconOnly} />
      {!iconOnly ? (
        <span className="hp-brand__type">
          <span className="hp-brand__wordmark">Histo<strong>Pilot</strong></span>
          {subtitle ? <span className="hp-brand__subtitle">{subtitle}</span> : null}
        </span>
      ) : null}
    </span>
  );
}
