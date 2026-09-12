import type { ReactNode } from 'react';
import { Icon } from './ui';
import './SetupContext.css';

/** Keep the input and saved outcome visible without repeating the page's steps. */
export default function SetupContext({ input, output, children }: {
  input: string;
  output: string;
  children?: ReactNode;
}) {
  return (
    <aside className="setup-context" aria-label="Setup context">
      <div className="setup-context-flow">
        <span><strong>Input</strong> {input}</span>
        <Icon name="arrow" size={16} />
        <span><strong>Save</strong> {output}</span>
      </div>
      {children ? <p>{children}</p> : null}
    </aside>
  );
}
