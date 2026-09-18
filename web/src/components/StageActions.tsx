import type { ReactNode, ButtonHTMLAttributes, AnchorHTMLAttributes } from 'react';
import { Icon } from './ui';
import './StageActions.css';

type StageActionProps = {
  children: ReactNode;
  tone?: 'primary' | 'secondary';
  size?: 'small';
} & (
  | (ButtonHTMLAttributes<HTMLButtonElement> & { href?: never })
  | (AnchorHTMLAttributes<HTMLAnchorElement> & { href: string; disabled?: never })
);

/** Use the same visual language for record creation and movement in every stage. */
function StageAction({ kind, tone = kind === 'back' ? 'secondary' : 'primary', size, children, className = '', ...props }: StageActionProps & { kind: 'create' | 'back' | 'continue' }) {
  const classes = `btn btn-${tone} stage-action stage-action-${kind}${size ? ` btn-${size}` : ''} ${className}`.trim();
  const icon = <span className="stage-action-icon" aria-hidden="true"><Icon name={kind === 'create' ? 'plus' : 'arrow'} size={16} /></span>;
  const content = <>{kind !== 'continue' ? icon : null}<span>{children}</span>{kind === 'continue' ? icon : null}</>;
  if (props.href !== undefined) {
    return <a {...props as AnchorHTMLAttributes<HTMLAnchorElement>} className={classes} data-stage-action={kind}>{content}</a>;
  }
  return <button type="button" {...props as ButtonHTMLAttributes<HTMLButtonElement>} className={classes} data-stage-action={kind}>{content}</button>;
}

export function StageCreateButton(props: StageActionProps) { return <StageAction {...props} kind="create" />; }
export function StageBackButton(props: StageActionProps) { return <StageAction {...props} kind="back" />; }
export function StageContinueButton(props: StageActionProps) { return <StageAction {...props} kind="continue" />; }
