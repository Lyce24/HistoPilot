import { Component, type ErrorInfo, type ReactNode } from 'react';
import './WorkspaceErrorBoundary.css';

interface Props { children: ReactNode; onExit?: () => void }
interface State { error: Error | null }

/** Keep navigation and job controls available when one module fails to render. */
export default class WorkspaceErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: unknown): State {
    return { error: error instanceof Error ? error : new Error('An unexpected display error occurred.') };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('HistoPilot could not display this view.', error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return <WorkspaceRecovery error={this.state.error} onRetry={() => this.setState({ error: null })} onExit={this.props.onExit} />;
  }
}

export function WorkspaceRecovery({ error, onRetry, onExit }: {
  error: Error; onRetry: () => void; onExit?: () => void;
}) {
  return <section className="workspace-recovery" role="alert" aria-labelledby="workspace-recovery-title">
    <h1 id="workspace-recovery-title">This view could not be displayed</h1>
    <p>Your saved project records remain on the server. Running jobs continue independently.</p>
    <p>Try opening this view again. Unsaved entries in this view may need to be entered again.</p>
    <div className="workspace-recovery-actions">
      <button className="btn btn-primary" onClick={onRetry}>Try this view again</button>
      {onExit ? <button className="btn btn-secondary" onClick={onExit}>Back to start</button> : null}
    </div>
    <details><summary>Error details</summary><pre>{error.message || 'An unexpected display error occurred.'}</pre></details>
  </section>;
}
