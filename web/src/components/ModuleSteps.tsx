import { Icon } from './ui';
import './ModuleSteps.css';

export default function ModuleSteps({ input, procedure, output }: {
  input: string;
  procedure: string;
  output: string;
}) {
  return (
    <section className="module-brief" aria-label="Module workflow">
      <div className="module-brief-item">
        <span className="module-brief-icon"><Icon name="folder" size={19} /></span>
        <div><h2>Start with</h2><p>{input}</p></div>
      </div>
      <div className="module-brief-item">
        <span className="module-brief-icon"><Icon name="branch" size={19} /></span>
        <div><h2>Work through</h2><p>{procedure}</p></div>
      </div>
      <div className="module-brief-item module-brief-outcome">
        <span className="module-brief-icon"><Icon name="lock" size={19} /></span>
        <div><h2>Finish with</h2><p>{output}</p><small>Freeze to complete this module and return to the roadmap.</small></div>
      </div>
    </section>
  );
}
