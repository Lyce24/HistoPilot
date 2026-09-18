import { createContext, useContext, type ReactNode } from 'react';

export interface NumericDraft { source: number; text: string }
export type NumericDrafts = Record<string, NumericDraft>;
interface DraftContext { values: NumericDrafts; onChange: (key: string, value: NumericDraft) => void; prefix: string }
const Context = createContext<DraftContext | null>(null);

/** Opt-in ownership of unfinished numeric text for recoverable editors. */
export function NumericDraftProvider({ values, onChange, children }: {
  values: NumericDrafts; onChange: DraftContext['onChange']; children: ReactNode;
}) {
  return <Context.Provider value={{ values, onChange, prefix: '' }}>{children}</Context.Provider>;
}

/** Repeated recipe fields must retain independent text for each configuration. */
export function NumericDraftScope({ name, children }: { name: string; children: ReactNode }) {
  const context = useContext(Context);
  return <Context.Provider value={context ? { ...context, prefix: `${context.prefix}${name}:` } : null}>{children}</Context.Provider>;
}

export function useNumericDraft(label: string) {
  const context = useContext(Context);
  const key = `${context?.prefix ?? ''}${label}`;
  return { draft: context?.values[key], setDraft: (value: NumericDraft) => context?.onChange(key, value) };
}
