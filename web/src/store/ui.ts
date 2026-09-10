import { create } from 'zustand';
import type { Page } from '../api/types';
// Selection only. Scientific records belong to the control service.
interface UIState {
  selectedResultId: string | null;
  selectedSlideId: string | null;
  selectedCohortId: string | null;
  encoderId: string | null;
  milId: string | null;
  setSelectedResultId: (id: string) => void;
  setSelectedSlideId: (id: string) => void;
  setSelectedCohortId: (id: string) => void;
  setEncoderId: (id: string) => void;
  setMilId: (id: string) => void;
  resetSelection: () => void;
  navigate: (page: Page) => void;
}
export const useUIStore = create<UIState>((set) => ({
  selectedResultId: null,
  selectedSlideId: null,
  selectedCohortId: null,
  encoderId: null,
  milId: null,
  setSelectedResultId: (id) => set({ selectedResultId: id }),
  setSelectedSlideId: (id) => set({ selectedSlideId: id }),
  setSelectedCohortId: (id) => set({ selectedCohortId: id }),
  setEncoderId: (id) => set({ encoderId: id }),
  setMilId: (id) => set({ milId: id }),
  resetSelection: () =>
    set({
      selectedResultId: null,
      selectedSlideId: null,
      selectedCohortId: null,
      encoderId: null,
      milId: null,
    }),
  navigate: (page) => {
    window.location.hash = page;
  },
}));
