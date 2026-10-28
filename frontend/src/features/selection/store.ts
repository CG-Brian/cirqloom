// Shared UI state (AGENTS.md §10). Holds ids and flags only: no three.js objects, no camera pose,
// no bundle data. Scene components subscribe with narrow selectors so a hover re-renders 2 nodes.
import { create } from "zustand";

export type ViewMode = "structure" | "functional" | "published-ioi";

/** A one-shot request to the camera controller. `nonce` makes repeated requests distinct. */
export type CameraCommand = { kind: "reset"; nonce: number };

export interface ExplorerState {
  viewMode: ViewMode;
  /** Experiment the user picked; null means the bundle's default (see getActiveExperiment). */
  experimentId: string | null;
  activeCircuitId: string | null;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  cameraCommand: CameraCommand | null;
}

export interface ExplorerActions {
  setViewMode(viewMode: ViewMode): void;
  setExperiment(experimentId: string): void;
  /** Activates the circuit overlay, or turns it off if it is already active. */
  toggleCircuit(circuitId: string): void;
  select(nodeId: string | null): void;
  hover(nodeId: string): void;
  /** Clears hover only if `nodeId` is still the hovered node, so a late pointer-out is ignored. */
  unhover(nodeId: string): void;
  resetCamera(): void;
}

export type ExplorerStore = ExplorerState & ExplorerActions;

export const initialExplorerState: ExplorerState = {
  viewMode: "structure",
  experimentId: null,
  activeCircuitId: null,
  selectedNodeId: null,
  hoveredNodeId: null,
  cameraCommand: null,
};

export const useExplorerStore = create<ExplorerStore>()((set) => ({
  ...initialExplorerState,
  setViewMode: (viewMode) => set({ viewMode }),
  setExperiment: (experimentId) => set({ experimentId }),
  toggleCircuit: (circuitId) =>
    set((s) => ({ activeCircuitId: s.activeCircuitId === circuitId ? null : circuitId })),
  select: (selectedNodeId) => set({ selectedNodeId }),
  hover: (hoveredNodeId) => set({ hoveredNodeId }),
  // Returning the same state object tells Zustand nothing changed, so no subscriber is notified.
  unhover: (nodeId) => set((s) => (s.hoveredNodeId === nodeId ? { hoveredNodeId: null } : s)),
  resetCamera: () =>
    set((s) => ({ cameraCommand: { kind: "reset", nonce: (s.cameraCommand?.nonce ?? 0) + 1 } })),
}));
