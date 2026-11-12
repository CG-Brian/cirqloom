// Shared UI state (docs/spec.md §10). Holds ids and flags only: no three.js objects, no camera pose,
// no bundle data. Scene components subscribe with narrow selectors so a hover re-renders 2 nodes.
import { create } from "zustand";
import type { Bounds } from "../../scene/layout";

/**
 * A one-shot request to the camera controller. `nonce` makes repeated requests distinct.
 * "focus" frames the given bounds (a circuit, a relation, a head) from the canonical direction.
 */
export type CameraCommand = { kind: "reset"; nonce: number } | { kind: "focus"; bounds: Bounds; nonce: number };

export interface ExplorerState {
  /** Experiment the user picked; null means the bundle's default (see getActiveExperiment). */
  experimentId: string | null;
  activeCircuitId: string | null;
  /** Detector whose mask the heatmap shows; kept across heads (a mask is the same for all). */
  maskDetectorId: string | null;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  /** A circuit relation (visual edge group) under the pointer / opened in the inspector. */
  hoveredEdgeGroupId: string | null;
  selectedEdgeGroupId: string | null;
  cameraCommand: CameraCommand | null;
}

export interface ExplorerActions {
  setExperiment(experimentId: string): void;
  /** Activates the circuit overlay, or turns it off if it is already active. */
  toggleCircuit(circuitId: string): void;
  /** Shows this detector's mask on the heatmap, or hides it if it is already shown. */
  toggleMaskDetector(detectorId: string): void;
  /** Selects a head (or nothing). Clears any selected relation: one thing is inspected at a time. */
  select(nodeId: string | null): void;
  /** Opens a relation in the inspector; clears the head selection. */
  selectEdgeGroup(groupId: string | null): void;
  hoverEdgeGroup(groupId: string): void;
  unhoverEdgeGroup(groupId: string): void;
  hover(nodeId: string): void;
  /** Clears hover only if `nodeId` is still the hovered node, so a late pointer-out is ignored. */
  unhover(nodeId: string): void;
  resetCamera(): void;
  focusCamera(bounds: Bounds): void;
}

export type ExplorerStore = ExplorerState & ExplorerActions;

export const initialExplorerState: ExplorerState = {
  experimentId: null,
  activeCircuitId: null,
  maskDetectorId: null,
  selectedNodeId: null,
  hoveredNodeId: null,
  hoveredEdgeGroupId: null,
  selectedEdgeGroupId: null,
  cameraCommand: null,
};

export const useExplorerStore = create<ExplorerStore>()((set) => ({
  ...initialExplorerState,
  setExperiment: (experimentId) => set({ experimentId }),
  // Relations belong to one circuit, so switching circuits closes them.
  toggleCircuit: (circuitId) =>
    set((s) => ({
      activeCircuitId: s.activeCircuitId === circuitId ? null : circuitId,
      hoveredEdgeGroupId: null,
      selectedEdgeGroupId: null,
    })),
  toggleMaskDetector: (detectorId) =>
    set((s) => ({ maskDetectorId: s.maskDetectorId === detectorId ? null : detectorId })),
  select: (selectedNodeId) => set({ selectedNodeId, selectedEdgeGroupId: null }),
  selectEdgeGroup: (selectedEdgeGroupId) => set({ selectedEdgeGroupId, selectedNodeId: null }),
  hoverEdgeGroup: (hoveredEdgeGroupId) => set({ hoveredEdgeGroupId }),
  unhoverEdgeGroup: (groupId) =>
    set((s) => (s.hoveredEdgeGroupId === groupId ? { hoveredEdgeGroupId: null } : s)),
  hover: (hoveredNodeId) => set({ hoveredNodeId }),
  // Returning the same state object tells Zustand nothing changed, so no subscriber is notified.
  unhover: (nodeId) => set((s) => (s.hoveredNodeId === nodeId ? { hoveredNodeId: null } : s)),
  resetCamera: () =>
    set((s) => ({ cameraCommand: { kind: "reset", nonce: (s.cameraCommand?.nonce ?? 0) + 1 } })),
  focusCamera: (bounds) =>
    set((s) => ({ cameraCommand: { kind: "focus", bounds, nonce: (s.cameraCommand?.nonce ?? 0) + 1 } })),
}));
