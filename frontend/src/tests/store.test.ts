import { beforeEach, describe, expect, it } from "vitest";
import { initialExplorerState, useExplorerStore } from "../features/selection/store";

const store = () => useExplorerStore.getState();

beforeEach(() => {
  useExplorerStore.setState(initialExplorerState);
});

describe("explorer store", () => {
  it("toggles a circuit on, off, and switches between circuits", () => {
    store().toggleCircuit("A");
    expect(store().activeCircuitId).toBe("A");
    store().toggleCircuit("B");
    expect(store().activeCircuitId).toBe("B");
    store().toggleCircuit("B");
    expect(store().activeCircuitId).toBeNull();
  });

  it("selects and clears a head", () => {
    store().select("n1");
    expect(store().selectedNodeId).toBe("n1");
    store().select(null);
    expect(store().selectedNodeId).toBeNull();
  });

  it("ignores a late pointer-out from a head that is no longer hovered", () => {
    store().hover("n1");
    store().hover("n2");
    store().unhover("n1");
    expect(store().hoveredNodeId).toBe("n2");
    store().unhover("n2");
    expect(store().hoveredNodeId).toBeNull();
  });

  it("does not notify subscribers for a no-op unhover", () => {
    let notified = 0;
    const stop = useExplorerStore.subscribe(() => notified++);
    store().unhover("nobody");
    stop();
    expect(notified).toBe(0);
  });

  it("issues a distinct camera command on every reset", () => {
    store().resetCamera();
    const first = store().cameraCommand;
    store().resetCamera();
    const second = store().cameraCommand;
    expect(first?.kind).toBe("reset");
    expect(second?.nonce).toBe((first?.nonce ?? 0) + 1);
  });
});
