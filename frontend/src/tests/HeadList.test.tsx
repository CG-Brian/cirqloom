import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { getCircuitOverlay } from "../features/overlays/selectors";
import { initialExplorerState, useExplorerStore } from "../features/selection/store";
import { useEscapeToDeselect } from "../features/selection/useEscapeToDeselect";
import { HeadList } from "../panels/HeadList";
import { loadFixture } from "./fixtureView";

const { view } = loadFixture();

function Harness() {
  useEscapeToDeselect();
  return <HeadList view={view} overlay={getCircuitOverlay(view, null)} />;
}

const head = (label: string) => screen.getByRole("button", { name: new RegExp(`^${label}\\b`) });

beforeEach(() => {
  useExplorerStore.setState(initialExplorerState);
});

describe("HeadList", () => {
  it("renders one button per head, grouped by layer", () => {
    render(<Harness />);
    expect(screen.getAllByRole("button")).toHaveLength(144);
    expect(screen.getAllByRole("group", { name: /^Layer \d+$/ })).toHaveLength(12);
  });

  it("names circuit roles in the accessible label", () => {
    render(<Harness />);
    expect(head("L3H3")).toHaveAccessibleName("L3H3, Fixture role B in Demo fixture circuit");
    expect(head("L3H4")).toHaveAccessibleName("L3H4");
  });

  it("selects on click and writes to the shared store", async () => {
    render(<Harness />);
    await userEvent.click(head("L2H5"));
    expect(head("L2H5")).toHaveAttribute("aria-pressed", "true");
    expect(useExplorerStore.getState().selectedNodeId).toBe("model:gpt2-small:head:L2H5");
  });

  it("reflects a selection made elsewhere (e.g. the 3D scene)", () => {
    render(<Harness />);
    act(() => useExplorerStore.getState().select("model:gpt2-small:head:L7H1"));
    expect(head("L7H1")).toHaveAttribute("aria-pressed", "true");
    expect(head("L7H1")).toHaveAttribute("tabindex", "0");
  });

  it("is a single tab stop", () => {
    render(<Harness />);
    const tabbable = screen.getAllByRole("button").filter((b) => b.tabIndex === 0);
    expect(tabbable).toHaveLength(1);
  });

  it("moves and selects with arrow keys, Home and End", async () => {
    render(<Harness />);
    await userEvent.click(head("L4H4"));
    await userEvent.keyboard("{ArrowRight}");
    expect(head("L4H5")).toHaveFocus();
    expect(head("L4H5")).toHaveAttribute("aria-pressed", "true");
    await userEvent.keyboard("{ArrowDown}");
    expect(head("L5H5")).toHaveAttribute("aria-pressed", "true");
    await userEvent.keyboard("{End}");
    expect(head("L5H11")).toHaveAttribute("aria-pressed", "true");
    await userEvent.keyboard("{Home}");
    expect(head("L5H0")).toHaveAttribute("aria-pressed", "true");
  });

  it("stops at the edges instead of wrapping", async () => {
    render(<Harness />);
    await userEvent.click(head("L0H0"));
    await userEvent.keyboard("{ArrowUp}{ArrowLeft}");
    expect(head("L0H0")).toHaveAttribute("aria-pressed", "true");
  });

  it("clears the selection on Escape", async () => {
    render(<Harness />);
    await userEvent.click(head("L1H1"));
    await userEvent.keyboard("{Escape}");
    expect(head("L1H1")).toHaveAttribute("aria-pressed", "false");
    expect(useExplorerStore.getState().selectedNodeId).toBeNull();
  });
});
