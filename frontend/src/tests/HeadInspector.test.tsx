import { act, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import type { Experiment, Graph, Manifest } from "../data/schema";
import { buildGraphView } from "../features/experiment/graphView";
import { initialExplorerState, useExplorerStore } from "../features/selection/store";
import { HeadInspector } from "../panels/HeadInspector";
import { loadFixture } from "./fixtureView";

const fixture = loadFixture();

function renderInspector(
  view = fixture.view,
  manifest: Manifest = fixture.manifest,
  experiment: Experiment | null = null,
) {
  return render(
    <HeadInspector
      view={view}
      manifest={manifest}
      experiment={experiment}
      collapsed={false}
      onToggleCollapsed={() => {}}
    />,
  );
}

const section = (title: string) =>
  screen.getByRole("heading", { name: title }).closest("section") as HTMLElement;

beforeEach(() => {
  useExplorerStore.setState(initialExplorerState);
});

describe("HeadInspector on the synthetic fixture", () => {
  it("asks for a selection when nothing is selected", () => {
    renderInspector();
    expect(screen.getByText(/select a head/i)).toBeInTheDocument();
  });

  it("shows identity and fixture membership for a member head", () => {
    useExplorerStore.setState({ selectedNodeId: "model:gpt2-small:head:L3H3" });
    renderInspector();
    expect(screen.getByText("model:gpt2-small:head:L3H3")).toBeInTheDocument();
    const membership = section("Circuit membership");
    expect(within(membership).getByText("Fixture role B")).toBeInTheDocument();
    expect(within(membership).getByText("Synthetic fixture")).toBeInTheDocument();
    expect(within(membership).getByText("Experimental")).toBeInTheDocument();
  });

  it("states that published annotations, observations and attention are absent", () => {
    useExplorerStore.setState({ selectedNodeId: "model:gpt2-small:head:L3H3" });
    renderInspector();
    expect(within(section("Published annotations")).getByText(/synthetic fixture/i)).toBeInTheDocument();
    expect(within(section("Measured observations")).getByText(/no experiments/i)).toBeInTheDocument();
    expect(within(section("Attention")).getByText(/no attention data/i)).toBeInTheDocument();
  });

  it("never shows a numeric score for the fixture", () => {
    useExplorerStore.setState({ selectedNodeId: "model:gpt2-small:head:L3H3" });
    renderInspector();
    expect(section("Measured observations").textContent).not.toMatch(/\d\.\d/);
    expect(section("Attention").querySelector("svg, canvas, table")).toBeNull();
  });

  it("says when a head belongs to no circuit", () => {
    useExplorerStore.setState({ selectedNodeId: "model:gpt2-small:head:L3H4" });
    renderInspector();
    expect(within(section("Circuit membership")).getByText(/not a member/i)).toBeInTheDocument();
  });

  it("follows selection changes from the store", () => {
    renderInspector();
    act(() => useExplorerStore.getState().select("model:gpt2-small:head:L9H2"));
    expect(screen.getByText("L9H2")).toBeInTheDocument();
  });
});

describe("HeadInspector with observations", () => {
  const nodeId = "model:gpt2-small:head:L0H0";
  const obs = {
    node_id: nodeId,
    experiment_id: "exp-a",
    detector_version: "1.0.0",
    evidence_type: "heuristic_attention" as const,
    method_reference: "test",
  };
  const experiment = (id: string, kind: Experiment["kind"]): Experiment => ({
    id,
    prompt_suite: "t",
    kind,
    description: `description of ${id}`,
    seeds: [0, 1],
    sequence_length: 8,
    tokens_uri: "t.json",
    attention: kind === "measurement" ? { uri: "a.f16", dtype: "float16", shape: [12, 12, 8, 8], prompt_index: 0 } : null,
  });
  const expA = experiment("exp-a", "measurement");
  const expB = experiment("exp-b", "control");
  const graph: Graph = {
    ...fixture.graph,
    observations: [
      {
        ...obs,
        detector_id: "previous_token",
        parameters: { burn_in: 2 },
        score: 0.8123,
        score_std: 0.0412,
        per_example_scores: Array.from({ length: 40 }, () => 0.8123),
        status: "measured",
        sample_count: 40,
      },
      {
        ...obs,
        detector_id: "first_token",
        parameters: {},
        score: null,
        score_std: null,
        per_example_scores: [],
        status: "unavailable",
        sample_count: 0,
      },
      {
        ...obs,
        experiment_id: "exp-b",
        detector_id: "induction",
        parameters: {},
        score: 0.0042,
        score_std: 0.001,
        per_example_scores: [0.004, 0.0044],
        status: "experimental",
        sample_count: 2,
      },
    ],
  };
  const manifest: Manifest = { ...fixture.manifest, experiments: [expA, expB] };
  const view = buildGraphView(manifest, graph);

  it("shows a measured score with its spread, sample count, settings and status", () => {
    useExplorerStore.setState({ selectedNodeId: nodeId });
    renderInspector(view, manifest, expA);
    const measured = section("Measured observations");
    expect(within(measured).getByText(/0\.812/)).toBeInTheDocument();
    expect(within(measured).getByText(/± 0\.041 sd/)).toBeInTheDocument();
    expect(within(measured).getByText(/n = 40 prompts/)).toBeInTheDocument();
    expect(within(measured).getByText("Settings: burn_in = 2")).toBeInTheDocument();
    expect(within(measured).getByText("Measured")).toBeInTheDocument();
  });

  it("shows an unavailable result as 'Not measured', never as zero", () => {
    useExplorerStore.setState({ selectedNodeId: nodeId });
    renderInspector(view, manifest, expA);
    const measured = section("Measured observations");
    expect(within(measured).getByText("Not measured")).toBeInTheDocument();
    expect(within(measured).getByText("Unavailable")).toBeInTheDocument();
    expect(measured.textContent).not.toMatch(/\b0\.0+\b/);
  });

  it("shows only the selected experiment's observations", () => {
    useExplorerStore.setState({ selectedNodeId: nodeId });
    renderInspector(view, manifest, expA);
    expect(within(section("Measured observations")).queryByText(/^induction/)).toBeNull();
  });

  it("labels a control run and warns that experimental scores are not causal", () => {
    useExplorerStore.setState({ selectedNodeId: nodeId });
    renderInspector(view, manifest, expB);
    const measured = section("Measured observations");
    expect(within(measured).getByText(/Control run/)).toBeInTheDocument();
    expect(within(measured).getByText(/not a causal role/)).toBeInTheDocument();
    expect(within(section("Attention")).getByText(/keeps scores only/)).toBeInTheDocument();
  });

  it("says attention is stored but does not draw a chart", () => {
    useExplorerStore.setState({ selectedNodeId: nodeId });
    renderInspector(view, manifest, expA);
    const attention = section("Attention");
    expect(within(attention).getByText(/seed 0/)).toBeInTheDocument();
    expect(attention.querySelector("svg, canvas, table")).toBeNull();
  });
});
