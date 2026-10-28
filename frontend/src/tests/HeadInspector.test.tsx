import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import type { DataProvider, HeadAttention } from "../data/DataProvider";
import type { Experiment, Graph, Manifest } from "../data/schema";
import { getVisualEdgeGroups, type VisualEdgeGroup } from "../features/circuits/edgeGroups";
import { buildGraphView } from "../features/experiment/graphView";
import { initialExplorerState, useExplorerStore } from "../features/selection/store";
import { HeadInspector } from "../panels/HeadInspector";
import { loadFixture, loadMeasured } from "./fixtureView";

const fixture = loadFixture();

/** 8 tokens; every row attends fully to its previous position (row 0 to itself). */
function toyAttention(experimentId: string, nodeId: string): HeadAttention {
  const tokens = ["<|endoftext|>", " a", " b", " c", " a", " b", " c", "\n"];
  const weights = tokens.map((_, q) => tokens.map((_, k) => (k === Math.max(0, q - 1) ? 1 : 0)));
  return { experimentId, nodeId, promptId: `${experimentId}:seed0`, tokens, weights };
}

const stubProvider = {
  loadHeadAttention: (experimentId: string, nodeId: string) =>
    Promise.resolve(toyAttention(experimentId, nodeId)),
} as unknown as DataProvider;

function renderInspector(
  view = fixture.view,
  manifest: Manifest = fixture.manifest,
  experiment: Experiment | null = null,
  provider: DataProvider = stubProvider,
  edgeGroups: VisualEdgeGroup[] = [],
) {
  return render(
    <HeadInspector
      provider={provider}
      edgeGroups={edgeGroups}
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

  it("draws the stored attention after it loads, with the prompt caveat", async () => {
    useExplorerStore.setState({ selectedNodeId: nodeId });
    renderInspector(view, manifest, expA);
    const attention = section("Attention");
    expect(within(attention).getByText(/Loading attention/)).toBeInTheDocument();
    expect(await within(attention).findByRole("img", { name: /8 by 8/ })).toBeInTheDocument();
    expect(within(attention).getByText(/1 of 2/)).toBeInTheDocument();
    expect(within(attention).getByText(/No mask shown/)).toBeInTheDocument();
  });

  it("reads a cell from the keyboard: real tokens, exact weight, future cells not zero", async () => {
    useExplorerStore.setState({ selectedNodeId: nodeId });
    renderInspector(view, manifest, expA);
    const map = await screen.findByRole("img", { name: /8 by 8/ });
    const readout = screen.getByTestId("heatmap-readout");
    fireEvent.keyDown(map, { key: "ArrowDown" }); // (1, 0)
    expect(readout).toHaveTextContent("q 1 “␣a” ← k 0 “<|endoftext|>”");
    expect(readout).toHaveTextContent("weight 1.00");
    fireEvent.keyDown(map, { key: "ArrowRight" });
    fireEvent.keyDown(map, { key: "ArrowRight" }); // (1, 2): future
    expect(readout).toHaveTextContent("future position: not attendable");
    expect(readout).not.toHaveTextContent("weight");
  });

  it("shows a load failure instead of a chart", async () => {
    const failing = {
      loadHeadAttention: () => Promise.reject(new Error("HTTP 404")),
    } as unknown as DataProvider;
    useExplorerStore.setState({ selectedNodeId: nodeId });
    renderInspector(view, manifest, expA, failing);
    expect(await screen.findByRole("alert")).toHaveTextContent("HTTP 404");
    expect(section("Attention").querySelector("canvas")).toBeNull();
  });

  it("offers the measurement run from a control run", () => {
    useExplorerStore.setState({ selectedNodeId: nodeId });
    renderInspector(view, manifest, expB);
    fireEvent.click(within(section("Attention")).getByRole("button", { name: "Show exp-a" }));
    expect(useExplorerStore.getState().experimentId).toBe("exp-a");
  });
});

describe("HeadInspector detector masks", () => {
  const nodeId = "model:gpt2-small:head:L0H0";
  const expA: Experiment = {
    id: "exp-a",
    prompt_suite: "t",
    kind: "measurement",
    description: "d",
    seeds: [0, 1],
    sequence_length: 8,
    tokens_uri: "t.json",
    attention: { uri: "a.f16", dtype: "float16", shape: [12, 12, 8, 8], prompt_index: 0 },
  };
  const observation = (detector_id: string) => ({
    node_id: nodeId,
    experiment_id: "exp-a",
    detector_id,
    detector_version: "1.0.0",
    evidence_type: "heuristic_attention" as const,
    parameters: {},
    score: 0.5,
    score_std: 0.1,
    per_example_scores: [0.4, 0.6],
    status: "experimental" as const,
    sample_count: 2,
    method_reference: "test",
  });
  const graph: Graph = {
    ...fixture.graph,
    observations: [observation("previous_token"), observation("no_mask_detector")],
    masks: [
      {
        experiment_id: "exp-a",
        detector_id: "previous_token",
        detector_version: "1.0.0",
        parameters: {},
        cells_qk: [[2, 1], [3, 2], [4, 3]],
      },
    ],
  };
  const manifest: Manifest = { ...fixture.manifest, experiments: [expA] };
  const view = buildGraphView(manifest, graph);

  it("toggles the exported mask onto the heatmap and marks cells in it", async () => {
    useExplorerStore.setState({ selectedNodeId: nodeId });
    renderInspector(view, manifest, expA);
    await screen.findByRole("img", { name: /8 by 8/ });
    const toggle = screen.getByRole("button", { name: /cells averaged by previous_token/ });
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/3 cells averaged by/)).toBeInTheDocument();

    const map = screen.getByRole("img", { name: /outlined: 3 cells averaged by previous_token v1\.0\.0/ });
    fireEvent.keyDown(map, { key: "ArrowDown" });
    fireEvent.keyDown(map, { key: "ArrowDown" });
    fireEvent.keyDown(map, { key: "ArrowRight" }); // (2, 1)
    expect(screen.getByTestId("heatmap-readout")).toHaveTextContent("in mask");

    fireEvent.click(toggle);
    expect(screen.getByText(/No mask shown/)).toBeInTheDocument();
  });

  it("says when a detector has no recorded cells instead of drawing one", async () => {
    useExplorerStore.setState({ selectedNodeId: nodeId, maskDetectorId: "no_mask_detector" });
    renderInspector(view, manifest, expA);
    await screen.findByRole("img", { name: /8 by 8/ });
    expect(screen.getByText("Cells not recorded for this detector.")).toBeInTheDocument();
    expect(screen.getByText(/no mask is drawn/)).toBeInTheDocument();
    expect(screen.getByText(/No mask shown/)).toBeInTheDocument();
  });
});

describe("HeadInspector on the GPT-2 bundle with the published IOI circuit", () => {
  const measured = loadMeasured();
  const repeatedL10 = measured.manifest.experiments.find((e) => e.id === "repeated_tokens_v1:L10")!;

  it("shows the published role and the repeated-token scores as separate kinds of evidence", () => {
    useExplorerStore.setState({ selectedNodeId: "model:gpt2-small:head:L5H5" });
    renderInspector(measured.view, measured.manifest, repeatedL10);
    const membership = section("Circuit membership");
    expect(within(membership).getByText("Induction")).toBeInTheDocument();
    expect(within(membership).getByText("Published annotation")).toBeInTheDocument();
    expect(within(membership).getByText("Published")).toBeInTheDocument();
    expect(within(section("Published annotations")).getByText(/arXiv:2211\.00593v1/)).toBeInTheDocument();

    const scores = section("Measured observations");
    expect(within(scores).getAllByText("Experimental").length).toBeGreaterThan(0);
    expect(within(scores).getByTestId("published-vs-measured")).toHaveTextContent(
      "measured here on repeated_tokens_v1 prompts. They do not test this head's published role (Induction in Indirect Object Identification (IOI))",
    );
  });

  it("shows the paper's note on a fuzzy member", () => {
    useExplorerStore.setState({ selectedNodeId: "model:gpt2-small:head:L0H10" });
    renderInspector(measured.view, measured.manifest, repeatedL10);
    expect(within(section("Circuit membership")).getByText(/Fuzzy Duplicate Token Head/)).toBeInTheDocument();
  });

  it("adds no published-vs-measured note to a head outside every circuit", () => {
    useExplorerStore.setState({ selectedNodeId: "model:gpt2-small:head:L1H1" });
    renderInspector(measured.view, measured.manifest, repeatedL10);
    expect(screen.queryByTestId("published-vs-measured")).toBeNull();
  });
});

describe("Relation inspector (IOI)", () => {
  const measured = loadMeasured();
  const groups = getVisualEdgeGroups(measured.view, "circuit:gpt2-small:ioi:wang-2022");
  const groupWhere = (locator: string) => groups.find((g) => g.locator?.includes(locator))!;

  it("opens for a selected relation and lists every original record with its locator", () => {
    const fig4b = groupWhere("Figure 4b");
    useExplorerStore.setState({ selectedEdgeGroupId: fig4b.id });
    renderInspector(measured.view, measured.manifest, null, stubProvider, groups);
    const panel = screen.getByRole("complementary", { name: "Relation inspector" });
    expect(within(panel).getByText("S-Inhibition heads → Name Mover queries")).toBeInTheDocument();
    expect(within(panel).getByText("Published experiment")).toBeInTheDocument();
    expect(within(panel).getByText(/Where: §3\.2, Figure 4b/)).toBeInTheDocument();
    const records = within(panel).getByRole("list", { name: "Evidence records" });
    expect(within(records).getAllByRole("listitem")).toHaveLength(4);
    for (const head of ["L7H3", "L7H9", "L8H6", "L8H10"]) {
      expect(within(records).getByText(new RegExp(`^${head} \\(S-Inhibition\\) → Name Mover$`))).toBeInTheDocument();
    }
    expect(within(panel).getByTestId("relation-granularity")).toHaveTextContent(
      "Measured one sender head at a time into the whole Name Mover group (queries). 4 records, one per sender head. The source reports no result for any single sender–receiver head pair",
    );
  });

  it("states the group-level experiment and the knockout condition honestly", () => {
    useExplorerStore.setState({ selectedEdgeGroupId: groupWhere("Figure 4c").id });
    const { unmount } = renderInspector(measured.view, measured.manifest, null, stubProvider, groups);
    expect(screen.getByTestId("relation-granularity")).toHaveTextContent(
      "Measured with all S-Inhibition heads changed together into the whole Name Mover group (queries). One record.",
    );
    unmount();
    useExplorerStore.setState({ selectedEdgeGroupId: groupWhere("Figure 15").id });
    renderInspector(measured.view, measured.manifest, null, stubProvider, groups);
    expect(screen.getByText("Only after the Name Mover heads are knocked out.")).toBeInTheDocument();
    expect(screen.getByText(/Not recorded \(never read off a figure\)/)).toBeInTheDocument();
  });

  it("shows the sign of a negative head's own contribution, not of the patching effect", () => {
    useExplorerStore.setState({ selectedEdgeGroupId: groups.find((g) => g.effect?.direction === "decrease")!.id });
    renderInspector(measured.view, measured.manifest, null, stubProvider, groups);
    expect(screen.getByText("The sender's own contribution decreases ioi logit difference.")).toBeInTheDocument();
  });

  it("goes from a record to its head, and from a head to its relations", () => {
    useExplorerStore.setState({ selectedEdgeGroupId: groupWhere("Figure 4b").id });
    renderInspector(measured.view, measured.manifest, null, stubProvider, groups);
    fireEvent.click(screen.getAllByRole("button", { name: "Inspect head" })[0]!);
    expect(useExplorerStore.getState()).toMatchObject({
      selectedNodeId: "model:gpt2-small:head:L7H3",
      selectedEdgeGroupId: null,
    });
    const relations = section("Relations in the shown circuit");
    fireEvent.click(within(relations).getByRole("button", { name: "S-Inhibition group → Name Mover queries" }));
    expect(useExplorerStore.getState().selectedEdgeGroupId).toBe(groupWhere("Figure 4c").id);
  });
});
