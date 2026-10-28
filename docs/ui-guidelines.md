# UI guidelines

The circuit, not the individual head, is the visual focus. Heads stay understated, and evidence lives in the 2D panels.

## Emphasis rules

All rules live in `frontend/src/features/overlays/selectors.ts` and are unit-tested. Components only apply them.

| Element | Default | Active circuit, member | Active circuit, non-member | Selected | Hovered |
| --- | --- | --- | --- | --- | --- |
| Head base colour | slate `#64748b` | unchanged | unchanged | unchanged | unchanged |
| Head opacity | 1 | 1 | 0.35 | 1 | 1 |
| Role ring | none | thin, role colour, 0.8 | none | kept | kept |
| Halo | none | role colour, 0.12 | none | kept | kept |
| Selection ring | none | — | — | neutral white `#f8fafc` | — |
| Scale | 1 | 1 | 1 | 1.08 | 1.15 |

Layer frames carry the circuit-level emphasis. A layer spanned by the active circuit gets an indigo outline (`#a5b4fc`), and other layers recede. Frame fill is always neutral.

Never:

- recolour a whole head by category;
- make heads glow, pulse or bloom;
- draw edges between heads that the data does not contain, or expand a group relation into head pairs;
- show a placeholder number or chart where data is missing.

## Role encoding

Roles are assigned by their index in the circuit's role list. Each colour (Okabe–Ito, colour-blind safe) is paired with a unique line pattern and marker shape, so roles stay distinguishable without colour:

| # | Colour | Pattern | Marker |
| --- | --- | --- | --- |
| 1 | `#56B4E9` sky | solid | circle |
| 2 | `#E69F00` orange | dashed | square |
| 3 | `#009E73` green | dotted | triangle |
| 4 | `#CC79A7` purple | dash-dot | diamond |
| 5 | `#0072B2` blue | dashed | circle |
| 6 | `#D55E00` vermillion | dotted | square |
| 7 | `#F0E442` yellow | dash-dot | triangle |

The 3D ring and the sidebar's `RoleMarker` use the same style.

## Layout

- **Desktop:** top bar, then circuit sidebar (256 px) | 3D scene | head inspector (384 px, wide enough for a 101 × 101 heatmap), then status bar. Both panels collapse to a 36 px rail.
- **Narrow screens (< 1024 px):** the scene is on top (55 vh), with the sidebar and inspector stacked below and no horizontal scroll.
- **Camera:** 15° FOV (a long lens, so all layer planes read alike), canonical view from the front-right and above, fitted to the canvas aspect. The view refits on resize until the user moves the camera.

## Interaction

- Click selects; a press that moves more than 4 px is an orbit drag, not a click. A click on empty space clears the selection.
- The head list is one Tab stop with a roving tabindex. Arrow keys move and select, Home/End jump within a layer. Escape clears the selection anywhere.
- 3D, list and inspector all read the same `selectedNodeId` from the store.
- The hover tooltip is a DOM overlay outside the canvas. Do not use drei `<Html>` inside the canvas: under React 19 StrictMode it errors.

## Circuit graph (relations)

- Data → drawing: `Graph.edges` (stored at the granularity of their evidence) → `getVisualEdgeGroups` (bundles records that agree on circuit, sender kind, receiver, channel, evidence kind and status, locator, effect, conditions and positions; never edits or splits a record) → `getCircuitLayout` (hubs, curves, spokes) → `CircuitEdgeLayer`.
- Role-group hubs float above the layer frames over the centroid of their members, in the role's colour and marker shape (circle → sphere, square → box, triangle → cone, diamond → octahedron). Senders that are a subset of a role get their own small neutral anchor over those heads. Faint dotted spokes link anchors to the heads they stand for (membership, not relations).
- Ribbons are curves between anchors, above the model, ending with an arrow at the receiver. Line pattern encodes evidence: solid = published experiment, dashed = published schematic, thinner = weight composition, thicker = causal (ours), dotted = holds only under a condition. Colour stays neutral. Relations between the same two anchors are lifted apart. A channel label (Q, K, V, logits ±) sits at the middle.
- Hover (3D or the sidebar list) emphasises the relation and the heads its evidence is about (sender heads, receiving group members); everything else recedes. Click opens the **Relation inspector** in place of the head inspector: sender, receiver, channel, positions, effect (the sender's own contribution), conditions, magnitude, a granularity statement, citation and locator, every original evidence record (with "Inspect head"), and what the ribbon does not show. A head's inspector lists the relations it takes part in.
- The sidebar's "Relations" list is the keyboard path to every ribbon. Escape or an empty-space click clears the selection.

## Inspector sections

1. **Identity:** stable id, model, revision ("none — no weights loaded" for fixtures), layer and head (zero-based).
2. **Circuit membership:** role marker, role, circuit, origin badge (Synthetic fixture / Published annotation), evidence-status badge, source.
3. **Published annotations:** only memberships whose circuit is published, with citation. Otherwise an explicit "none" note.
4. **Measured observations:** detector, version, status, score (or "Not measured"), n, evidence type, experiment, method.
5. **Attention:** a heatmap only when the selected experiment stores attention; otherwise an explicit note (no experiments; control run, with a link to the measurement run of the same length; loading; load error). See *Attention heatmap* below.
6. **What this does not show:** a short causal caveat.

## Attention heatmap

- One head, one example prompt (the experiment's stored `prompt_index`). Row = destination (query) position q, column = source (key) position k, both labelled on the axes; tokens appear in the readout.
- Colour: one fixed 0–1 scale for every head, with a legend. Never rescaled per head, so a weak head looks weak.
- Future cells (k > q) are hatched and read "not attendable": they are not weight 0.
- Readout below the map: `q, query token ← k, source token`, the exact weight (float16, 3 significant digits), and "in mask". Hover or focus the map and use the arrow keys. Whitespace is visible (`␣`, `↵`, `⇥`).
- Mask: "Show cells" on a detector row picks one detector (kept across heads and experiments). The map dims every cell outside that detector's exported mask and outlines mask cells when cells are at least 6 px. Only masks from the bundle are drawn; a missing mask is stated, never guessed. One mask at a time, so each outline belongs to one score.
- Caveat under the map: one prompt of n, so its cells need not average to the score; float16 precision; token-to-token attention inside one head, not a connection between heads.

Every status badge has a text label; dashed and dotted borders mark non-measured states.
