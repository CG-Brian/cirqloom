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
- draw edges between heads that the data does not contain;
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

- **Desktop:** top bar, then circuit sidebar (256 px) | 3D scene | head inspector (288 px), then status bar. Both panels collapse to a 36 px rail.
- **Narrow screens (< 1024 px):** the scene is on top (55 vh), with the sidebar and inspector stacked below and no horizontal scroll.
- **Camera:** 15° FOV (a long lens, so all layer planes read alike), canonical view from the front-right and above, fitted to the canvas aspect. The view refits on resize until the user moves the camera.

## Interaction

- Click selects; a press that moves more than 4 px is an orbit drag, not a click. A click on empty space clears the selection.
- The head list is one Tab stop with a roving tabindex. Arrow keys move and select, Home/End jump within a layer. Escape clears the selection anywhere.
- 3D, list and inspector all read the same `selectedNodeId` from the store.
- The hover tooltip is a DOM overlay outside the canvas. Do not use drei `<Html>` inside the canvas: under React 19 StrictMode it errors.

## Inspector sections

1. **Identity:** stable id, model, revision ("none — no weights loaded" for fixtures), layer and head (zero-based).
2. **Circuit membership:** role marker, role, circuit, origin badge (Synthetic fixture / Published annotation), evidence-status badge, source.
3. **Published annotations:** only memberships whose circuit is published, with citation. Otherwise an explicit "none" note.
4. **Measured observations:** detector, version, status, score (or "Not measured"), n, evidence type, experiment, method.
5. **Attention:** a heatmap only when the bundle has measured runs; otherwise an explicit "unavailable" note.
6. **What this does not show:** a short causal caveat.

Every status badge has a text label; dashed and dotted borders mark non-measured states.
