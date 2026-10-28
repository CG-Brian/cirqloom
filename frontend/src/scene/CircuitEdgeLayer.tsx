// The active circuit as a graph: role-group hubs above the layers, evidence-typed ribbons between
// them, and faint spokes from each hub down to the heads it stands for. Draws only what
// getCircuitLayout computed; no relation is derived here.
//
// A ribbon is drawn between group anchors for readability. It is not a physical wire between
// heads and its width is not an amount of information (see the relation inspector).
import { Line } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import { useMemo } from "react";
import {
  BoxGeometry,
  ConeGeometry,
  OctahedronGeometry,
  Quaternion,
  SphereGeometry,
  Vector3,
  type BufferGeometry,
} from "three";
import type { CircuitLayout } from "../features/circuits/circuitLayout";
import {
  CHANNEL_LABEL,
  getEdgeDisplayStyle,
  type EdgeFocusState,
  type VisualEdgeGroup,
} from "../features/circuits/edgeGroups";
import type { GraphView } from "../features/experiment/graphView";
import type { MarkerShape } from "../features/overlays/selectors";
import { useExplorerStore } from "../features/selection/store";
import { CLICK_SLOP_PX } from "./HeadNode";
import { LayerLabel } from "./LayerLabel";
import type { Vec3 } from "./layout";

const HUB_GEOMETRY: Record<MarkerShape, BufferGeometry> = {
  circle: new SphereGeometry(0.11, 20, 14),
  square: new BoxGeometry(0.18, 0.18, 0.18),
  triangle: new ConeGeometry(0.12, 0.2, 3),
  diamond: new OctahedronGeometry(0.13),
};
const SUBSET_ANCHOR = new SphereGeometry(0.06, 12, 8);
const LOGITS_BOX = new BoxGeometry(0.3, 0.3, 0.6);
const ARROW = new ConeGeometry(0.06, 0.18, 12);
const UP = new Vector3(0, 1, 0);

/** Pixel hit width for ribbons: much wider than the drawn line, so they are easy to point at. */
const HIT_WIDTH_PX = 14;

const DASH: Record<"solid" | "dashed" | "dotted", { dashed: boolean; dashSize: number; gapSize: number }> = {
  solid: { dashed: false, dashSize: 1, gapSize: 0 },
  dashed: { dashed: true, dashSize: 0.16, gapSize: 0.1 },
  dotted: { dashed: true, dashSize: 0.04, gapSize: 0.09 },
};

function arrowQuaternion(direction: Vec3): Quaternion {
  return new Quaternion().setFromUnitVectors(UP, new Vector3(...direction).normalize());
}

function channelText(group: VisualEdgeGroup): string {
  const sign = group.effect ? (group.effect.direction === "increase" ? " +" : " −") : "";
  return `${CHANNEL_LABEL[group.channel].short}${sign}`;
}

interface CircuitEdgeLayerProps {
  view: GraphView;
  groups: VisualEdgeGroup[];
  layout: CircuitLayout;
}

export function CircuitEdgeLayer({ view, groups, layout }: CircuitEdgeLayerProps) {
  const hovered = useExplorerStore((s) => s.hoveredEdgeGroupId);
  const selected = useExplorerStore((s) => s.selectedEdgeGroupId);
  const hoverEdgeGroup = useExplorerStore((s) => s.hoverEdgeGroup);
  const unhoverEdgeGroup = useExplorerStore((s) => s.unhoverEdgeGroup);
  const selectEdgeGroup = useExplorerStore((s) => s.selectEdgeGroup);
  const focused = hovered ?? selected;

  const groupById = useMemo(() => new Map(groups.map((g) => [g.id, g])), [groups]);
  const focusedGeometry = layout.edges.find((e) => e.groupId === focused);
  const arrows = useMemo(
    () => new Map(layout.edges.map((e) => [e.groupId, arrowQuaternion(e.arrow.direction)])),
    [layout],
  );

  const stateOf = (groupId: string): EdgeFocusState =>
    groupId === focused || groupId === selected ? "focused" : focused ? "dimmed" : "idle";

  return (
    <group>
      {[...layout.anchors.values()].map((anchor) => {
        const active = focusedGeometry?.from === anchor.key || focusedGeometry?.to === anchor.key;
        const opacity = focused && !active ? 0.25 : 0.9;
        return (
          <group key={anchor.key}>
            {anchor.kind === "role_group" && anchor.style && (
              <mesh geometry={HUB_GEOMETRY[anchor.style.marker]} position={anchor.position} raycast={() => null}>
                <meshBasicMaterial color={anchor.style.color} transparent opacity={opacity} />
              </mesh>
            )}
            {anchor.kind === "subset" && (
              <mesh geometry={SUBSET_ANCHOR} position={anchor.position} raycast={() => null}>
                <meshBasicMaterial color="#cbd5e1" transparent opacity={opacity} />
              </mesh>
            )}
            {anchor.kind === "special" && (
              <>
                <mesh geometry={LOGITS_BOX} position={anchor.position} raycast={() => null}>
                  <meshBasicMaterial color="#94a3b8" transparent opacity={opacity * 0.6} />
                </mesh>
                <LayerLabel
                  text={anchor.label}
                  position={[anchor.position[0], anchor.position[1] + 0.35, anchor.position[2]]}
                  height={0.22}
                  color="#cbd5e1"
                />
              </>
            )}
            {/* Spokes: which heads this hub stands for. Membership, not a relation. */}
            {anchor.nodeIds.map((id) => {
              const head = view.nodeById.get(id);
              return head ? (
                <Line
                  key={id}
                  points={[anchor.position, head.position]}
                  color={anchor.style?.color ?? "#cbd5e1"}
                  lineWidth={1}
                  dashed
                  dashSize={0.05}
                  gapSize={0.07}
                  transparent
                  opacity={active ? 0.55 : focused ? 0.06 : 0.16}
                  raycast={() => null}
                />
              ) : null;
            })}
          </group>
        );
      })}

      {layout.edges.map((geometry) => {
        const group = groupById.get(geometry.groupId);
        if (!group) {
          return null;
        }
        const state = stateOf(group.id);
        const style = getEdgeDisplayStyle(group, state);
        const dash = DASH[style.pattern];
        const onClick = (e: ThreeEvent<MouseEvent>) => {
          if (e.delta > CLICK_SLOP_PX) {
            return;
          }
          e.stopPropagation();
          selectEdgeGroup(group.id);
        };
        return (
          <group key={group.id}>
            <Line
              points={geometry.points}
              color={style.color}
              lineWidth={style.lineWidth}
              dashed={dash.dashed}
              dashSize={dash.dashSize}
              gapSize={dash.gapSize}
              transparent
              opacity={style.opacity}
              raycast={() => null}
            />
            {/* Invisible, wide twin that receives the pointer. */}
            <Line
              points={geometry.points}
              lineWidth={HIT_WIDTH_PX}
              transparent
              opacity={0}
              depthWrite={false}
              userData={{ edgeGroupId: group.id }}
              onClick={onClick}
              onPointerOver={(e) => {
                e.stopPropagation();
                hoverEdgeGroup(group.id);
                document.body.style.cursor = "pointer";
              }}
              onPointerOut={() => {
                unhoverEdgeGroup(group.id);
                document.body.style.cursor = "";
              }}
            />
            <mesh
              geometry={ARROW}
              position={geometry.arrow.position}
              quaternion={arrows.get(group.id)}
              raycast={() => null}
            >
              <meshBasicMaterial color={style.color} transparent opacity={style.opacity} />
            </mesh>
            <LayerLabel
              text={channelText(group)}
              position={[geometry.mid[0], geometry.mid[1] + 0.14, geometry.mid[2]]}
              height={0.2}
              color={state === "focused" ? "#f8fafc" : "#94a3b8"}
            />
          </group>
        );
      })}
    </group>
  );
}
