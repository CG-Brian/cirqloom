// One attention head. Reads only its own flags from the store and applies the style computed by
// the pure selectors. No highlight rules live here.
import { Billboard } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import { CircleGeometry, SphereGeometry, TorusGeometry } from "three";
import type { HeadView } from "../features/experiment/graphView";
import {
  getNodeDisplayStyle,
  type CircuitOverlay,
  type LinePattern,
} from "../features/overlays/selectors";
import { useExplorerStore } from "../features/selection/store";

/** Pointer travel (px) above which a press is treated as an orbit drag, not a click. */
export const CLICK_SLOP_PX = 4;

const HEAD_RADIUS = 0.16;
const ROLE_RING_RADIUS = 0.24;
const SELECTION_RING_RADIUS = 0.31;

// Shared by all 144 heads instead of one copy each.
const SPHERE = new SphereGeometry(HEAD_RADIUS, 24, 16);
const HALO = new CircleGeometry(ROLE_RING_RADIUS + 0.04, 40);
const SELECTION_RING = new TorusGeometry(SELECTION_RING_RADIUS, 0.018, 8, 64);

/** [startAngle, arcLength] pairs that draw each line pattern as torus arcs. */
function patternArcs(pattern: LinePattern): [number, number][] {
  const full = Math.PI * 2;
  const repeat = (count: number, parts: [number, number][]) =>
    Array.from({ length: count }, (_, i) =>
      parts.map(([offset, length]): [number, number] => [
        ((i + offset) * full) / count,
        (length * full) / count,
      ]),
    ).flat();
  switch (pattern) {
    case "solid":
      return [[0, full]];
    case "dashed":
      return repeat(8, [[0, 0.6]]);
    case "dotted":
      return repeat(16, [[0, 0.3]]);
    case "dash-dot":
      return repeat(6, [
        [0, 0.45],
        [0.65, 0.12],
      ]);
  }
}

const arcGeometries = new Map<number, TorusGeometry>();
function arcGeometry(arc: number): TorusGeometry {
  let geometry = arcGeometries.get(arc);
  if (!geometry) {
    geometry = new TorusGeometry(ROLE_RING_RADIUS, 0.012, 6, 48, arc);
    arcGeometries.set(arc, geometry);
  }
  return geometry;
}

interface HeadNodeProps {
  node: HeadView;
  overlay: CircuitOverlay | null;
}

export function HeadNode({ node, overlay }: HeadNodeProps) {
  const isSelected = useExplorerStore((s) => s.selectedNodeId === node.id);
  const isHovered = useExplorerStore((s) => s.hoveredNodeId === node.id);
  const select = useExplorerStore((s) => s.select);
  const hover = useExplorerStore((s) => s.hover);
  const unhover = useExplorerStore((s) => s.unhover);

  const style = getNodeDisplayStyle({ nodeId: node.id, isSelected, isHovered, overlay });
  const { roleRing, selectionRing } = style;

  const onClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > CLICK_SLOP_PX) {
      return;
    }
    e.stopPropagation();
    select(node.id);
  };
  const onPointerOver = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    hover(node.id);
    document.body.style.cursor = "pointer";
  };
  const onPointerOut = () => {
    unhover(node.id);
    document.body.style.cursor = "";
  };

  return (
    <group position={node.position} scale={style.scale}>
      <mesh
        geometry={SPHERE}
        onClick={onClick}
        onPointerOver={onPointerOver}
        onPointerOut={onPointerOut}
      >
        <meshStandardMaterial
          color={style.color}
          transparent
          opacity={style.opacity}
          roughness={0.65}
          metalness={0.05}
        />
      </mesh>

      {(roleRing || selectionRing) && (
        <Billboard>
          {roleRing && (
            <>
              <mesh geometry={HALO} position={[0, 0, -0.01]} raycast={() => null}>
                <meshBasicMaterial
                  color={roleRing.color}
                  transparent
                  opacity={roleRing.haloOpacity}
                  depthWrite={false}
                />
              </mesh>
              {patternArcs(roleRing.pattern).map(([start, arc]) => (
                <mesh
                  key={start}
                  geometry={arcGeometry(arc)}
                  rotation={[0, 0, start]}
                  raycast={() => null}
                >
                  <meshBasicMaterial
                    color={roleRing.color}
                    transparent
                    opacity={roleRing.ringOpacity}
                  />
                </mesh>
              ))}
            </>
          )}
          {selectionRing && (
            <mesh geometry={SELECTION_RING} raycast={() => null}>
              <meshBasicMaterial color={selectionRing.color} transparent opacity={selectionRing.opacity} />
            </mesh>
          )}
        </Billboard>
      )}
    </group>
  );
}
