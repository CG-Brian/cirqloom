// One transformer layer: a slim translucent frame, its label, and its heads.
// The frame carries the circuit-level emphasis; heads carry only subtle per-member rings.
import { useEffect, useMemo } from "react";
import { DoubleSide, EdgesGeometry, PlaneGeometry } from "three";
import type { HeadView } from "../features/experiment/graphView";
import { getLayerFrameStyle, type CircuitOverlay } from "../features/overlays/selectors";
import { HeadNode } from "./HeadNode";
import { LayerLabel } from "./LayerLabel";
import { HEAD_SPACING, layerFrameSize } from "./layout";

interface LayerGroupProps {
  layer: number;
  x: number;
  headsPerLayer: number;
  nodes: HeadView[];
  overlay: CircuitOverlay | null;
  edgeFocusSets: Map<string, Set<string>>;
}

export function LayerGroup({ layer, x, headsPerLayer, nodes, overlay, edgeFocusSets }: LayerGroupProps) {
  const frame = getLayerFrameStyle(layer, overlay);
  const { height, depth } = layerFrameSize(headsPerLayer);

  // A single plane in the layer's y–z plane, so the translucent fill is drawn once, not twice.
  const [plane, edges] = useMemo(() => {
    const frameHeight = height + HEAD_SPACING * 0.4;
    const frameDepth = depth + HEAD_SPACING * 0.4;
    const fill = new PlaneGeometry(frameDepth, frameHeight).rotateY(Math.PI / 2);
    return [fill, new EdgesGeometry(fill)];
  }, [height, depth]);
  useEffect(
    () => () => {
      plane.dispose();
      edges.dispose();
    },
    [plane, edges],
  );

  return (
    <>
      <group position={[x, 0, 0]}>
        <mesh geometry={plane} raycast={() => null}>
          <meshBasicMaterial
            color={frame.fillColor}
            transparent
            opacity={frame.fillOpacity}
            side={DoubleSide}
            depthWrite={false}
          />
        </mesh>
        <lineSegments geometry={edges} raycast={() => null}>
          <lineBasicMaterial color={frame.edgeColor} transparent opacity={frame.edgeOpacity} />
        </lineSegments>
        <LayerLabel text={`L${layer}`} position={[0, -height / 2 - HEAD_SPACING * 0.6, depth / 2]} />
      </group>
      {nodes.map((node) => (
        <HeadNode key={node.id} node={node} overlay={overlay} edgeFocusSets={edgeFocusSets} />
      ))}
    </>
  );
}
