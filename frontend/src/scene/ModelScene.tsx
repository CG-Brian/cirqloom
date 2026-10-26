// The 3D canvas. Subscribes only to the active circuit; hover and selection are read per head.
import { Canvas } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import type { GraphView } from "../features/experiment/graphView";
import { getCircuitOverlay } from "../features/overlays/selectors";
import { useExplorerStore } from "../features/selection/store";
import { CameraController } from "./CameraController";
import { CLICK_SLOP_PX } from "./HeadNode";
import { HoverTooltip } from "./HoverTooltip";
import { CAMERA_FOV, canonicalCameraPose, layerX } from "./layout";
import { LayerGroup } from "./LayerGroup";

const BACKGROUND = "#0b0f14";
const TOOLTIP_OFFSET_PX = 14;

export function ModelScene({ view }: { view: GraphView }) {
  const activeCircuitId = useExplorerStore((s) => s.activeCircuitId);
  const select = useExplorerStore((s) => s.select);

  const overlay = useMemo(() => getCircuitOverlay(view, activeCircuitId), [view, activeCircuitId]);
  // First-frame guess only; CameraController refits once the real canvas size is known.
  const initialPose = useMemo(() => canonicalCameraPose(view.bounds, 16 / 9), [view.bounds]);

  // onPointerMissed also fires after an orbit drag that ends over empty space.
  // Only a press that barely moved counts as "click on nothing → deselect".
  const pressedAt = useRef<{ x: number; y: number } | null>(null);
  const onPointerMissed = (e: MouseEvent) => {
    const start = pressedAt.current;
    if (start && Math.hypot(e.clientX - start.x, e.clientY - start.y) <= CLICK_SLOP_PX) {
      select(null);
    }
  };

  // The tooltip follows the pointer by direct style updates, not React state, so pointer
  // movement never re-renders anything.
  const tooltip = useRef<HTMLDivElement>(null);
  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = tooltip.current;
    if (el) {
      const box = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - box.left + TOOLTIP_OFFSET_PX;
      const y = e.clientY - box.top + TOOLTIP_OFFSET_PX;
      el.style.transform = `translate(${x}px, ${y}px)`;
    }
  };

  return (
    <div
      className="relative h-full w-full"
      data-testid="model-scene"
      onPointerDown={(e) => {
        pressedAt.current = { x: e.clientX, y: e.clientY };
      }}
      onPointerMove={onPointerMove}
    >
      <Canvas
        dpr={[1, 2]}
        camera={{ fov: CAMERA_FOV, position: initialPose.position, near: 0.1, far: 500 }}
        onPointerMissed={onPointerMissed}
      >
        <color attach="background" args={[BACKGROUND]} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[6, 10, 8]} intensity={0.9} />
        <CameraController bounds={view.bounds} />
        {view.nodesByLayer.map((nodes, layer) => (
          <LayerGroup
            key={layer}
            layer={layer}
            x={layerX(layer, view.layers)}
            headsPerLayer={view.headsPerLayer}
            nodes={nodes}
            overlay={overlay}
          />
        ))}
      </Canvas>
      <HoverTooltip ref={tooltip} view={view} overlay={overlay} />
    </div>
  );
}
