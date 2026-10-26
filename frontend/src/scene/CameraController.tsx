// Orbit / pan / zoom plus a smooth "reset view". Camera pose lives here, not in the store,
// so moving the camera never re-renders React UI.
import { OrbitControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type ComponentRef } from "react";
import { Vector3 } from "three";
import { useExplorerStore } from "../features/selection/store";
import { canonicalCameraPose, type Bounds } from "./layout";

const SMOOTHING_RATE = 6;
const ARRIVED_EPSILON = 0.01;

interface Goal {
  position: Vector3;
  target: Vector3;
}

export function CameraController({ bounds }: { bounds: Bounds }) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const camera = useThree((s) => s.camera);
  const aspect = useThree((s) => (s.size.height > 0 ? s.size.width / s.size.height : 1));
  const resetNonce = useExplorerStore((s) => s.cameraCommand?.nonce ?? 0);
  const goal = useRef<Goal | null>(null);
  const placed = useRef(false);

  const pose = useMemo(() => canonicalCameraPose(bounds, aspect), [bounds, aspect]);

  // Snap to the canonical view once, when the real canvas size is first known.
  // Later resizes only change where "Reset view" goes; they never move the user's camera.
  useEffect(() => {
    const c = controls.current;
    if (placed.current || !c) {
      return;
    }
    camera.position.set(...pose.position);
    c.target.set(...pose.target);
    c.update();
    placed.current = true;
  }, [camera, pose]);

  useEffect(() => {
    if (resetNonce === 0) {
      return;
    }
    goal.current = {
      position: new Vector3(...pose.position),
      target: new Vector3(...pose.target),
    };
    // pose is read at the moment of the request; a later resize must not restart the animation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetNonce]);

  useFrame((_, delta) => {
    const g = goal.current;
    const c = controls.current;
    if (!g || !c) {
      return;
    }
    // Frame-rate independent exponential smoothing.
    const t = 1 - Math.exp(-SMOOTHING_RATE * delta);
    camera.position.lerp(g.position, t);
    c.target.lerp(g.target, t);
    if (
      camera.position.distanceTo(g.position) < ARRIVED_EPSILON &&
      c.target.distanceTo(g.target) < ARRIVED_EPSILON
    ) {
      camera.position.copy(g.position);
      c.target.copy(g.target);
      goal.current = null;
    }
    c.update();
  });

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={4}
      maxDistance={pose.distance * 2.5}
      // The user grabbing the camera always wins over a running animation.
      onStart={() => {
        goal.current = null;
      }}
    />
  );
}
