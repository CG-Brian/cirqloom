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
  /** A focus leaves the user away from the canonical view; a reset brings them back to it. */
  kind: "reset" | "focus";
}

export function CameraController({ bounds }: { bounds: Bounds }) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const camera = useThree((s) => s.camera);
  const aspect = useThree((s) => (s.size.height > 0 ? s.size.width / s.size.height : 1));
  const command = useExplorerStore((s) => s.cameraCommand);
  const nonce = command?.nonce ?? 0;
  const goal = useRef<Goal | null>(null);
  /** True once the user has moved the camera away from the canonical view. */
  const userMoved = useRef(false);

  const pose = useMemo(() => canonicalCameraPose(bounds, aspect), [bounds, aspect]);

  // Keep the canonical view fitted to the canvas (first size, panels collapsing, window
  // resizes) until the user moves the camera. After that, resizes never move their view.
  useEffect(() => {
    const c = controls.current;
    if (userMoved.current || goal.current || !c) {
      return;
    }
    camera.position.set(...pose.position);
    c.target.set(...pose.target);
    c.update();
  }, [camera, pose]);

  useEffect(() => {
    if (command === null) {
      return;
    }
    const to = command.kind === "focus" ? canonicalCameraPose(command.bounds, aspect) : pose;
    goal.current = {
      position: new Vector3(...to.position),
      target: new Vector3(...to.target),
      kind: command.kind,
    };
    // The pose is read at the moment of the request; a later resize must not restart the animation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce]);

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
      // Back at the canonical view after a reset, so later resizes may refit it again. After a
      // focus the view is the user's: resizes leave it alone.
      userMoved.current = g.kind === "focus";
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
      // "start" also fires for a plain click, so decide on release whether the view moved.
      onEnd={() => {
        const c = controls.current;
        userMoved.current =
          !c ||
          camera.position.distanceTo(new Vector3(...pose.position)) > ARRIVED_EPSILON ||
          c.target.distanceTo(new Vector3(...pose.target)) > ARRIVED_EPSILON;
      }}
    />
  );
}
