import { PerspectiveCamera, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import {
  boundsOf,
  CAMERA_FOV,
  canonicalCameraPose,
  headPosition,
  LAYER_SPACING,
} from "../scene/layout";

const LAYERS = 12;
const HEADS = 12;

function allPositions() {
  const out = [];
  for (let layer = 0; layer < LAYERS; layer++) {
    for (let head = 0; head < HEADS; head++) {
      out.push(headPosition(layer, head, LAYERS, HEADS));
    }
  }
  return out;
}

describe("layout", () => {
  it("gives all 144 heads distinct positions", () => {
    const keys = new Set(allPositions().map((p) => p.join(",")));
    expect(keys.size).toBe(LAYERS * HEADS);
  });

  it("is deterministic", () => {
    expect(allPositions()).toEqual(allPositions());
  });

  it("places layers left to right with fixed spacing", () => {
    for (let layer = 1; layer < LAYERS; layer++) {
      const prev = headPosition(layer - 1, 0, LAYERS, HEADS);
      const curr = headPosition(layer, 0, LAYERS, HEADS);
      expect(curr[0] - prev[0]).toBeCloseTo(LAYER_SPACING);
    }
  });

  it("puts H0 top-left within its layer", () => {
    const [, y0, z0] = headPosition(0, 0, LAYERS, HEADS);
    for (let head = 1; head < HEADS; head++) {
      const [, y, z] = headPosition(0, head, LAYERS, HEADS);
      expect(y).toBeLessThanOrEqual(y0);
      expect(z).toBeGreaterThanOrEqual(z0);
    }
  });

  it("centres the scene on the origin", () => {
    const { center } = boundsOf(allPositions());
    center.forEach((c) => expect(c).toBeCloseTo(0));
  });
});

describe("canonical camera pose", () => {
  const bounds = boundsOf(allPositions());

  it("looks at the centre of the scene from the stated distance", () => {
    const pose = canonicalCameraPose(bounds, 16 / 9);
    expect(pose.target).toEqual(bounds.center);
    const [px, py, pz] = pose.position;
    const [tx, ty, tz] = pose.target;
    expect(Math.hypot(px - tx, py - ty, pz - tz)).toBeCloseTo(pose.distance);
  });

  // Independent check with a real three.js camera: every head projects inside the screen.
  it.each([16 / 9, 1, 0.6])("keeps every head on screen at aspect %f", (aspect) => {
    const pose = canonicalCameraPose(bounds, aspect);
    const camera = new PerspectiveCamera(CAMERA_FOV, aspect, 0.1, 500);
    camera.position.set(...pose.position);
    camera.lookAt(...pose.target);
    camera.updateMatrixWorld();
    for (const position of allPositions()) {
      const ndc = new Vector3(...position).project(camera);
      expect(Math.abs(ndc.x)).toBeLessThan(1);
      expect(Math.abs(ndc.y)).toBeLessThan(1);
    }
  });

  it("is the same every time", () => {
    expect(canonicalCameraPose(bounds, 1.5)).toEqual(canonicalCameraPose(bounds, 1.5));
  });
});
