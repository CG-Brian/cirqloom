// Pure view geometry. Positions are derived from (layer, head) and never stored in data.

export const LAYER_SPACING = 2.2;
export const HEAD_SPACING = 0.7;
export const HEAD_COLUMNS = 4;

export type Vec3 = [number, number, number];

export interface Bounds {
  min: Vec3;
  max: Vec3;
  center: Vec3;
  radius: number;
}

export function gridShape(headsPerLayer: number): { rows: number; cols: number } {
  const cols = Math.min(HEAD_COLUMNS, headsPerLayer);
  return { rows: Math.ceil(headsPerLayer / cols), cols };
}

export function layerX(layer: number, nLayers: number): number {
  return (layer - (nLayers - 1) / 2) * LAYER_SPACING;
}

/** Layers run left to right along x; H0 sits top-left of its layer; the scene is centred. */
export function headPosition(
  layer: number,
  head: number,
  nLayers: number,
  headsPerLayer: number,
): Vec3 {
  const { rows, cols } = gridShape(headsPerLayer);
  const row = Math.floor(head / cols);
  const col = head % cols;
  return [
    layerX(layer, nLayers),
    ((rows - 1) / 2 - row) * HEAD_SPACING,
    (col - (cols - 1) / 2) * HEAD_SPACING,
  ];
}

export function layerFrameSize(headsPerLayer: number): { height: number; depth: number } {
  const { rows, cols } = gridShape(headsPerLayer);
  return { height: rows * HEAD_SPACING, depth: cols * HEAD_SPACING };
}

export function boundsOf(points: Vec3[]): Bounds {
  if (points.length === 0) {
    return { min: [0, 0, 0], max: [0, 0, 0], center: [0, 0, 0], radius: 0 };
  }
  const axis = (i: 0 | 1 | 2) => points.map((p) => p[i]);
  const min: Vec3 = [Math.min(...axis(0)), Math.min(...axis(1)), Math.min(...axis(2))];
  const max: Vec3 = [Math.max(...axis(0)), Math.max(...axis(1)), Math.max(...axis(2))];
  const center: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const radius = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2;
  return { min, max, center, radius };
}

export const CAMERA_FOV = 15;

/** From the target toward the camera: front-right and above, so layer planes read in 3D. */
export const VIEW_DIRECTION: Vec3 = [0.9, 0.55, 1];

/** Extra room around head centres for the head spheres, rings and layer labels. */
const FIT_PADDING = 0.7;

export interface CameraPose {
  position: Vec3;
  target: Vec3;
  distance: number;
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalize = (v: Vec3): Vec3 => {
  const len = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / len, v[1] / len, v[2] / len];
};

/**
 * A reproducible pose looking at the centre of `bounds` from VIEW_DIRECTION, at the smallest
 * distance where every (padded) corner of the box fits the frustum for this aspect ratio.
 */
export function canonicalCameraPose(
  bounds: Bounds,
  aspect: number,
  fovDeg = CAMERA_FOV,
  margin = 1.05,
): CameraPose {
  const back = normalize(VIEW_DIRECTION);
  const right = normalize(cross([0, 1, 0], back));
  const up = cross(back, right);
  const tanV = Math.tan((fovDeg * Math.PI) / 360);
  const tanH = tanV * aspect;

  const [cx, cy, cz] = bounds.center;
  let distance = 0;
  for (const x of [bounds.min[0] - FIT_PADDING, bounds.max[0] + FIT_PADDING]) {
    for (const y of [bounds.min[1] - FIT_PADDING, bounds.max[1] + FIT_PADDING]) {
      for (const z of [bounds.min[2] - FIT_PADDING, bounds.max[2] + FIT_PADDING]) {
        const v: Vec3 = [x - cx, y - cy, z - cz];
        const depth = dot(v, back);
        distance = Math.max(
          distance,
          depth + Math.abs(dot(v, right)) / tanH,
          depth + Math.abs(dot(v, up)) / tanV,
        );
      }
    }
  }
  distance *= margin;

  return {
    position: [cx + back[0] * distance, cy + back[1] * distance, cz + back[2] * distance],
    target: bounds.center,
    distance,
  };
}
