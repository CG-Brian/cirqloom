// A small camera-facing text label drawn into a canvas texture. Used instead of drei <Html>,
// which mounts a separate React root per label and errors under React 19 StrictMode.
import { useEffect, useMemo } from "react";
import { CanvasTexture, SRGBColorSpace } from "three";

const FONT_PX = 48;
const FONT = `${FONT_PX}px ui-monospace, SFMono-Regular, Menlo, monospace`;

function makeTexture(text: string, color: string): { texture: CanvasTexture; aspect: number } {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2D canvas is not available for scene labels");
  }
  ctx.font = FONT;
  canvas.width = Math.ceil(ctx.measureText(text).width + FONT_PX * 0.5);
  canvas.height = Math.ceil(FONT_PX * 1.4);
  ctx.font = FONT;
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return { texture, aspect: canvas.width / canvas.height };
}

interface LayerLabelProps {
  text: string;
  position: [number, number, number];
  /** Label height in world units. */
  height?: number;
  color?: string;
}

export function LayerLabel({ text, position, height = 0.32, color = "#94a3b8" }: LayerLabelProps) {
  const { texture, aspect } = useMemo(() => makeTexture(text, color), [text, color]);
  useEffect(() => () => texture.dispose(), [texture]);

  return (
    <sprite position={position} scale={[height * aspect, height, 1]} raycast={() => null}>
      <spriteMaterial map={texture} transparent depthWrite={false} />
    </sprite>
  );
}
