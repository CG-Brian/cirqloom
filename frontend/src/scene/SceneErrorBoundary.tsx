// Contains a failure of the 3D view (most often: WebGL is unavailable or the context could not
// be created) to the scene area. The head list, relation list and inspectors live outside this
// boundary and keep working, so every head and relation stays reachable without the canvas.
import { Component, type ReactNode } from "react";

interface SceneErrorBoundaryProps {
  children: ReactNode;
}

interface SceneErrorBoundaryState {
  error: Error | null;
}

export class SceneErrorBoundary extends Component<SceneErrorBoundaryProps, SceneErrorBoundaryState> {
  state: SceneErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): SceneErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error) {
    // Surface the cause for debugging; the UI below states it as well.
    console.warn("3D view unavailable:", error);
  }

  render() {
    const { error } = this.state;
    if (error === null) {
      return this.props.children;
    }
    return (
      <div
        role="alert"
        data-testid="scene-fallback"
        className="flex h-full w-full flex-col items-center justify-center gap-2 bg-[#0b0f14] p-6 text-center"
      >
        <p className="text-sm font-semibold text-slate-200">The 3D view could not start</p>
        <p className="max-w-md text-xs leading-relaxed text-slate-400">
          This browser or device did not provide WebGL. Everything else still works: pick heads
          from the head list, open circuit relations from the sidebar, and read scores, heatmaps
          and sources in the inspector.
        </p>
        <p className="max-w-md font-mono text-[10px] break-words text-slate-500">{error.message}</p>
      </div>
    );
  }
}
