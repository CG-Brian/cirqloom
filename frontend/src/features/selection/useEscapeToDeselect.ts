import { useEffect } from "react";
import { useExplorerStore } from "./store";

/** Escape clears the head selection from anywhere on the page (AGENTS.md §9). */
export function useEscapeToDeselect(): void {
  const select = useExplorerStore((s) => s.select);
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !e.defaultPrevented) {
        select(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [select]);
}
