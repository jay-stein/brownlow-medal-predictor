import { useEffect, useState } from "react";

export default function useNarrow(maxWidth = 720) {
  const [narrow, setNarrow] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia(`(max-width: ${maxWidth}px)`).matches;
  });

  useEffect(() => {
    const query = window.matchMedia(`(max-width: ${maxWidth}px)`);
    const handler = (event) => setNarrow(event.matches);
    query.addEventListener("change", handler);
    setNarrow(query.matches);
    return () => query.removeEventListener("change", handler);
  }, [maxWidth]);

  return narrow;
}
