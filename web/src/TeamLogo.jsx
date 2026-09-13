import { useEffect, useState } from "react";
import { teamColor } from "./teams.js";

let logosPromise = null;

function loadLogos() {
  if (!logosPromise) {
    logosPromise = fetch(`${import.meta.env.BASE_URL}team-logos.json`)
      .then((response) => (response.ok ? response.json() : {}))
      .catch(() => ({}));
  }
  return logosPromise;
}

function monogram(team) {
  const words = team.split(/\s+/).filter(Boolean);
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return words
    .slice(0, 2)
    .map((word) => word[0])
    .join("")
    .toUpperCase();
}

export default function TeamLogo({ team, size = 20 }) {
  const [src, setSrc] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    setFailed(false);
    loadLogos().then((mapping) => {
      if (alive) setSrc(mapping[team] ?? null);
    });
    return () => {
      alive = false;
    };
  }, [team]);

  if (!src || failed) {
    return (
      <span
        className="monogram"
        title={team}
        style={{
          width: size,
          height: size,
          fontSize: Math.max(8, size * 0.42),
          background: teamColor(team),
        }}
      >
        {monogram(team)}
      </span>
    );
  }

  return (
    <span className="logo-chip" title={team} style={{ width: size + 6, height: size + 6 }}>
      <img
        className="team-logo"
        src={src}
        alt={team}
        width={size}
        height={size}
        loading="lazy"
        onError={() => setFailed(true)}
      />
    </span>
  );
}
