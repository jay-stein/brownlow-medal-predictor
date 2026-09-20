import { useEffect, useMemo, useState } from "react";
import CountRace from "./CountRace.jsx";
import TeamLogo from "./TeamLogo.jsx";
import {
  findRoundMatch,
  playerProjection,
  remainingEstimates,
  sigmaFromHalfWidth,
  winOdds,
} from "./countNight.js";

const STORAGE_KEY = "brownlow.count-night.v1";
const MAX_PLAYERS = 5;

function pct(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function signed(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  const rounded = value.toFixed(digits);
  return value > 0 ? `+${rounded}` : rounded;
}

function loadStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function Autocomplete({ players, excludeIds, onSelect }) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);

  const options = useMemo(() => {
    const available = players.filter((player) => !excludeIds.has(player.id));
    const needle = text.trim().toLowerCase();
    if (!needle) return available.slice(0, 8);
    return available
      .filter((player) => {
        const name = player.name.toLowerCase();
        return name.includes(needle) || name.split(/\s+/).some((word) => word.startsWith(needle));
      })
      .slice(0, 8);
  }, [players, excludeIds, text]);

  useEffect(() => {
    setActive(0);
  }, [text]);

  const choose = (option) => {
    if (!option) return;
    onSelect(option.id);
    setText("");
    setOpen(false);
  };

  return (
    <div className="autocomplete">
      <input
        value={text}
        placeholder="Add a player — start typing a name"
        onChange={(event) => {
          setText(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 120)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setActive((index) => Math.min(index + 1, options.length - 1));
            setOpen(true);
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setActive((index) => Math.max(index - 1, 0));
          } else if (event.key === "Enter") {
            event.preventDefault();
            choose(options[active]);
          } else if (event.key === "Escape") {
            setOpen(false);
          }
        }}
      />
      {open && options.length ? (
        <div className="autocomplete-list">
          {options.map((option, index) => (
            <button
              type="button"
              key={option.id}
              className={index === active ? "autocomplete-option active" : "autocomplete-option"}
              onMouseDown={(event) => {
                event.preventDefault();
                choose(option);
              }}
              onMouseEnter={() => setActive(index)}
            >
              <TeamLogo team={option.team} size={16} />
              <span>{option.name}</span>
              <em>{option.team}</em>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function MedalIcon() {
  return (
    <svg className="count-medal" viewBox="0 0 40 44" aria-hidden="true">
      <defs>
        <linearGradient id="count-medal-gold" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#fff3c4" />
          <stop offset="45%" stopColor="#d4af37" />
          <stop offset="100%" stopColor="#9a7b1c" />
        </linearGradient>
      </defs>
      <polygon points="12,0 20,0 22,16 14,16" fill="#8a1f2d" />
      <polygon points="20,0 28,0 26,16 22,16" fill="#1f3a8a" />
      <circle cx="20" cy="28" r="13" fill="url(#count-medal-gold)" stroke="#7a611a" strokeWidth="1" />
      <circle cx="20" cy="28" r="9" fill="none" stroke="rgba(60, 44, 0, 0.35)" />
      <text x="20" y="32" textAnchor="middle" fontSize="9.5" fontWeight="700" fill="#3a2c00">
        321
      </text>
    </svg>
  );
}

export default function CountNight({ players, rounds, matches, meta, onOpenMatch }) {
  const calibration = meta?.countNight ?? null;
  const stored = useMemo(loadStored, []);
  const [roundIndex, setRoundIndex] = useState(() => stored?.roundIndex ?? 0);
  const [entries, setEntries] = useState(() =>
    Array.isArray(stored?.entries) ? stored.entries : []
  );

  const roundList = rounds ?? [];
  const roundCount = roundList.length;

  useEffect(() => {
    if (!players?.length) return;
    setEntries((current) => current.filter((entry) => players.some((p) => p.id === entry.id)));
  }, [players]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ roundIndex, entries }));
    } catch {
      // storage unavailable; the tab still works for the session
    }
  }, [roundIndex, entries]);

  const rows = entries
    .map((entry) => {
      const player = players?.find((candidate) => candidate.id === entry.id);
      if (!player) return null;
      const projection = playerProjection(
        player,
        roundIndex,
        entry.votes ?? 0,
        calibration,
        roundCount
      );
      if (!projection) return null;
      return {
        entry,
        player,
        projection,
        sigma: sigmaFromHalfWidth(projection.half90),
        remaining: remainingEstimates(player, roundIndex),
      };
    })
    .filter(Boolean);

  const odds = winOdds(
    rows.map((row) => ({
      id: row.player.id,
      projected: row.projection.projected,
      sigma: row.sigma,
      eligible: !row.player.ineligible,
    }))
  );
  const projectedWinner =
    rows.length > 1
      ? rows.reduce(
          (best, row) => (odds[row.player.id] > (odds[best.player.id] ?? 0) ? row : best),
          rows[0]
        )
      : null;

  const scale = useMemo(() => {
    if (!rows.length) return { min: 0, max: 1 };
    const lows = rows.map((row) =>
      Math.min(row.projection.projected - row.projection.half90, row.entry.votes ?? 0)
    );
    const highs = rows.map((row) =>
      Math.max(row.projection.projected + row.projection.half90, row.entry.votes ?? 0)
    );
    const min = Math.min(...lows);
    const max = Math.max(...highs, min + 1);
    return { min, max };
  }, [rows]);

  const position = (value) => ((value - scale.min) / (scale.max - scale.min)) * 100;

  const addPlayer = (id) => {
    setEntries((current) =>
      current.length >= MAX_PLAYERS || current.some((entry) => entry.id === id)
        ? current
        : [...current, { id, votes: 0 }]
    );
  };
  const removePlayer = (id) => setEntries((current) => current.filter((entry) => entry.id !== id));
  const setVotes = (id, votes) =>
    setEntries((current) => current.map((entry) => (entry.id === id ? { ...entry, votes } : entry)));

  return (
    <section className="panel count-panel">
      <div className="panel-head count-head">
        <div className="count-title">
          <MedalIcon />
          <div>
            <h2>Count Night</h2>
            <span className="count-subtitle">live from the count</span>
          </div>
        </div>
        <span className="hint">enter the votes as they are read out · live projections</span>
      </div>
      <p className="stat-legend">
        Pick the round you are up to, add up to five players and type their votes so far. The
        projection is their total so far plus the model's expected votes for the rounds still to
        come; the range is calibrated on 2013–2025 count trajectories, and the win odds account for
        suspensions (an ineligible player cannot win).
      </p>
      {!calibration ? (
        <p className="count-warning">
          Calibration table missing from the payload — regenerate the forecast to enable ranges
          and odds.
        </p>
      ) : null}

      <div className="count-controls">
        <span className="count-control-label">Round</span>
        <div className="round-stepper">
          <button
            type="button"
            onClick={() => setRoundIndex((index) => Math.max(0, index - 1))}
            disabled={roundIndex <= 0}
            aria-label="Previous round"
          >
            −
          </button>
          <select
            className="round-select"
            value={roundIndex}
            onChange={(event) => setRoundIndex(Number(event.target.value))}
          >
            {roundList.map((round, index) => (
              <option key={round.number} value={index}>
                {round.label}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => setRoundIndex((index) => Math.min(roundCount - 1, index + 1))}
            disabled={roundIndex >= roundCount - 1}
            aria-label="Next round"
          >
            +
          </button>
        </div>
        <span className="count-rounds-left">
          {roundIndex < roundCount ? `${roundCount - 1 - roundIndex} rounds left` : ""}
        </span>
        {entries.length ? (
          <button
            type="button"
            className="count-reset"
            onClick={() => {
              setEntries([]);
              setRoundIndex(0);
            }}
          >
            Reset
          </button>
        ) : null}
      </div>

      {entries.length < MAX_PLAYERS ? (
        <Autocomplete
          players={players ?? []}
          excludeIds={new Set(entries.map((entry) => entry.id))}
          onSelect={addPlayer}
        />
      ) : null}

      {entries.length === 0 ? (
        <p className="count-empty">
          Add your first player above — try typing a surname. Projections and live odds appear as
          you go.
        </p>
      ) : null}

      {projectedWinner ? (
        <div className="count-summary">
          <span>Projected winner among these</span>
          <b>
            <TeamLogo team={projectedWinner.player.team} size={16} /> {projectedWinner.player.name}
          </b>
          <em>{pct(odds[projectedWinner.player.id])}</em>
        </div>
      ) : null}

      {rows.length ? (
        <>
          <p className="count-race-caption">
            Dots are the votes entered; dashed lines project each player forward, and the shaded
            cone widens with every unknown round — next round's total is more certain than the
            finish. Faint lines are the model's original path.
          </p>
          <CountRace
            rows={rows}
            roundIndex={roundIndex}
            roundList={roundList}
            calibration={calibration}
          />
        </>
      ) : null}

      <div className="count-entries">
        {rows.map(({ entry, player, projection, remaining }) => {
          const low = projection.projected - projection.half90;
          const high = projection.projected + projection.half90;
          const observed = entry.votes ?? 0;
          const pace = observed - projection.priorThrough;
          const max = 3 * (roundIndex + 1);
          return (
            <article className="count-card" key={player.id}>
              <header>
                <TeamLogo team={player.team} size={20} />
                <b>{player.name}</b>
                <span className="count-team">{player.team}</span>
                {player.ineligible ? <i className="ineligible-badge">ineligible</i> : null}
                <button type="button" className="count-remove" onClick={() => removePlayer(player.id)}>
                  ×
                </button>
              </header>
              <div className="count-projection">
                <span className="count-number">{projection.projected.toFixed(1)}</span>
                <span className="count-range">
                  {low.toFixed(1)} – {high.toFixed(1)}
                </span>
                {rows.length > 1 ? <span className="count-odds">{pct(odds[player.id])} to win</span> : null}
              </div>
              <div className="projection-bar">
                <span
                  className="bar-fill"
                  style={{ left: `${position(low)}%`, width: `${Math.max(0, position(high) - position(low))}%` }}
                />
                <span
                  className="bar-expected"
                  style={{ left: `${position(projection.priorThrough)}%` }}
                  title={`model expected ${projection.priorThrough.toFixed(1)} by now`}
                />
                <span
                  className="bar-observed"
                  style={{ left: `${position(observed)}%` }}
                  title={`votes entered: ${observed}`}
                />
                <span
                  className="bar-projected"
                  style={{ left: `${position(projection.projected)}%` }}
                  title={`projected: ${projection.projected.toFixed(1)}`}
                />
              </div>
              <div className="count-meta">
                <label>
                  Votes so far
                  <input
                    type="number"
                    min={0}
                    max={max}
                    value={observed}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      const clamped = Number.isFinite(value)
                        ? Math.min(Math.max(Math.trunc(value), 0), max)
                        : 0;
                      setVotes(player.id, clamped);
                    }}
                  />
                </label>
                <span
                  className={pace >= 0 ? "pace-chip ahead" : "pace-chip behind"}
                  title={`model expected ${projection.priorThrough.toFixed(1)} by ${
                    roundList[roundIndex]?.label ?? "now"
                  }`}
                >
                  {signed(pace)} vs model
                </span>
                <span className="count-prior" title="model's pre-season projection">
                  pre-season {projection.priorFinal.toFixed(1)}
                </span>
              </div>
              {remaining.some((value) => value > 0.05) ? (
                <>
                  <span className="count-rounds-label">Expected votes, round by round</span>
                  <div className="count-rounds">
                    {remaining.map((value, offset) => {
                      if (value <= 0.05) return null;
                      const alpha = Math.min(0.4, 0.05 + value * 0.13);
                      const hot = value >= 2.5;
                      const roundEntry = roundList[roundIndex + 1 + offset];
                      const label = roundEntry?.label ?? `+${offset + 1}`;
                      const match = findRoundMatch(matches, player.team, roundEntry?.number);
                      const title = match
                        ? `${match.home} v ${match.away} — model expects ${value.toFixed(
                            1
                          )} votes · tap to open ${player.team}'s matches`
                        : `${label}: model expects ${value.toFixed(1)} votes`;
                      const style = {
                        background: `rgba(212, 175, 55, ${alpha.toFixed(3)})`,
                        borderColor: hot
                          ? "rgba(212, 175, 55, 0.65)"
                          : "rgba(255, 255, 255, 0.1)",
                      };
                      const className = ["round-chip", hot ? "hot" : "", match && onOpenMatch ? "clickable" : ""]
                        .filter(Boolean)
                        .join(" ");
                      if (match && onOpenMatch) {
                        return (
                          <button
                            type="button"
                            key={offset}
                            className={className}
                            style={style}
                            title={title}
                            onClick={() => onOpenMatch(match, player.team)}
                          >
                            <em>{label}</em>
                            {value.toFixed(1)}
                          </button>
                        );
                      }
                      return (
                        <span key={offset} className={className} style={style} title={title}>
                          <em>{label}</em>
                          {value.toFixed(1)}
                        </span>
                      );
                    })}
                  </div>
                </>
              ) : (
                <p className="count-meta">No rounds left to project.</p>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
