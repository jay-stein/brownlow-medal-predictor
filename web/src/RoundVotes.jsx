import { Fragment, useMemo, useState } from "react";

const ROWS = [
  { key: "p3", label: "3 votes", color: "246, 226, 122" },
  { key: "p2", label: "2 votes", color: "227, 184, 74" },
  { key: "p1", label: "1 vote", color: "176, 128, 40" },
];

function percent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

export default function RoundVotes({ player, rounds }) {
  const [activeRound, setActiveRound] = useState(null);

  const hotRounds = useMemo(() => {
    const probabilities = player.rounds.p3 ?? [];
    return rounds
      .map((round, index) => ({ ...round, index, probability: probabilities[index] ?? 0 }))
      .sort((a, b) => b.probability - a.probability)
      .filter((entry) => entry.probability >= 0.05)
      .slice(0, 3);
  }, [player, rounds]);

  const hotIndexes = new Set(hotRounds.map((entry) => entry.index));
  const detail =
    activeRound === null
      ? null
      : {
          ...rounds[activeRound],
          p1: player.rounds.p1?.[activeRound] ?? 0,
          p2: player.rounds.p2?.[activeRound] ?? 0,
          p3: player.rounds.p3?.[activeRound] ?? 0,
          mean: player.rounds.incMean?.[activeRound] ?? 0,
        };

  return (
    <div className="round-votes">
      <div className="hot-rounds">
        <span className="hot-label">Likeliest 3-vote rounds</span>
        {hotRounds.length === 0 ? (
          <span className="hot-empty">No round is a strong 3-vote chance</span>
        ) : (
          hotRounds.map((entry) => (
            <span className="hot-chip" key={entry.number}>
              {entry.label} <b>{percent(entry.probability)}</b>
            </span>
          ))
        )}
      </div>

      <div className="heatmap-scroll">
        <div
          className="heatmap"
          style={{ gridTemplateColumns: `96px repeat(${rounds.length}, minmax(34px, 1fr))` }}
        >
          <div className="heatmap-corner" />
          {rounds.map((round, index) => (
            <div
              key={`head-${round.number}`}
              className={`heatmap-col-label ${hotIndexes.has(index) ? "hot" : ""}`}
            >
              {round.label}
            </div>
          ))}
          {ROWS.map((row) => (
            <Fragment key={row.key}>
              <div className="heatmap-row-label">{row.label}</div>
              {rounds.map((round, index) => {
                const probability = player.rounds[row.key]?.[index] ?? 0;
                const alpha = probability <= 0 ? 0.04 : 0.1 + 0.9 * probability;
                const strong = probability >= 0.45;
                return (
                  <button
                    key={`${row.key}-${round.number}`}
                    type="button"
                    className={`heatmap-cell ${strong ? "strong" : ""} ${
                      activeRound === index ? "active" : ""
                    }`}
                    style={{
                      background: `rgba(${row.color}, ${alpha})`,
                      color: strong ? "#241a00" : "#dfe5f3",
                    }}
                    onMouseEnter={() => setActiveRound(index)}
                    onMouseLeave={() =>
                      setActiveRound((value) => (value === index ? null : value))
                    }
                    onFocus={() => setActiveRound(index)}
                    onClick={() => setActiveRound(index)}
                    title={`${round.label}: P(3) ${percent(player.rounds.p3?.[index])}, P(2) ${percent(
                      player.rounds.p2?.[index]
                    )}, P(1) ${percent(player.rounds.p1?.[index])}`}
                  >
                    {probability >= 0.08 ? percent(probability) : ""}
                  </button>
                );
              })}
            </Fragment>
          ))}
        </div>
      </div>

      <div className="round-detail">
        {detail ? (
          <>
            <span className="round-detail-title">{detail.label}</span>
            <span>
              3 votes <b>{percent(detail.p3)}</b>
            </span>
            <span>
              2 votes <b>{percent(detail.p2)}</b>
            </span>
            <span>
              1 vote <b>{percent(detail.p1)}</b>
            </span>
            <span>
              Expected <b>{detail.mean.toFixed(2)}</b> pts
            </span>
          </>
        ) : (
          <span className="round-detail-hint">
            Hover a cell to see exact 3-2-1 probabilities for that round
          </span>
        )}
      </div>
    </div>
  );
}
