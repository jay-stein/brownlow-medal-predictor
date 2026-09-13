import { Fragment, useState } from "react";
import TeamLogo from "./TeamLogo.jsx";
import WormChart from "./WormChart.jsx";

const TEAM_ROWS = [
  { key: "fiveSix", label: "5-6 votes", color: "246, 226, 122", bracket: (probs) => (probs?.[5] ?? 0) + (probs?.[6] ?? 0) },
  { key: "threeFour", label: "3-4 votes", color: "227, 184, 74", bracket: (probs) => (probs?.[3] ?? 0) + (probs?.[4] ?? 0) },
  { key: "oneTwo", label: "1-2 votes", color: "176, 128, 40", bracket: (probs) => (probs?.[1] ?? 0) + (probs?.[2] ?? 0) },
  { key: "zero", label: "0 votes", color: "120, 130, 160", bracket: (probs) => probs?.[0] ?? 0 },
];

function percent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function TeamRoundVotes({ team, rounds, matches, teamName }) {
  const [activeRound, setActiveRound] = useState(null);
  const probs = team.rounds?.incProbs ?? [];

  const hotRounds = rounds
    .map((round, index) => ({ ...round, index, probability: TEAM_ROWS[0].bracket(probs[index]) }))
    .sort((a, b) => b.probability - a.probability)
    .filter((entry) => entry.probability >= 0.05)
    .slice(0, 3);
  const hotIndexes = new Set(hotRounds.map((entry) => entry.index));

  const opponentFor = (roundNumber) => {
    const match = (matches ?? []).find(
      (candidate) =>
        candidate.round === roundNumber && (candidate.home === teamName || candidate.away === teamName)
    );
    if (!match) return null;
    return match.home === teamName ? `vs ${match.away}` : `at ${match.home}`;
  };

  const detail = activeRound === null ? null : {
    ...rounds[activeRound],
    opponent: opponentFor(rounds[activeRound].number),
    expected: team.rounds?.incMean?.[activeRound] ?? 0,
  };

  return (
    <div className="round-votes">
      <div className="hot-rounds">
        <span className="hot-label">Likeliest big-vote rounds</span>
        {hotRounds.length === 0 ? (
          <span className="hot-empty">No round is a strong 5-6 vote chance</span>
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
          {TEAM_ROWS.map((row) => (
            <Fragment key={row.key}>
              <div className="heatmap-row-label">{row.label}</div>
              {rounds.map((round, index) => {
                const probability = row.bracket(probs[index]);
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
                    onMouseLeave={() => setActiveRound((value) => (value === index ? null : value))}
                    onFocus={() => setActiveRound(index)}
                    onClick={() => setActiveRound(index)}
                    title={`${round.label}: ${row.label} ${percent(probability)}`}
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
            <span className="round-detail-title">
              {detail.label}
              {detail.opponent ? ` · ${detail.opponent}` : ""}
            </span>
            <span>
              5-6 votes <b>{percent(TEAM_ROWS[0].bracket(probs[activeRound]))}</b>
            </span>
            <span>
              3-4 votes <b>{percent(TEAM_ROWS[1].bracket(probs[activeRound]))}</b>
            </span>
            <span>
              1-2 votes <b>{percent(TEAM_ROWS[2].bracket(probs[activeRound]))}</b>
            </span>
            <span>
              0 votes <b>{percent(TEAM_ROWS[3].bracket(probs[activeRound]))}</b>
            </span>
            <span>
              Expected <b>{Number(detail.expected).toFixed(2)}</b> votes
            </span>
          </>
        ) : (
          <span className="round-detail-hint">
            Hover a cell to see the vote distribution for that round
          </span>
        )}
      </div>
    </div>
  );
}

export default function TeamsView({ teams, rounds, matches }) {
  const [expanded, setExpanded] = useState(null);
  const [mode, setMode] = useState("cumulative");
  if (!teams?.length) return null;
  const max = Math.max(...teams.map((team) => team.q95 ?? team.expected ?? 0), 1);

  return (
    <section className="panel teams-panel">
      <div className="panel-head">
        <h2>Team totals</h2>
        <span className="hint">
          expected Brownlow votes across every listed player · click a club to see its season worm
        </span>
      </div>
      <div className="team-grid">
        {teams.map((team) => {
          const isOpen = expanded === team.name;
          const chartData =
            isOpen && team.rounds?.cumMean
              ? (rounds ?? []).map((round, index) => {
                  const source = team.rounds;
                  const row = {
                    round: round.label,
                    cumMean: source.cumMean[index],
                    cumQ05: source.cumQ05[index],
                    cumQ25: source.cumQ25[index],
                    cumMedian: source.cumMedian[index],
                    cumQ75: source.cumQ75[index],
                    cumQ95: source.cumQ95[index],
                    cumOuter: source.cumQ95[index] - source.cumQ05[index],
                    cumInner: source.cumQ75[index] - source.cumQ25[index],
                    incMean: source.incMean[index],
                    incQ05: source.incQ05[index],
                    incQ25: source.incQ25[index],
                    incMedian: source.incMedian[index],
                    incQ75: source.incQ75[index],
                    incQ95: source.incQ95[index],
                    incOuter: source.incQ95[index] - source.incQ05[index],
                    incInner: source.incQ75[index] - source.incQ25[index],
                  };
                  (source.paths ?? []).forEach((path, pathIndex) => {
                    row[`path_${pathIndex}`] = path[index];
                  });
                  return row;
                })
              : null;
          return (
            <article className={isOpen ? "team-card expanded" : "team-card"} key={team.name}>
              <header
                className="team-card-head"
                onClick={() => {
                  setExpanded(isOpen ? null : team.name);
                  setMode("cumulative");
                }}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    setExpanded(isOpen ? null : team.name);
                    setMode("cumulative");
                  }
                }}
              >
                <TeamLogo team={team.name} size={24} />
                <h3>{team.name}</h3>
                <strong>{Number(team.expected ?? 0).toFixed(1)}</strong>
                <span className="team-more">{isOpen ? "Less" : "More"}</span>
              </header>
              <div className="team-band" aria-hidden="true">
                <span
                  className="team-range"
                  style={{
                    left: `${(team.q05 / max) * 100}%`,
                    width: `${((team.q95 - team.q05) / max) * 100}%`,
                  }}
                />
                <span className="team-value" style={{ width: `${(team.expected / max) * 100}%` }} />
              </div>
              <p className="team-range-text">
                90% range {Number(team.q05 ?? 0).toFixed(0)}–{Number(team.q95 ?? 0).toFixed(0)} ·{" "}
                {team.nPlayers} players
              </p>
              <ul className="team-top">
                {team.players.map((player) => (
                  <li key={player.id}>
                    <span>{player.name}</span>
                    <b>{Number(player.expected ?? 0).toFixed(1)}</b>
                  </li>
                ))}
              </ul>
              {isOpen && chartData ? (
                <div className="team-detail">
                  <div className="toggles">
                    <button
                      type="button"
                      className={mode === "cumulative" ? "active" : ""}
                      onClick={() => setMode("cumulative")}
                    >
                      Cumulative worm
                    </button>
                    <button
                      type="button"
                      className={mode === "round" ? "active" : ""}
                      onClick={() => setMode("round")}
                    >
                      Round votes
                    </button>
                  </div>
                  {mode === "cumulative" ? (
                    <WormChart
                      data={chartData}
                      player={{ rounds: { paths: team.rounds?.paths ?? [] } }}
                      showPaths={false}
                    />
                  ) : (
                    <TeamRoundVotes
                      team={team}
                      rounds={rounds ?? []}
                      matches={matches ?? []}
                      teamName={team.name}
                    />
                  )}
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
