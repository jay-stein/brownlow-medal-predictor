import { useState } from "react";
import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import TeamLogo from "./TeamLogo.jsx";
import WormChart from "./WormChart.jsx";

function TeamRoundTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div className="tooltip-title">{label}</div>
      <div className="tooltip-row">
        <span>Expected votes</span>
        <b>{row.incMean}</b>
      </div>
      <div className="tooltip-row">
        <span>50% range</span>
        <b>
          {row.incQ25} – {row.incQ75}
        </b>
      </div>
      <div className="tooltip-row">
        <span>90% range</span>
        <b>
          {row.incQ05} – {row.incQ95}
        </b>
      </div>
    </div>
  );
}

function TeamRoundChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={data} margin={{ top: 16, right: 24, bottom: 8, left: 0 }}>
        <defs>
          <linearGradient id="teamRoundOuter" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f6e27a" stopOpacity={0.2} />
            <stop offset="100%" stopColor="#d4af37" stopOpacity={0.04} />
          </linearGradient>
          <linearGradient id="teamRoundInner" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f6e27a" stopOpacity={0.4} />
            <stop offset="100%" stopColor="#c39a1f" stopOpacity={0.14} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(212,175,55,0.12)" vertical={false} />
        <XAxis
          dataKey="round"
          interval={2}
          tick={{ fill: "#9aa4bf", fontSize: 12 }}
          tickLine={false}
          axisLine={{ stroke: "rgba(212,175,55,0.25)" }}
        />
        <YAxis tick={{ fill: "#9aa4bf", fontSize: 12 }} tickLine={false} axisLine={false} width={44} />
        <Tooltip
          content={<TeamRoundTooltip />}
          cursor={{ fill: "rgba(246,226,122,0.06)" }}
        />
        <Area type="monotone" dataKey="incQ05" stackId="outer" stroke="none" fill="transparent" isAnimationActive={false} />
        <Area type="monotone" dataKey="incOuter" stackId="outer" stroke="none" fill="url(#teamRoundOuter)" isAnimationActive={false} />
        <Area type="monotone" dataKey="incQ25" stackId="inner" stroke="none" fill="transparent" isAnimationActive={false} />
        <Area type="monotone" dataKey="incInner" stackId="inner" stroke="none" fill="url(#teamRoundInner)" isAnimationActive={false} />
        <Bar
          dataKey="incMean"
          fill="#f6e27a"
          fillOpacity={0.8}
          radius={[4, 4, 0, 0]}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function teamChartData(team, rounds) {
  const source = team.rounds;
  if (!source?.cumMean) return null;
  return rounds.map((round, index) => {
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
  });
}

export default function TeamsView({ teams, rounds }) {
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
          const chartData = isOpen ? teamChartData(team, rounds ?? []) : null;
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
                <span className="team-chevron">{isOpen ? "−" : "+"}</span>
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
                    <TeamRoundChart data={chartData} />
                  )}
                  <p className="team-range-text">
                    {mode === "cumulative"
                      ? "Cumulative expected team votes by round, with 50% and 90% bands from the 10,000 simulated counts."
                      : "Expected team votes in each round, with 50% and 90% bands. A team plays once per round, so each bar is that match's share of the six votes."}
                  </p>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
