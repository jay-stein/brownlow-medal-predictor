import { useEffect, useMemo, useState } from "react";
import RoundVotes from "./RoundVotes.jsx";
import WormChart from "./WormChart.jsx";
import { teamColor } from "./teams.js";

function pct(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [mode, setMode] = useState("cumulative");
  const [showPaths, setShowPaths] = useState(true);
  const [query, setQuery] = useState("");

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}forecast_2026.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  const players = data?.players ?? [];

  const filtered = useMemo(() => {
    if (!query) return players;
    const needle = query.toLowerCase();
    return players.filter(
      (entry) =>
        entry.name.toLowerCase().includes(needle) || entry.team.toLowerCase().includes(needle)
    );
  }, [players, query]);

  const player = players.find((entry) => entry.id === selectedId) ?? players[0] ?? null;

  const chartData = useMemo(() => {
    if (!data || !player) return [];
    const rounds = player.rounds;
    return data.rounds.map((round, index) => {
      const row = {
        round: round.label,
        cumMean: rounds.cumMean[index],
        cumQ05: rounds.cumQ05[index],
        cumQ25: rounds.cumQ25[index],
        cumMedian: rounds.cumMedian[index],
        cumQ75: rounds.cumQ75[index],
        cumQ95: rounds.cumQ95[index],
        cumOuter: rounds.cumQ95[index] - rounds.cumQ05[index],
        cumInner: rounds.cumQ75[index] - rounds.cumQ25[index],
        incMean: rounds.incMean[index],
        incQ05: rounds.incQ05[index],
        incQ25: rounds.incQ25[index],
        incMedian: rounds.incMedian[index],
        incQ75: rounds.incQ75[index],
        incQ95: rounds.incQ95[index],
        incOuter: rounds.incQ95[index] - rounds.incQ05[index],
        incInner: rounds.incQ75[index] - rounds.incQ25[index],
      };
      rounds.paths.forEach((path, pathIndex) => {
        row[`path_${pathIndex}`] = path[index];
      });
      return row;
    });
  }, [data, player]);

  if (error) return <div className="loading">Could not load forecast data: {error}</div>;
  if (!data || !player) return <div className="loading">Loading forecast…</div>;

  const meta = data.meta ?? {};

  return (
    <div className="app">
      <header className="hero">
        <div className="medal" aria-hidden="true">
          <span>26</span>
        </div>
        <p className="eyebrow">AFL · {data.season} season · post-round-24 forecast</p>
        <h1>
          Brownlow Medal <em>Forecast</em>
        </h1>
        <p className="subtitle">
          {Number(meta.nSims ?? 10000).toLocaleString()} simulated counts · Plackett–Luce allocation
          with calibrated uncertainty
        </p>
        <div className="chips">
          <span>Model: {meta.model ?? "ranking"}</span>
          <span>τ = {meta.tau ?? "—"}</span>
          <span>Effect scale = {meta.effectScale ?? "—"}</span>
          <span>Generated {meta.generated ?? "—"}</span>
        </div>
      </header>

      <main className="layout">
        <section className="panel leaderboard-panel">
          <div className="panel-head">
            <h2>Contenders</h2>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search player or team"
            />
          </div>
          <ol className="leaderboard">
            {filtered.map((entry) => {
              const rank = players.indexOf(entry) + 1;
              const active = entry.id === player.id;
              return (
                <li
                  key={entry.id}
                  className={active ? "active" : ""}
                  onClick={() => setSelectedId(entry.id)}
                >
                  <span className="rank">{rank}</span>
                  <span className="dot" style={{ background: teamColor(entry.team) }} />
                  <span className="who">
                    <span className="name">{entry.name}</span>
                    <span className="team">{entry.team}</span>
                  </span>
                  <span className="votes">{entry.expectedVotes?.toFixed(1)}</span>
                  <span className="prob">
                    <span className="prob-bar">
                      <span
                        style={{ width: `${Math.min(100, (entry.pFirstOrJoint ?? 0) * 100)}%` }}
                      />
                    </span>
                    <span className="prob-value">{pct(entry.pFirstOrJoint)}</span>
                  </span>
                </li>
              );
            })}
          </ol>
        </section>

        <section className="panel chart-panel">
          <div className="chart-head">
            <div>
              <h2>{player.name}</h2>
              <p className="team-line">
                <span className="dot" style={{ background: teamColor(player.team) }} />{" "}
                {player.team}
              </p>
            </div>
            <div className="toggles">
              <button
                className={mode === "cumulative" ? "active" : ""}
                onClick={() => setMode("cumulative")}
              >
                Cumulative worm
              </button>
              <button
                className={mode === "round" ? "active" : ""}
                onClick={() => setMode("round")}
              >
                Round votes
              </button>
              {mode === "cumulative" ? (
                <button
                  className={showPaths ? "active" : ""}
                  onClick={() => setShowPaths((value) => !value)}
                >
                  Simulations
                </button>
              ) : null}
            </div>
          </div>

          <div className="stat-grid">
            <div className="stat">
              <span>Expected votes</span>
              <strong>{player.expectedVotes?.toFixed(1)}</strong>
              <small>median {player.medianVotes}</small>
            </div>
            <div className="stat">
              <span>90% range</span>
              <strong>
                {player.q05}–{player.q95}
              </strong>
              <small>
                central 50%: {player.q25}–{player.q75}
              </small>
            </div>
            <div className="stat">
              <span>P(first or joint)</span>
              <strong>{pct(player.pFirstOrJoint)}</strong>
              <small>
                {player.winLow != null
                  ? `${pct(player.winLow)}–${pct(player.winHigh)} across specs`
                  : "single specification"}
              </small>
            </div>
            <div className="stat">
              <span>P(top 5)</span>
              <strong>{pct(player.pTop5)}</strong>
              <small>outright {pct(player.pOutright)}</small>
            </div>
          </div>

          <div className="chart-wrap">
            {mode === "cumulative" ? (
              <WormChart data={chartData} player={player} showPaths={showPaths} />
            ) : (
              <RoundVotes player={player} rounds={data.rounds} />
            )}
          </div>

          <div className="legend">
            {mode === "cumulative" ? (
              <>
                <span>
                  <i className="swatch outer" /> 90% band
                </span>
                <span>
                  <i className="swatch inner" /> 50% band
                </span>
                <span>
                  <i className="swatch median" /> median
                </span>
                <span>
                  <i className="swatch mean" /> mean
                </span>
                {showPaths ? (
                  <span>
                    <i className="swatch paths" /> sampled simulations
                  </span>
                ) : null}
              </>
            ) : (
              <>
                <span>
                  <i className="swatch votes3" /> 3 votes
                </span>
                <span>
                  <i className="swatch votes2" /> 2 votes
                </span>
                <span>
                  <i className="swatch votes1" /> 1 vote
                </span>
                <span className="legend-note">brighter = more likely</span>
              </>
            )}
          </div>
        </section>
      </main>

      <footer className="footnote">
        <p>
          Method: match-grouped LambdaMART scores trained on 2012–2025 votes; Plackett–Luce
          allocation with temperature τ fitted on {meta.tauMatches ?? "prior"} out-of-sample matches;
          one persistent player-season effect at the calibrated scale. Backtest (2015–2025): 88.5%
          contender coverage at the nominal 90% level, and the eventual winner averaged a 36%
          pre-count probability.
        </p>
        <p>
          Post-round-24 conditional forecast: match statistics are observed, only the hidden votes
          are simulated. Brownlow eligibility (suspensions) is not applied. Vote totals, bands and
          probabilities are model estimates, not certainties.
        </p>
      </footer>
    </div>
  );
}
