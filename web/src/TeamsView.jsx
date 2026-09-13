import TeamLogo from "./TeamLogo.jsx";

export default function TeamsView({ teams }) {
  if (!teams?.length) return null;
  const max = Math.max(...teams.map((team) => team.q95 ?? team.expected ?? 0), 1);

  return (
    <section className="panel teams-panel">
      <div className="panel-head">
        <h2>Team totals</h2>
        <span className="hint">expected Brownlow votes across every listed player, with the 90% range</span>
      </div>
      <div className="team-grid">
        {teams.map((team) => (
          <article className="team-card" key={team.name}>
            <header>
              <TeamLogo team={team.name} size={24} />
              <h3>{team.name}</h3>
              <strong>{Number(team.expected ?? 0).toFixed(1)}</strong>
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
          </article>
        ))}
      </div>
    </section>
  );
}
