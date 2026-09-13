import { useMemo, useState } from "react";
import { MatchCard } from "./MatchesView.jsx";
import TeamLogo from "./TeamLogo.jsx";

export default function TeamMatchesView({ matches, teams }) {
  const teamNames = useMemo(
    () => (teams ?? []).map((team) => team.name).sort((a, b) => a.localeCompare(b)),
    [teams]
  );
  const [selected, setSelected] = useState(null);
  const activeTeam = selected ?? teamNames[0] ?? null;

  const visible = useMemo(() => {
    if (!activeTeam) return [];
    return (matches ?? [])
      .filter((match) => match.home === activeTeam || match.away === activeTeam)
      .sort(
        (a, b) =>
          (a.round ?? 0) - (b.round ?? 0) ||
          String(a.date ?? "").localeCompare(String(b.date ?? ""))
      );
  }, [matches, activeTeam]);

  if (!matches?.length || !teamNames.length) return null;

  return (
    <section className="panel matches-panel">
      <div className="panel-head">
        <h2>Matches by team</h2>
        <span className="hint">
          every game for one club, round by round · highlighted rows are that club's players
        </span>
      </div>
      <div className="round-switcher team-chips">
        {teamNames.map((name) => (
          <button
            type="button"
            key={name}
            className={name === activeTeam ? "active" : ""}
            onClick={() => setSelected(name)}
          >
            <TeamLogo team={name} size={16} />
            {name}
          </button>
        ))}
      </div>
      <p className="stat-legend">
        <b>{activeTeam}</b>: {visible.length} games, chronological. Each tile shows both teams —
        {` ${activeTeam} players are grouped first and highlighted.`}
      </p>
      <div className="match-grid">
        {visible.map((match) => (
          <MatchCard key={match.id} match={match} focusTeam={activeTeam} />
        ))}
      </div>
    </section>
  );
}
