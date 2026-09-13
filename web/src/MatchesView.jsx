import { useMemo, useState } from "react";
import TeamLogo from "./TeamLogo.jsx";
import { teamColor } from "./teams.js";

function pct(value, digits = 0) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function ProbCell({ value, boost = 1 }) {
  const probability = value ?? 0;
  const alpha = Math.min(0.3, 0.04 + probability * 0.22 * boost);
  return (
    <span
      className="prob-cell"
      style={{ background: `rgba(212, 175, 55, ${alpha.toFixed(3)})` }}
      title={`${pct(probability, 1)} probability`}
    >
      <i style={{ width: `${Math.min(100, probability * 100)}%` }} />
      <em>{pct(probability)}</em>
    </span>
  );
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-AU", { weekday: "short", day: "numeric", month: "short" });
}

function summarise(match) {
  if (match.homeScore == null || match.awayScore == null) return null;
  const margin = Math.abs(match.homeScore - match.awayScore);
  const winner = match.homeScore > match.awayScore ? match.home : match.away;
  if (match.homeScore === match.awayScore) return `${match.home} and ${match.away} drew.`;
  const top = match.votes?.[0];
  if (!top) return `${winner} won by ${margin}.`;
  const details = [];
  if (top.disposals != null) details.push(`${top.disposals} disposals`);
  if (top.goals) details.push(`${top.goals} goal${top.goals === 1 ? "" : "s"}`);
  if (top.coachVotes != null) details.push(`coaches' votes ${top.coachVotes}`);
  const tail = details.length ? ` — ${details.join(", ")}` : "";
  return `${winner} won by ${margin}. Model 3-vote: ${top.name} (${pct(top.p3)}${tail}).`;
}

function mentionsTopPick(match) {
  const top = match.votes?.[0];
  const report = match.report;
  if (!top || !report?.text) return false;
  const surname = top.name.split(" ").slice(-1)[0].replace(/[^A-Za-z'-]/g, "").toLowerCase();
  if (surname.length < 4) return false;
  return report.text.toLowerCase().includes(surname);
}

function MatchCard({ match }) {
  const homeWon = match.homeScore > match.awayScore;
  const top = match.votes?.[0];
  const triple = match.triples?.[0];
  return (
    <article className="match-card">
      <header>
        <span className="round-tag">{match.round === 0 ? "OR" : `R${match.round}`}</span>
        <span className="match-meta">
          {formatDate(match.date)}
          {match.venue ? ` · ${match.venue}` : ""}
        </span>
      </header>
      <h3 className="match-score">
        <span className={homeWon ? "winner" : ""}>
          <TeamLogo team={match.home} size={18} />
          {match.home}
        </span>
        <b>
          {match.homeScore ?? "—"}–{match.awayScore ?? "—"}
        </b>
        <span className={!homeWon ? "winner" : ""}>
          {match.away}
          <TeamLogo team={match.away} size={18} />
        </span>
      </h3>
      {summarise(match) ? <p className="match-summary">{summarise(match)}</p> : null}
      {match.report ? (
        <div className="match-report">
          <p className="report-text">{match.report.text}</p>
          <p className="report-credit">
            <a href={match.report.url} target="_blank" rel="noreferrer">
              {match.report.headline || "Match report"}
            </a>{" "}
            · {match.report.source}
            {mentionsTopPick(match) ? (
              <span className="report-flag">names our 3-vote pick</span>
            ) : null}
          </p>
        </div>
      ) : null}
      <div className="vote-table">
        <div className="vote-head">
          <span>Player</span>
          <span title="Probability of receiving 3 votes">3</span>
          <span title="Probability of receiving 2 votes">2</span>
          <span title="Probability of receiving 1 vote">1</span>
          <span>Game</span>
        </div>
        {(match.votes ?? []).slice(0, 4).map((vote) => (
          <div
            className={vote.id === top?.id ? "vote-row top" : "vote-row"}
            key={vote.id}
          >
            <span className="vote-player">
              <i className="dot" style={{ background: teamColor(vote.team) }} />
              {vote.name}
            </span>
            <ProbCell value={vote.p3} boost={1.15} />
            <ProbCell value={vote.p2} />
            <ProbCell value={vote.p1} boost={0.85} />
            <span className="vote-stats">
              {vote.disposals != null ? `${vote.disposals}d` : "—"}
              {vote.goals ? ` ${vote.goals}g` : ""}
              {vote.coachVotes != null ? ` · ${vote.coachVotes} cv` : ""}
            </span>
          </div>
        ))}
      </div>
      {triple ? (
        <p className="match-triple">
          Likeliest exact 3-2-1: {triple.players.join(" / ")} ({pct(triple.p, 1)})
        </p>
      ) : null}
    </article>
  );
}

export default function MatchesView({ matches, rounds }) {
  const available = useMemo(
    () => (rounds ?? []).filter((round) => (matches ?? []).some((match) => match.round === round.number)),
    [matches, rounds]
  );
  const [selected, setSelected] = useState(null);
  const active = selected ?? available[available.length - 1]?.number ?? 0;
  const visible = (matches ?? []).filter((match) => match.round === active);

  if (!matches?.length) return null;

  return (
    <section className="panel matches-panel">
      <div className="panel-head">
        <h2>Match votes</h2>
        <span className="hint">
          predicted 3-2-1 probabilities, scoreline and the standout players for every game
        </span>
      </div>
      <p className="stat-legend">
        <b>Game</b>: d = disposals · g = goals · cv = coaches' votes. <b>3 / 2 / 1</b> are the
        model's probabilities for that vote — darker cells and longer bars mean more likely.
      </p>
      <div className="round-switcher">
        {available.map((round) => (
          <button
            type="button"
            key={round.number}
            className={round.number === active ? "active" : ""}
            onClick={() => setSelected(round.number)}
          >
            {round.label}
          </button>
        ))}
      </div>
      <div className="match-grid">
        {visible.map((match) => (
          <MatchCard key={match.id} match={match} />
        ))}
      </div>
    </section>
  );
}
