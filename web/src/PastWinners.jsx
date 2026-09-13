import TeamLogo from "./TeamLogo.jsx";

const RECORDS = [
  {
    season: 2018,
    expected: "Tom Mitchell",
    expectedTeam: "Hawthorn",
    expectedP: 0.927,
    expectedVotes: 31.4,
    actual: "Tom Mitchell",
    actualTeam: "Hawthorn",
    actualVotes: 28,
    rank: 1,
    actualP: 0.927,
    hit: true,
  },
  {
    season: 2019,
    expected: "Patrick Dangerfield",
    expectedTeam: "Geelong Cats",
    expectedP: 0.502,
    expectedVotes: 25.7,
    actual: "Nat Fyfe",
    actualTeam: "Fremantle",
    actualVotes: 33,
    rank: 2,
    actualP: 0.205,
    hit: false,
  },
  {
    season: 2020,
    expected: "Lachie Neale",
    expectedTeam: "Brisbane Lions",
    expectedP: 0.793,
    expectedVotes: 26.9,
    actual: "Lachie Neale",
    actualTeam: "Brisbane Lions",
    actualVotes: 31,
    rank: 1,
    actualP: 0.793,
    hit: true,
  },
  {
    season: 2021,
    expected: "Ollie Wines",
    expectedTeam: "Port Adelaide",
    expectedP: 0.471,
    expectedVotes: 32.2,
    actual: "Ollie Wines",
    actualTeam: "Port Adelaide",
    actualVotes: 36,
    rank: 1,
    actualP: 0.471,
    hit: true,
  },
  {
    season: 2022,
    expected: "Clayton Oliver",
    expectedTeam: "Melbourne",
    expectedP: 0.378,
    expectedVotes: 28.7,
    actual: "Patrick Cripps",
    actualTeam: "Carlton",
    actualVotes: 29,
    rank: 4,
    actualP: 0.169,
    hit: false,
  },
  {
    season: 2023,
    expected: "Nick Daicos",
    expectedTeam: "Collingwood",
    expectedP: 0.275,
    expectedVotes: 28.9,
    actual: "Lachie Neale",
    actualTeam: "Brisbane Lions",
    actualVotes: 31,
    rank: 8,
    actualP: 0.024,
    hit: false,
  },
  {
    season: 2024,
    expected: "Patrick Cripps",
    expectedTeam: "Carlton",
    expectedP: 0.527,
    expectedVotes: 31.6,
    actual: "Patrick Cripps",
    actualTeam: "Carlton",
    actualVotes: 45,
    rank: 1,
    actualP: 0.527,
    hit: true,
  },
  {
    season: 2025,
    expected: "Nick Daicos",
    expectedTeam: "Collingwood",
    expectedP: 0.331,
    expectedVotes: 31.1,
    actual: "Matt Rowell",
    actualTeam: "Gold Coast SUNS",
    actualVotes: 39,
    rank: 7,
    actualP: 0.035,
    hit: false,
  },
];

const SUMMARY = {
  hits: RECORDS.filter((record) => record.hit).length,
  meanWinnerP: RECORDS.reduce((sum, record) => sum + record.actualP, 0) / RECORDS.length,
  meanVoteError:
    RECORDS.reduce((sum, record) => sum + Math.abs(record.expectedVotes - record.actualVotes), 0) /
    RECORDS.length,
};

function pct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function ordinal(rank) {
  const suffixes = ["th", "st", "nd", "rd"];
  const remainder = rank % 100;
  return `${rank}${suffixes[(remainder - 20) % 10] || suffixes[remainder] || suffixes[0]}`;
}

export default function PastWinners() {
  return (
    <section className="panel past-panel">
      <div className="panel-head">
        <h2>Past winners</h2>
        <span className="hint">
          what the model expected before each count vs what actually happened
        </span>
      </div>
      <p className="stat-legend">
        Every season is forecast with only earlier data, using the same rolling procedure as the
        backtests (10,000 simulations, suspensions applied). <b>Expected</b> is the player with the
        highest P(first or joint) before the count; <b>winner rank</b> is where the actual medallist
        sat in that ordering. 2018–2020 used the same procedure with less training history, which is
        why the model grows more confident over time.
      </p>

      <div className="summary-strip">
        <div>
          <span>Favourite called the winner</span>
          <strong>
            {SUMMARY.hits}/{RECORDS.length}
          </strong>
        </div>
        <div>
          <span>Average winner probability</span>
          <strong>{pct(SUMMARY.meanWinnerP)}</strong>
        </div>
        <div>
          <span>Mean winner vote error</span>
          <strong>{SUMMARY.meanVoteError.toFixed(1)} votes</strong>
        </div>
      </div>

      <div className="winner-grid">
        {RECORDS.map((record) => (
          <article className={record.hit ? "winner-card hit" : "winner-card miss"} key={record.season}>
            <header>
              <span className="round-tag">{record.season}</span>
              <span className={record.hit ? "verdict hit" : "verdict miss"}>
                {record.hit ? "called it" : "missed"}
              </span>
            </header>
            <div className="winner-rows">
              <div className="winner-row">
                <span className="winner-label">Expected</span>
                <span className="winner-name">
                  <TeamLogo team={record.expectedTeam} size={16} />
                  {record.expected}
                </span>
                <b>{pct(record.expectedP)}</b>
              </div>
              <div className="winner-row">
                <span className="winner-label">Won</span>
                <span className="winner-name">
                  <TeamLogo team={record.actualTeam} size={16} />
                  {record.actual}
                </span>
                <b>{record.actualVotes} votes</b>
              </div>
            </div>
            <p className="winner-note">
              Model expected {record.expectedVotes.toFixed(1)} votes for {record.expected}; the
              medallist ranked {ordinal(record.rank)} by win probability ({pct(record.actualP)}).
            </p>
          </article>
        ))}
      </div>
    </section>
  );
}
