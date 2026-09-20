import TeamLogo from "./TeamLogo.jsx";

const RECORDS = [
  {
    season: 2014,
    expected: "Gary Ablett",
    expectedTeam: "Gold Coast SUNS",
    expectedP: 0.194,
    expectedVotes: 20.4,
    actual: "Matt Priddis",
    actualTeam: "West Coast Eagles",
    actualVotes: 26,
    rank: 3,
    actualP: 0.141,
    hit: false,
  },
  {
    season: 2015,
    expected: "Matt Priddis",
    expectedTeam: "West Coast Eagles",
    expectedP: 0.504,
    expectedVotes: 29.8,
    actual: "Nat Fyfe",
    actualTeam: "Fremantle",
    actualVotes: 31,
    rank: 2,
    actualP: 0.167,
    hit: false,
  },
  {
    season: 2016,
    expected: "Patrick Dangerfield",
    expectedTeam: "Geelong Cats",
    expectedP: 0.878,
    expectedVotes: 39.0,
    actual: "Patrick Dangerfield",
    actualTeam: "Geelong Cats",
    actualVotes: 35,
    rank: 1,
    actualP: 0.878,
    hit: true,
  },
  {
    season: 2017,
    expected: "Dustin Martin",
    expectedTeam: "Richmond",
    expectedP: 0.933,
    expectedVotes: 38.1,
    actual: "Dustin Martin",
    actualTeam: "Richmond",
    actualVotes: 36,
    rank: 1,
    actualP: 0.933,
    hit: true,
  },
  {
    season: 2018,
    expected: "Tom Mitchell",
    expectedTeam: "Hawthorn",
    expectedP: 0.733,
    expectedVotes: 31.6,
    actual: "Tom Mitchell",
    actualTeam: "Hawthorn",
    actualVotes: 28,
    rank: 1,
    actualP: 0.733,
    hit: true,
  },
  {
    season: 2019,
    expected: "Patrick Dangerfield",
    expectedTeam: "Geelong Cats",
    expectedP: 0.411,
    expectedVotes: 28.6,
    actual: "Nat Fyfe",
    actualTeam: "Fremantle",
    actualVotes: 33,
    rank: 2,
    actualP: 0.384,
    hit: false,
  },
  {
    season: 2020,
    expected: "Lachie Neale",
    expectedTeam: "Brisbane Lions",
    expectedP: 0.808,
    expectedVotes: 26.8,
    actual: "Lachie Neale",
    actualTeam: "Brisbane Lions",
    actualVotes: 31,
    rank: 1,
    actualP: 0.808,
    hit: true,
  },
  {
    season: 2021,
    expected: "Ollie Wines",
    expectedTeam: "Port Adelaide",
    expectedP: 0.464,
    expectedVotes: 32.4,
    actual: "Ollie Wines",
    actualTeam: "Port Adelaide",
    actualVotes: 36,
    rank: 1,
    actualP: 0.464,
    hit: true,
  },
  {
    season: 2022,
    expected: "Patrick Cripps",
    expectedTeam: "Carlton",
    expectedP: 0.318,
    expectedVotes: 29.3,
    actual: "Patrick Cripps",
    actualTeam: "Carlton",
    actualVotes: 29,
    rank: 1,
    actualP: 0.318,
    hit: true,
  },
  {
    season: 2023,
    expected: "Marcus Bontempelli",
    expectedTeam: "Western Bulldogs",
    expectedP: 0.368,
    expectedVotes: 29.9,
    actual: "Lachie Neale",
    actualTeam: "Brisbane Lions",
    actualVotes: 31,
    rank: 5,
    actualP: 0.041,
    hit: false,
  },
  {
    season: 2024,
    expected: "Patrick Cripps",
    expectedTeam: "Carlton",
    expectedP: 0.711,
    expectedVotes: 35.0,
    actual: "Patrick Cripps",
    actualTeam: "Carlton",
    actualVotes: 45,
    rank: 1,
    actualP: 0.711,
    hit: true,
  },
  {
    season: 2025,
    expected: "Nick Daicos",
    expectedTeam: "Collingwood",
    expectedP: 0.419,
    expectedVotes: 32.6,
    actual: "Matt Rowell",
    actualTeam: "Gold Coast SUNS",
    actualVotes: 39,
    rank: 5,
    actualP: 0.063,
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
        sat in that ordering. The earliest seasons (2014–2017) used the same machinery with the
        limited history available from 2013, which is why the model grows more confident over time.
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
