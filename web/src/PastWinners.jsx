import TeamLogo from "./TeamLogo.jsx";

const RECORDS = [
  {
    season: 2014,
    expected: "Gary Ablett",
    expectedTeam: "Gold Coast SUNS",
    expectedP: 0.183,
    expectedVotes: 19.9,
    actual: "Matt Priddis",
    actualTeam: "West Coast Eagles",
    actualVotes: 26,
    rank: 3,
    actualP: 0.138,
    hit: false,
  },
  {
    season: 2015,
    expected: "Matt Priddis",
    expectedTeam: "West Coast Eagles",
    expectedP: 0.465,
    expectedVotes: 28.7,
    actual: "Nat Fyfe",
    actualTeam: "Fremantle",
    actualVotes: 31,
    rank: 2,
    actualP: 0.174,
    hit: false,
  },
  {
    season: 2016,
    expected: "Patrick Dangerfield",
    expectedTeam: "Geelong Cats",
    expectedP: 0.848,
    expectedVotes: 37.3,
    actual: "Patrick Dangerfield",
    actualTeam: "Geelong Cats",
    actualVotes: 35,
    rank: 1,
    actualP: 0.848,
    hit: true,
  },
  {
    season: 2017,
    expected: "Dustin Martin",
    expectedTeam: "Richmond",
    expectedP: 0.921,
    expectedVotes: 37.3,
    actual: "Dustin Martin",
    actualTeam: "Richmond",
    actualVotes: 36,
    rank: 1,
    actualP: 0.921,
    hit: true,
  },
  {
    season: 2018,
    expected: "Tom Mitchell",
    expectedTeam: "Hawthorn",
    expectedP: 0.905,
    expectedVotes: 31.7,
    actual: "Tom Mitchell",
    actualTeam: "Hawthorn",
    actualVotes: 28,
    rank: 1,
    actualP: 0.905,
    hit: true,
  },
  {
    season: 2019,
    expected: "Patrick Dangerfield",
    expectedTeam: "Geelong Cats",
    expectedP: 0.504,
    expectedVotes: 27.8,
    actual: "Nat Fyfe",
    actualTeam: "Fremantle",
    actualVotes: 33,
    rank: 2,
    actualP: 0.381,
    hit: false,
  },
  {
    season: 2020,
    expected: "Lachie Neale",
    expectedTeam: "Brisbane Lions",
    expectedP: 0.812,
    expectedVotes: 27.1,
    actual: "Lachie Neale",
    actualTeam: "Brisbane Lions",
    actualVotes: 31,
    rank: 1,
    actualP: 0.812,
    hit: true,
  },
  {
    season: 2021,
    expected: "Ollie Wines",
    expectedTeam: "Port Adelaide",
    expectedP: 0.464,
    expectedVotes: 32.7,
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
    expectedP: 0.254,
    expectedVotes: 28.3,
    actual: "Patrick Cripps",
    actualTeam: "Carlton",
    actualVotes: 29,
    rank: 1,
    actualP: 0.254,
    hit: true,
  },
  {
    season: 2023,
    expected: "Marcus Bontempelli",
    expectedTeam: "Western Bulldogs",
    expectedP: 0.306,
    expectedVotes: 29.1,
    actual: "Lachie Neale",
    actualTeam: "Brisbane Lions",
    actualVotes: 31,
    rank: 7,
    actualP: 0.036,
    hit: false,
  },
  {
    season: 2024,
    expected: "Patrick Cripps",
    expectedTeam: "Carlton",
    expectedP: 0.632,
    expectedVotes: 33.6,
    actual: "Patrick Cripps",
    actualTeam: "Carlton",
    actualVotes: 45,
    rank: 1,
    actualP: 0.632,
    hit: true,
  },
  {
    season: 2025,
    expected: "Nick Daicos",
    expectedTeam: "Collingwood",
    expectedP: 0.401,
    expectedVotes: 32.3,
    actual: "Matt Rowell",
    actualTeam: "Gold Coast SUNS",
    actualVotes: 39,
    rank: 6,
    actualP: 0.050,
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
        Every season is forecast with only earlier data (10,000 simulations, suspensions applied).
        <b> Expected</b> is the player with the highest P(first or joint) before the count;
        <b> winner rank</b> is where the actual medallist sat in that ordering. From 2021 the
        temperature, effect scale and historical-effect selection are rolling (chosen on earlier
        seasons only); the 2014–2020 cards reuse the production calibration and effect settings,
        which were chosen later, so treat those as illustrative — the model also had far less
        history to learn from in the early seasons.
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
