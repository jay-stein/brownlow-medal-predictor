const ROLLING = [
  ["2021", "2.78", "0.93", "0.47", "yes", "44.0%"],
  ["2022", "1.87", "1.00", "0.73", "no", "17.0%"],
  ["2023", "2.05", "0.93", "0.53", "no", "1.9%"],
  ["2024", "3.79", "0.73", "0.13", "yes", "49.5%"],
  ["2025", "2.65", "0.93", "0.73", "no", "3.8%"],
];

const VARIANTS = [
  ["Legacy (100-model Normal draws)", "4.77", "0.20", "0/5", "0.000"],
  ["Coaches' votes only", "2.85", "0.88", "2/5", "0.217"],
  ["Stacking ensemble", "2.65", "0.93", "1/5", "0.186"],
  ["Season form (production, Student-t(4) effects)", "2.63", "0.91", "2/5", "0.232"],
  ["Elo / travel / close-game context", "2.66", "0.96", "2/5", "0.238"],
  ["Direct Plackett-Luce objective", "3.09", "0.85", "2/5", "0.216"],
];

const REFERENCES = [
  {
    name: "AFL.com.au",
    url: "https://www.afl.com.au",
    note: "official match reports — headlines and short excerpts are shown on each match tile, linked to the source article — plus the public match index used to map games.",
  },
  {
    name: "AFL Tables",
    url: "https://afltables.com",
    note: "historical player statistics and Brownlow votes (2012–2025 label seasons).",
  },
  {
    name: "AFL Coaches Association",
    url: "https://aflcoaches.com.au/awards/the-aflca-champion-player-of-the-year-award/leaderboard",
    note: "per-match coaching panel votes (0–10 per player), the strongest non-umpire signal in the model.",
  },
  {
    name: "fitzRoy",
    url: "https://github.com/jimmyday12/fitzRoy",
    note: "James Day's R package, used to fetch the AFL API and AFL Tables data.",
  },
  {
    name: "Squiggle",
    url: "https://squiggle.com.au",
    note: "club logo images used in the visualisation; club names and logos remain the trademarks of their clubs.",
  },
  {
    name: "Wikipedia Brownlow Medal articles",
    url: "https://en.wikipedia.org/wiki/2024_Brownlow_Medal",
    note: "ineligible leading vote-getters by season, used for award-stage eligibility.",
  },
  {
    name: "Zero Hanger MRO/tribunal tracker",
    url: "https://www.zerohanger.com/2026-afl-suspensions-mro-and-tribunal-tracker/",
    note: "current-season suspension records behind the ineligible flags.",
  },
];

export default function NerdyStuff({ meta }) {
  return (
    <section className="panel nerdy-panel">
      <div className="panel-head">
        <h2>Nerdy Stuff</h2>
        <span className="hint">the method in full, how it performs, where it fails, and who to credit</span>
      </div>

      <div className="nerdy-grid">
        <article className="nerdy-card">
          <h3>How the forecast works</h3>
          <ol>
            <li>
              <b>Data.</b> AFL API (Champion Data) player stats and results since 2012 via fitzRoy; AFL
              Tables votes; AFL Coaches Association panel votes; official match reports.
            </li>
            <li>
              <b>Audited labels.</b> Every player-game is labelled voted, genuine zero, or unresolved.
              Unresolved rows are quarantined from training rather than treated as zeros.
            </li>
            <li>
              <b>Features.</b> 139 per player-game: raw stats, within-team shares, match context,
              coaches' votes and leave-one-game-out season form (per-game means, coach-vote
              total/rate/rank, team win rate).
            </li>
            <li>
              <b>Score generator.</b> A match-grouped LambdaMART ranker trained on earlier seasons
              only; boosting rounds chosen by time-ordered season validation.
            </li>
            <li>
              <b>Allocation.</b> The 3-2-1 vote is a fixed six-vote budget, modelled as a sequential
              Plackett-Luce draw. Marginal probabilities are exact and audited against exhaustive
              enumeration and simulation.
            </li>
            <li>
              <b>Calibration.</b> Temperature and persistent-effect scale are chosen jointly, with
              match likelihood integrated over the same player-season effects the simulator draws,
              combined with season CRPS under a predeclared rule. The effect shape is variance-normalised
              Student-t(4): heavier tails than a Gaussian fit the rolling seasons better (CRPS 2.63
              versus 2.68, 90% coverage 0.91 versus 0.85).
            </li>
            <li>
              <b>Uncertainty.</b> One persistent player-season effect per player, plus a partially
              pooled historical effect built from prior-season residuals, so chronically underrated
              players shift up without counting past votes twice.
            </li>
            <li>
              <b>Eligibility.</b> Suspended players keep their simulated votes (they affect everyone's
              totals) but are excluded from winning the medal.
            </li>
          </ol>
        </article>

        <article className="nerdy-card">
          <h3>Rolling performance (2021-2025)</h3>
          <p className="nerdy-note">
            Every season is reproduced with only earlier information: the score model trains on
            earlier labels, calibration and effects are selected on earlier out-of-sample seasons,
            then the season is forecast once. Award probabilities respect suspensions. These seasons
            informed several design choices, so treat the table as a <b>post-selection diagnostic</b>:
            automated selection across all candidate families on evidence alone scores 2.65 CRPS with
            1/5 favourites on the same seasons, and <b>2026 is the only untouched target</b>. With the
            historical effects applied at the shape-consistent selection (50/2/0), the same five
            seasons score 2.55 CRPS at a 25.9% average winner probability.
          </p>
          <div className="table-scroll">
            <table className="nerdy-table">
              <thead>
                <tr>
                  <th>Season</th>
                  <th>Contender CRPS</th>
                  <th>90% cov</th>
                  <th>50% cov</th>
                  <th>Fav won</th>
                  <th>Winner P</th>
                </tr>
              </thead>
            <tbody>
              {ROLLING.map((row) => (
                <tr key={row[0]}>
                  {row.map((cell, index) => (
                    <td key={index}>{cell}</td>
                  ))}
                </tr>
              ))}
              <tr className="nerdy-total">
                <td>Mean</td>
                <td>2.63</td>
                <td>0.91</td>
                <td>0.52</td>
                <td>2/5</td>
                <td>23.2%</td>
              </tr>
            </tbody>
            </table>
          </div>
          <p className="nerdy-note">
            <b>CRPS</b> (continuous ranked probability score) scores the whole predicted distribution
            against the observed total; it rewards being close <i>and</i> honest about spread, so a
            confidently wrong model is punished and so is a needlessly wide one. Lower is better.
          </p>
        </article>

        <article className="nerdy-card">
          <h3>What was compared</h3>
          <div className="table-scroll">
            <table className="nerdy-table">
              <thead>
                <tr>
                  <th>Approach</th>
                  <th>CRPS</th>
                  <th>90% cov</th>
                  <th>Fav won</th>
                  <th>Winner P</th>
                </tr>
              </thead>
            <tbody>
              {VARIANTS.map((row) => (
                <tr key={row[0]}>
                  {row.map((cell, index) => (
                    <td key={index}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
            </table>
          </div>
          <p className="nerdy-note">
            The original overconfident model (100 feature-bagged XGBoost models with independent
            Normal draws) is reconstructed faithfully as the baseline. It covered a fifth of its 90%
            intervals and never picked the winner. Recency windows, recency weighting and the
            centre-bounce ablation were all within noise of the all-history baseline.
          </p>
        </article>

        <article className="nerdy-card">
          <h3>Known shortcomings</h3>
          <ul>
            <li>
              <b>Record vote inflation.</b> 2024 produced the highest count in history; the model's
              centre for such seasons is still too low even after the historical player effects and
              heavy-tailed season effects.
            </li>
            <li>
              <b>Context-free utilities.</b> A player's score depends on their own features and team
              shares, not directly on the strength of teammates and opponents.
            </li>
            <li>
              <b>Post-season boundary.</b> This is a count-night conditional forecast: 2026
              statistics, coaches' votes and season form are observed. It is not a live mid-season
              forecast.
            </li>
            <li>
              <b>No quarter-level data.</b> Quarter-by-quarter player stats are not publicly
              available (the AFL's period endpoints are premium), so late-game impact is unmodelled.
            </li>
            <li>
              <b>Small evaluation sample.</b> Five rolling seasons and five medal outcomes cannot
              prove winner-probability calibration; the model is honest rather than decisive.
            </li>
            <li>
              <b>Selection optimism.</b> Model family, effect shape and calibration were chosen after
              seeing 2021-2025, so those records flatter the procedure. Automated selection on
              evidence alone scores 2.65 CRPS and 1/5 favourites; 2026 is the clean test.
            </li>
          </ul>
        </article>

        <article className="nerdy-card">
          <h3>Technical settings</h3>
          <ul className="nerdy-settings">
            <li>
              <span>Score generator</span>
              <b>match-grouped LambdaMART (rank:ndcg)</b>
            </li>
            <li>
              <span>Temperature</span>
              <b>{meta.tau ?? "-"}</b>
            </li>
            <li>
              <span>Persistent-effect scale</span>
              <b>{meta.effectScale ?? "-"}</b>
            </li>
            <li>
              <span>Persistent-effect shape</span>
              <b>
                {meta.effectDistribution === "student_t"
                  ? `Student-t(df=${meta.effectTdf ?? 4})`
                  : (meta.effectDistribution ?? "Normal")}
              </b>
            </li>
            <li>
              <span>Historical player effects</span>
              <b>
                {meta.playerEffect
                  ? `shrinkage ${meta.playerEffect.shrinkage} games, mapping ${meta.playerEffect.mapping}, ${
                      Number(meta.playerEffect.half_life) > 0
                        ? `half-life ${meta.playerEffect.half_life}`
                        : "no decay"
                    }`
                  : "shrinkage 50 games, mapping 2, no decay"}
              </b>
            </li>
            <li>
              <span>Simulations</span>
              <b>{Number(meta.nSims ?? 10000).toLocaleString()} Plackett-Luce counts</b>
            </li>
            <li>
              <span>Eligibility</span>
              <b>{meta.nIneligible ?? 0} suspended players excluded</b>
            </li>
          </ul>
        </article>

        <article className="nerdy-card nerdy-references">
          <h3>References and credits</h3>
          <ul>
            {REFERENCES.map((reference) => (
              <li key={reference.name}>
                <a href={reference.url} target="_blank" rel="noreferrer">
                  {reference.name}
                </a>{" "}
                — {reference.note}
              </li>
            ))}
          </ul>
          <p className="nerdy-note">
            Code and full documentation:{" "}
            <a
              href="https://github.com/jay-stein/brownlow-medal-predictor"
              target="_blank"
              rel="noreferrer"
            >
              github.com/jay-stein/brownlow-medal-predictor
            </a>{" "}
            (MIT licence).
          </p>
        </article>
      </div>
    </section>
  );
}
