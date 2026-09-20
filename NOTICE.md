# Data sources and attribution

This project's code is released under the [MIT License](LICENSE). The
underlying football data is the property of its providers and remains subject
to their terms. Small prediction-source snapshots are committed only to keep
the pipeline reproducible; the large raw extracts are not redistributed and are
fetched at run time.

- [fitzRoy](https://github.com/jimmyday12/fitzRoy), the excellent R package by
  James Day and contributors, used to fetch:
  - AFL API player statistics, match results and squad details (Champion Data)
  - [AFL Tables](https://afltables.com) player statistics and Brownlow votes
- [AFL Coaches Association](https://aflcoaches.com.au) Champion Player leaderboards:
  per-match coaching panel votes.
- [AFL.com.au](https://www.afl.com.au) match centre pages: short excerpts from official
  match reports (headline, lead paragraph) with links back to the source articles.
- AFL CFS match-centre feed (`/cfs/afl/matchItem`): official scoring timeline per match
  (scorer, period, clock, running score), used for late-game and momentum features.
- [Squiggle](https://squiggle.com.au) API: club logo images used in the web visualisation.
  Club logos and names are trademarks of their respective clubs.
- AFL.com.au Brownlow award endpoint, used for Brownlow predictor votes.
- Betfair Data Supplier API, used for bookmaker vote predictions.
- ESPN Brownlow predictor and WheeloRatings predictions, used for comparison.
