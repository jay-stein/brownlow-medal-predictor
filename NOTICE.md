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
- AFL.com.au Brownlow award endpoint, used for Brownlow predictor votes.
- Betfair Data Supplier API, used for bookmaker vote predictions.
- ESPN Brownlow predictor and WheeloRatings predictions, used for comparison.
