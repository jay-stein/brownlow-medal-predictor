# The Brownlow model, explained simply

A high-school-level guide to how the model works, in three levels of detail. For the
full treatment, see [`methodology.md`](methodology.md); for the 2026 numbers, see
[`forecast-2026.md`](forecast-2026.md).

## The 30-second version

We fed a computer 12+ years of AFL games and the Brownlow votes umpires gave. It learned
what a vote-winning game looks like. Now, for any season, it guesses the 3-2-1 for every
single match — then we replay that whole season **10,000 times** and count how often each
player wins the medal.

## The 3-minute version

Think of it as five steps:

**1. Collect the evidence.** For every player, every game: disposals, goals, tackles,
whether their team won, and — importantly — the **coaches' votes** (two coaches each award
5-4-3-2-1, so a player gets 0–10). Coaches vote like umpires do, so they're a strong hint.
We also have the real Brownlow 3-2-1 for past years, which is what the model learns from.

**2. Learn the pattern.** The model compares thousands of games: "what did 3-vote games
look like vs 0-vote games?" It's like a coach learning to scan a stats sheet and go "yep,
that's a best-on-ground game."

**3. Score each game.** For a new game, every player gets a **vote-worthiness score** —
not "he'll get exactly 2 votes", just "he's more likely than most to be in the votes."

**4. Turn scores into 3-2-1.** Humans vote, so we make it a weighted raffle: the higher
your score, the more raffle tickets you hold. Draw for the 3-vote player, then the 2 from
the players left, then the 1. Repeat for all ~207 matches in a season. (This is the part
that lets a slightly-worse game sometimes still get votes — umpires are humans, not
calculators.)

**5. Replay the season 10,000 times.** Every replay gives a full vote tally. Count how
often Daicos tops the list → that's his probability. Look at the middle 90% of his
simulated tallies → that's the "35–51 votes" interval.

**Then we check ourselves.** We hide a season from the model, train only on the years
before it, and see how its predictions went. That's the rolling test.

## The deeper dive

### Why "score" and not "predict 2.7 votes"?

A prediction of 2.7 is meaningless — votes only come in 0, 1, 2, 3. So instead of hitting
an exact number, the model **ranks** players within each game: "who's most likely to be
the standout?" It's graded on getting the order right, not the exact amount. A bit like
ranking a class on an exam rather than guessing everyone's marks.

### What's inside the "learning"?

A system called **LambdaMART**. Imagine 200 small decision trees: "If disposals > 30 and
goals > 2 → very likely votes." Each new tree fixes mistakes the previous trees made. We
stop adding trees when the model starts memorising the training years instead of learning
general patterns — like a student who memorises past exam answers instead of the method.

### What did we tell the model about each player's season?

This year we added **season form**: a player's per-game averages for the season,
*excluding the game being predicted* (so it can't cheat by seeing its own answer). We also
let it know whether the game was close, whether the team won, and so on.

### The raffle dial (temperature τ)

The raffle has a "sharpness" dial. Sharp: the top-scored player almost always gets the 3.
Flat: everyone has a decent chance. We pick that dial by testing on past seasons —
currently 0.8.

### The season-long luck bump (the t4 thing)

Stats don't capture everything: new role, confidence, niggles, umpires warming to a
player. So at the start of each simulated season, every player gets one hidden
bonus/penalty that lasts **all year** — a "good year / bad year" card. How often those
cards are extreme is what **t4** controls: rare under the old bell-curve (Gaussian)
setting, occasional under t4 (a Student-t distribution with 4 degrees of freedom). That
occasional extreme year is how counts get 45-vote seasons.

It has nothing to do with carrying form across seasons — each simulated season rolls
fresh cards.

### The cross-season nudge (historical effects)

Separately, if the model has *underestimated* a player for a few seasons in a row, we
nudge them up a little — the "gets noticed in following seasons" idea. We only apply a
*shrunken* fraction of the past surprise, so one fluke year doesn't fool us. Under the
current settings that nudge is deliberately gentle (50 games of shrinkage, 2 utility
units per vote).

### Scoring the model (CRPS and coverage)

Two main checks:

- **CRPS** asks: were your probabilities close to what happened, *without* being vague?
  Saying "someone between 0 and 60 wins" is technically safe but useless.
- **Coverage** asks: when we say "90% interval", does the truth land inside it 9 times
  out of 10? We track that.

### The one rule that matters most: no time travel

When predicting 2021, the model may only use 2016–2020. When predicting 2026, only
2012–2025. We recently audited this and found a subtle version: if we try several models
and pick the one with the best score on the same years we're reporting, the reported score
is too flattering. So the docs now mark those years as a "diagnostic", and **2026 — which
no decision has touched — is the real test.**

### The result right now

Daicos **43.3 expected votes**, a 90% range of **35–51**, and a **90.7%** chance of
finishing first or equal-first. Smith, Cripps and Bontempelli head the chasing pack.
