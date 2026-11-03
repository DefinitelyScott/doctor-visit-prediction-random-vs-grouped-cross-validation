# Design notes

Written so every choice here can be explained out loud in an interview.

## The question, and why this one

A standard "load, split, train, report" mini-project shows you can call `.fit()`.
The question here is more useful: **is my validation score telling me about new
people, or about people the model has already seen?** Row-level leakage from
repeated entities (patients, customers, devices, users) is one of the most common
ways a reported metric ends up higher than production performance. A repo that
measures that effect, rather than just warning about it, is a better
demonstration of judgement.

A modest or even disappointing outcome would still be reportable. As it happens
the effect is large for tree models and negligible for the linear model, which is
a more interesting result than "everything drops a bit".

## Why this dataset

- Bundled with `statsmodels`: no download, no licence question, no URL to rot.
- Not a tutorial default (no Titanic, Iris, MNIST).
- Real repeated structure: 20,190 rows collapse to 2,760 distinct feature
  profiles, with group sizes clustering at 3 and 5, consistent with people
  observed over 3 or 5 years. I checked this before choosing the dataset; it is
  not assumed.
- Moderate signal (best AUC about 0.7) leaves room for differences to show.
- Binary target `any_visit` (68.8% positive) keeps the metric story simple.

Considered and rejected: scikit-learn's breast-cancer data (too common, no
repeated entities, near-ceiling scores) and `fetch_*` datasets (network
dependence).

## Person proxy: the biggest judgement call

The extract has no person ID, so I group rows with identical feature values. The
alternative was to skip grouping and only argue about leakage in prose, which
proves nothing. The proxy errs in both directions (see README limitations), and I
say so rather than hiding it. Two things back it up: the group-size pattern above,
and the direct measurement that random CV leaks 98.6% of test rows while grouped CV
leaks 0.0%. That second number is by construction under the proxy, so it
validates the splitter, not the proxy itself.

## Why these four models

A ladder: no-skill baseline, linear, bagged deep trees, boosted trees.

- `baseline_prior` (`DummyClassifier`) gives the floor and a log-loss reference;
  the forest's grouped log loss only looks bad because it sits above this.
- Logistic regression, standardised: interpretable and low capacity, the control
  that should not benefit from memorisation.
- Random forest with `min_samples_leaf=1`: deliberately deep, because depth is
  what allows memorisation. This is a setting chosen to test the hypothesis, not
  a tuned "best" model, and I would say so if asked.
- `HistGradientBoostingClassifier`: a strong modern default, in between.

## Metrics

AUC (ranking, threshold-free), Brier score and log loss (probability quality).
Accuracy was skipped: with 69% positives it is dominated by the base rate. Log loss
is what exposes the forest being confidently wrong.

## CV design

- `StratifiedKFold` vs `StratifiedGroupKFold`, same seeds, 5 folds x 3 repeats.
  Stratification keeps the 69/31 mix per fold; repeats show split-to-split spread.
- Within a scheme every model sees identical folds, so I report fold-paired
  differences (mean, SD, folds won). Between schemes the splits differ, so I only
  compare means, and say so.
- No hyperparameter tuning, so there is no nested CV to get wrong. Tuning would
  be the next step and would need nesting inside the grouped folds.

## Engineering choices

- Small modules, each with one job; the splitter yields indices so tests can
  assert the properties that matter (partition, no group overlap, determinism).
- Fail loudly: `load_visits` raises if columns are missing or NaNs appear.
- `leaked_share` is a first-class measurement, not a comment.
- Tests include a synthetic dataset where the answer is known by design: labels
  are noise tied to the person, so a forest must score ~1.0 under random CV and
  near chance under grouped CV. If the splitter or scorer were wrong, that test
  would fail.
- Pinned requirements; one-command reproduction; seeded everywhere.

## What I would do next

- Find a version of the data with true person IDs, to quantify how wrong the
  proxy is.
- Tune each model inside grouped CV (nested) and see whether the ranking holds.
- Add calibration curves to show *how* the forest's probabilities fail.
- Try a regularised forest (larger `min_samples_leaf`) to show the gap shrink
  as capacity falls, which would confirm memorisation as the mechanism.
  Right now that mechanism is the best explanation, not something I tested.
