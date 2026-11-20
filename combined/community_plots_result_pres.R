# README: shared results, tables and plots for the group presentation
# - Loads the saved fits from football_combined_nested_tune.R, it never fits anything itself
# - Run from the project root (open sports-case-studies-67.Rproj), paths are relative to it
# - Before sourcing, football_combined_nested_tune.R must be set to only load:
#       every RERUN switch FALSE, QUICK = FALSE, and every results/combined/<wflow_id>.rds present
#   otherwise this script stops instead of starting a refit that overwrites results/combined/*.rds
# - To change a model or refit: edit and run football_combined_nested_tune.R directly, not from here
# - After sourcing you have its objects (see the README at its top): euro, folds, nested_res,
#       inner_scores, choices, fits, preds, wdl, scores
#   and in Section 4 the expanding-window results as expanding$wdl, expanding$scores, ...
#   (needs results/combined_expanding, made by running it with RESAMPLING = "expanding")
#   e.g. filter(scores, model_class == "rf", rule == "range"). Section 0.2 shows what each looks like
# - Put shared plots/tables below, one section each. Personal results go in your own script
# - Save with save_plot(plot, "name") and save_table(df, "name") (Section 0.2), so everything lands in
#       results/combined/plots and results/combined/tables

# Section 0: Load the combined results----
NESTED_TUNE_FILE = "combined/football_combined_nested_tune.R"

# LOAD_ONLY = TRUE makes the nested tune file stop (before fitting anything) if it would refit
# finally = removed again afterwards (even on an error), so running the nested tune file yourself still refits
LOAD_ONLY = TRUE
tryCatch(source(NESTED_TUNE_FILE), finally = rm(LOAD_ONLY))

# Section 0.1: Additional libraries and helpers----
library(ggplot2)
library(stringr)

# collect_all(): run a nestedtune helper on every workflow's result and stack them into one flat table
#   with a wflow_id column, e.g. collect_all(collect_metrics), collect_all(agreement)
#   extra arguments go to the helper: collect_all(collect_metrics, summarize = FALSE)
collect_all = function(helper, ...) {
  nested_res |>
    mutate(out = map(result, \(r) helper(r, ...))) |>
    select(wflow_id, out) |>
    unnest(out)
}

# Section 0.2: What the objects look like----
# A quick tour so we all know how to interact with them. Run it line by line (Ctrl+Enter) the first time,
# set SHOW_OBJECTS = FALSE once you know them so sourcing stays quiet
# glimpse() = one line per column with its type and first values (shorter than str() for tibbles)
SHOW_OBJECTS = TRUE
VIEW_OBJECTS = FALSE # TRUE = also open the flat tables in RStudio's viewer

if (SHOW_OBJECTS) {
  # an algorithm = workflow + selection target + decision rule, e.g. "base_rf | mae | range"
  # scores: one row per metric per algorithm
  #   e.g. filter(scores, metric == "macro_f1") or pivot_wider(names_from = metric, values_from = value)
  print("scores object:")
  glimpse(scores)

  # wdl: one row per match per algorithm. p_win / p_draw / p_loss from Skellam, the actual result,
  #   and the call under the algorithm's rule at its t (t is picked per outer year)
  print("wdl object:")
  glimpse(wdl)

  # preds: one row per held-out team-game per algorithm. Goals vs the predicted lambda (.pred)
  print("preds object:")
  glimpse(preds)

  # choices: one row per workflow x outer year x target x rule. Which candidate (params_label) and t
  #   each algorithm picked, and its inner macro F1 (inner_f1)
  print("choices object:")
  glimpse(choices)

  # inner_scores: what the choices were made from. One row per workflow x outer year x candidate x
  #   metric: the mean over the 5 inner folds. Metrics: mae, poisson_log_loss, f1_<rule>_<t>
  print("inner_scores object:")
  glimpse(inner_scores)

  # fits: one row per distinct pick (workflow x outer year x candidate), its outer predictions in pred
  #   preds is choices joined to fits, so algorithms that share a pick share its predictions
  print("fits object:")
  glimpse(fits)

  # nested_res: one row per workflow, the nestedtune result in a list column
  print(nested_res)
  #   each result has one row per outer year: splits, .metrics, .selected, .inner_metrics, ...
  #   (outer predictions aren't saved here: preds / fits hold them, refit for every algorithm)
  print(nested_res$result[[which(nested_res$wflow_id == "base_pois")]])
  #   summary() says what the run means: failed folds, what each year selected, the estimate
  print(summary(nested_res$result[[which(nested_res$wflow_id == "base_pois")]]))
  #   all workflows at once: collect_all(<helper>) gives one flat table with a wflow_id column
  print(collect_all(collect_metrics))
  #   Helpers for one result r (e.g. r = nested_res$result[[2]]), all work in collect_all() too:
  #   nestedtune (methods for a nested result):
  #     summary(r)                  what the run means: failed folds, selections, the estimate
  #     collect_metrics(r)          the nested estimate: mean and std_err across the outer years
  #                                 summarize = FALSE: one row per outer year, type = "wide": a column per metric
  #     collect_predictions(r), augment(r), compute_metrics(r, ...)  need saved outer predictions,
  #                                 which we don't keep (save_pred = FALSE): use preds / fits instead
  #     collect_selections(r)       hyperparameters each outer year picked (= target "mae" in choices)
  #     agreement(r)                how often each hyperparameter setting was picked across the outer years
  #     collect_inner_metrics(r)    every candidate's inner-CV score in every outer year (type = "wide" too)
  #     collect_notes(r)            warnings / errors from fitting
  #     collect_extracts(r)         anything saved with control_grid(extract = ...) (we don't save any)
  #     autoplot(r, type = "parameters" or "performance")  plot the selections or the per-year scores
  #     extract_procedure(r)        the settings the run used: $tuner, $grid, $select, $first_metric, $workflow, ...
  #   Final model (one fit on all 6 tournaments, for predicting new data):
  #     f = nested_final_fit(...)   predict(f, new_data), extract_tune_results(f), extract_scored_candidates(f)
  #     predict(r) doesn't work on a nested result: it has no single model, only one per outer year
  #   tune (on a workflow or final fit, not on r itself):
  #     extract_workflow(), extract_fit_parsnip(), extract_fit_engine() (the raw glmnet / ranger / ... object),
  #     extract_recipe(), extract_spec_parsnip(), extract_parameter_set_dials()
  #     select_best(), show_best() work on the inner tune results, e.g. from extract_tune_results(f)

  # folds: one row per outer year, each with its own 5 inner folds (inner_resamples)
  print(folds)
  print(folds$inner_resamples[[1]])
  #   the rows of a split: analysis() = training years, assessment() = the held-out year
  print(count(analysis(folds$splits[[1]]), Year))
  print(count(assessment(folds$splits[[1]]), Year))
}

if (VIEW_OBJECTS && interactive()) {
  View(scores)
  View(wdl)
  View(preds)
  View(choices)
  View(fits)
}

# Section 0.3: Saving plots and tables----
# plots go to results/combined/plots/<name>.png, tables to results/combined/tables/<name>.tex
# (the presentation can pull a table in with \input{.../tables/<name>.tex}, so it updates with the results)
PLOT_DIR  = "results/combined/plots"
TABLE_DIR = "results/combined/tables"
dir.create(PLOT_DIR,  showWarnings = FALSE, recursive = TRUE)
dir.create(TABLE_DIR, showWarnings = FALSE, recursive = TRUE)

# save a plot to PLOT_DIR/<name>.png, and show it when running interactively (RStudio)
# (printing under Rscript would dump every plot into a stray Rplots.pdf instead)
save_plot = function(plot, name, width = 8, height = 5) {
  if (interactive()) print(plot)
  ggsave(file.path(PLOT_DIR, paste0(name, ".png")), plot, width = width, height = height, bg = "white")
  invisible(plot)
}

options(knitr.kable.NA = "--")

# save a data frame as a LaTeX (booktabs) table to TABLE_DIR/<name>.tex, ... goes to knitr::kable
#   e.g. save_table(scores, "scores", col.names = c(...), escape = FALSE)
save_table = function(df, name, digits = 3, ...) {
  tab = knitr::kable(df, format = "latex", booktabs = TRUE, digits = digits, linesep = "", ...)
  writeLines(tab, file.path(TABLE_DIR, paste0(name, ".tex")))
  invisible(tab)
}

# Section 1: Headline: model class x decision rule x selection target----
# One tile per algorithm on base covariates. Facets = selection target, so one picture shows the best
# algorithm and whether the best rule depends on how the hyperparameters were picked
# ("none" = algorithms with nothing to tune: the null model and the historical frequency baseline)
target_order = c("none", "mae", "poisson_log_loss", "macro_f1")
headline = scores |>
  filter(covariates %in% c("base", "none")) |>
  mutate(target = factor(target, target_order),
         rule = factor(rule, c("argmax", "range", "p_draw")))

plot_headline = function(metric_name, label, digits = 3) {
  ggplot(filter(headline, metric == metric_name), aes(rule, model_class, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = format(round(value, digits), nsmall = digits)), size = 3.5) +
    facet_grid(~ target, scales = "free_x", space = "free_x") +
    # begin = 0.35 skips the darkest colors, so the black numbers stay readable on every tile
    scale_fill_viridis_c(option = "magma", begin = 0.35, direction = if (metric_name == "macro_f1") 1 else -1) +
    labs(x = "decision rule", y = NULL, fill = label,
         title = paste(label, "by model class, decision rule and selection target"))
}

save_plot(plot_headline("macro_f1", "Macro F1"), "headline_macro_f1", width = 10, height = 4)
# RPS doesn't depend on the rule (it only uses the probabilities), but it does on the target,
# because each target can pick different hyperparameters
save_plot(plot_headline("rps", "RPS"), "headline_rps", width = 10, height = 4)

# Section 2: Paired effects: does a change help on the same 246 matches?----
# compare_algorithms(a, b): algorithm a (the change) against b (what it's compared to)
#   wdl_a / wdl_b: where to find each one, e.g. expanding$wdl for the expanding-window results (Section 4)
#   only matches both have are compared (so expanding vs LOTO is on 2012-2024 only)
#   delta_macro_f1 / delta_accuracy: a minus b, pooled over all matches
#   years_better: in how many of the 6 held-out years a had the higher macro F1
#   mcnemar_p: McNemar's test on right / wrong per match (it tests accuracy, not macro F1)
compare_algorithms = function(a, b, wdl_a = wdl, wdl_b = wdl) {
  both = inner_join(filter(wdl_a, algorithm == a) |> select(Year, match_id, actual, call_a = call),
                    filter(wdl_b, algorithm == b) |> select(Year, match_id, call_b = call),
                    by = c("Year", "match_id"))
  per_year = both |>
    group_by(Year) |>
    summarise(f1_a = macro_f1(actual, call_a), f1_b = macro_f1(actual, call_b))
  right_a = factor(both$call_a == both$actual, c(FALSE, TRUE))
  right_b = factor(both$call_b == both$actual, c(FALSE, TRUE))
  tibble(a = a, b = b,
         delta_macro_f1 = macro_f1(both$actual, both$call_a) - macro_f1(both$actual, both$call_b),
         delta_accuracy = mean(right_a == "TRUE") - mean(right_b == "TRUE"),
         years_better   = sum(per_year$f1_a > per_year$f1_b),
         n_years        = nrow(per_year),
         mcnemar_p      = if (all(right_a == right_b)) 1 else mcnemar.test(table(right_a, right_b))$p.value)
}

# the algorithms on base covariates, one row each
base_algorithms = scores |>
  filter(covariates == "base", model_class != "null") |>
  distinct(algorithm, wflow_id, target, rule)

# threshold effect: rule vs argmax with the same goal model (two-stage targets, same predictions)
threshold_pairs = base_algorithms |>
  filter(target %in% c("mae", "poisson_log_loss"), rule != "argmax") |>
  mutate(b = paste(wflow_id, target, "argmax", sep = " | "))

# selection effect: joint (macro F1) vs two-stage (mae), same rule
selection_pairs = base_algorithms |>
  filter(target == "macro_f1") |>
  mutate(b = paste(wflow_id, "mae", rule, sep = " | "))

# covariate effect: with vs without AgeFitDiff, target mae, same rule
covariate_pairs = scores |>
  filter(covariates == "cov") |>
  distinct(algorithm, wflow_id, rule) |>
  mutate(b = sub("^cov_", "base_", algorithm))

paired_effects = bind_rows(threshold = threshold_pairs, selection = selection_pairs,
                           covariates = covariate_pairs, .id = "effect") |>
  mutate(result = map2(algorithm, b, compare_algorithms)) |>
  select(effect, result) |>
  unnest(result)
print(paired_effects, n = Inf)
save_table(paired_effects, "paired_effects")

# Section 3: Tables----
# main table: one row per algorithm on base covariates, the scores side by side
main_table = scores |>
  filter(covariates %in% c("base", "none"),
         metric %in% c("macro_f1", "accuracy", "draw_recall", "draws_called", "rps", "log_loss", "mae")) |>
  select(model_class, target, rule, metric, value) |>
  pivot_wider(names_from = metric, values_from = value) |>
  arrange(desc(macro_f1))
print(main_table, n = Inf)
save_table(main_table, "main_table")

# t picked for each held-out year, per algorithm with a threshold (base covariates)
choices |>
  filter(covariates == "base", rule != "argmax") |>
  mutate(Year = sub("test ", "", id)) |>
  select(wflow_id, target, rule, Year, t) |>
  pivot_wider(names_from = Year, values_from = t) |>
  print(n = Inf)

# confusion matrix of one algorithm (rows = called, columns = actual)
with(filter(wdl, algorithm == "base_pois | mae | range"), table(called = call, actual = actual))

# Section 4: Sub-experiments: is the reference choice justified, does the headline survive?----
# Each sub-result changes ONE thing from the reference slice (LOTO, base covariates, target mae, every
# model x rule) and asks: does the winner (model, rule) change, and how much does the score move?

# The expanding-window results, loaded into their own environment so their objects (scores, wdl, ...)
#   don't overwrite the LOTO ones: use expanding$wdl, expanding$scores, ...
#   (they come from running football_combined_nested_tune.R with RESAMPLING = "expanding")
expanding = new.env()
local({
  RESAMPLING = "expanding"
  LOAD_ONLY  = TRUE
  source(NESTED_TUNE_FILE, local = TRUE)
}, envir = expanding)
expanding_years = unique(expanding$wdl$Year) # 2012-2024: the years both schemes test on

# slice_scores(): macro F1 and RPS of every model x rule at target mae, on some covariates and years
#   one row per (model_class, rule)
slice_scores = function(wdl, covariates_used, years) {
  wdl |>
    filter(covariates == covariates_used, target == "mae", Year %in% years) |>
    group_by(model_class, rule, algorithm) |>
    summarise(macro_f1 = macro_f1(actual, call),
              rps      = mean(((p_win - (actual == "win"))^2 + (p_win + p_draw - (actual != "loss"))^2) / 2),
              .groups  = "drop")
}

# the slices: the reference on all 6 years and on 2012-2024 (to compare with expanding), and the changes
reference_all       = slice_scores(wdl, "base", unique(wdl$Year))
reference_2012_2024 = slice_scores(wdl, "base", expanding_years)
with_age            = slice_scores(wdl, "cov", unique(wdl$Year))
expanding_window    = slice_scores(expanding$wdl, "base", expanding_years)

# robustness_row(): one row of the robustness table
#   winner = best (model, rule) on macro F1. The paired columns compare the reference winner's
#   (model, rule) with and without the change, on the same matches
robustness_row = function(sub_result, reference, changed, wdl_changed = wdl) {
  w_ref = slice_max(reference, macro_f1, n = 1, with_ties = FALSE)
  w_new = slice_max(changed, macro_f1, n = 1, with_ties = FALSE)
  same  = filter(changed, model_class == w_ref$model_class, rule == w_ref$rule) # the reference winner, changed
  paired = compare_algorithms(same$algorithm, w_ref$algorithm, wdl_a = wdl_changed)
  if (identical(reference, changed)) paired[] = NA # the reference row: nothing to compare with
  tibble(sub_result,
         reference_winner = paste(w_ref$model_class, w_ref$rule),
         winner           = paste(w_new$model_class, w_new$rule),
         same_winner      = reference_winner == winner,
         winner_macro_f1  = w_new$macro_f1,
         winner_rps       = w_new$rps,
         delta_macro_f1   = w_new$macro_f1 - w_ref$macro_f1,  # new winner vs reference winner
         ref_winner_delta = paired$delta_macro_f1,              # reference winner, changed vs not
         years_better     = if (is.na(paired$n_years)) NA else paste0(paired$years_better, "/", paired$n_years),
         mcnemar_p        = paired$mcnemar_p)
}

# seed noise: the reference slice again, with rf / xg / nnet refit with other seeds (Section 13 of the
#   nested tune file). pois doesn't depend on the seed, so its scores are the same at every seed
#   seed_reference: one row per (model, rule) per seed. seed_winners: the best (model, rule) per seed
all_seeds = unique(seed_scores$seed)
seed_reference = bind_rows(
  reference_all |> filter(model_class == "pois") |> crossing(seed = all_seeds),
  seed_scores |> filter(covariates == "base", target == "mae") |> select(model_class, rule, algorithm, seed, macro_f1, rps))
seed_winners = seed_reference |>
  group_by(seed) |>
  slice_max(macro_f1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(winner = paste(model_class, rule))
seed_row = local({
  w_ref = slice_max(reference_all, macro_f1, n = 1, with_ties = FALSE)
  other = filter(seed_winners, seed != 6769) # the 5 other seeds
  tibble(sub_result       = "refit seed (5 other seeds, nothing re-tuned)",
         reference_winner = paste(w_ref$model_class, w_ref$rule),
         winner           = paste0(names(table(other$winner)), " (", table(other$winner), "/", nrow(other), ")", collapse = ", "),
         same_winner      = all(other$winner == reference_winner),
         winner_macro_f1  = mean(other$macro_f1),            # averaged over the seeds
         winner_rps       = mean(other$rps),
         delta_macro_f1   = mean(other$macro_f1) - w_ref$macro_f1)
})

robustness = bind_rows(
  robustness_row("reference: LOTO, base covariates, target mae", reference_all, reference_all),
  seed_row,
  robustness_row("+ AgeFitDiff", reference_all, with_age),
  robustness_row("expanding window (vs LOTO on 2012-2024)", reference_2012_2024, expanding_window,
                 wdl_changed = expanding$wdl))
print(robustness, width = Inf)
save_table(robustness, "robustness")

# backup slide: every model x rule, expanding vs LOTO on the same matches (2012-2024)
resampling_effects = expanding_window |>
  select(algorithm) |>
  mutate(result = map(algorithm, \(a) compare_algorithms(a, a, wdl_a = expanding$wdl))) |>
  select(result) |>
  unnest(result) |>
  left_join(select(reference_2012_2024, algorithm, macro_f1_loto = macro_f1), by = c("b" = "algorithm")) |>
  left_join(select(expanding_window, algorithm, macro_f1_expanding = macro_f1), by = c("a" = "algorithm"))
print(resampling_effects, width = Inf)
save_table(resampling_effects, "resampling_effects")

# backup slide: how much each algorithm's macro F1 moves from the refit seed alone (6 seeds)
#   pois and null aren't here: they're deterministic, so the seed doesn't move them at all
seed_noise = seed_scores |>
  group_by(algorithm, covariates, model_class, target, rule) |>
  summarise(sd_macro_f1    = sd(macro_f1),
            min_macro_f1   = min(macro_f1),
            max_macro_f1   = max(macro_f1),
            range_macro_f1 = max_macro_f1 - min_macro_f1,
            .groups = "drop") |>
  arrange(desc(range_macro_f1))
print(seed_noise, n = Inf)
save_table(seed_noise, "seed_noise")

