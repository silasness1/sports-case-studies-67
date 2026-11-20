# README: shared results, tables and plots for the group presentation
# - Loads the saved fits from football_combined_nested_tune.R, it never fits anything itself
# - Run from the project root (open sports-case-studies-67.Rproj), paths are relative to it
# - Before sourcing, football_combined_nested_tune.R must be set to only load:
#       every RERUN switch FALSE, QUICK = FALSE, and every results/combined/<wflow_id>.rds present
#   otherwise this script stops instead of starting a refit that overwrites results/combined/*.rds
# - To change a model or refit: edit and run football_combined_nested_tune.R directly, not from here
# - After sourcing you have its objects (see the README at its top): euro, folds, nested_res,
#       inner_scores, choices, fits, preds, wdl, scores
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
#   delta_macro_f1 / delta_accuracy: a minus b, pooled over all matches
#   years_better: in how many of the 6 held-out years a had the higher macro F1
#   mcnemar_p: McNemar's test on right / wrong per match (it tests accuracy, not macro F1)
compare_algorithms = function(a, b) {
  both = inner_join(filter(wdl, algorithm == a) |> select(Year, match_id, actual, call_a = call),
                    filter(wdl, algorithm == b) |> select(Year, match_id, call_b = call),
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
