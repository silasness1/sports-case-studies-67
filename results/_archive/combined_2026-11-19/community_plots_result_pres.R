# README: shared results, tables and plots for the group presentation
# - Loads the saved fits from football_combined_nested_tune.R, it never fits anything itself
# - Run from the project root (open sports-case-studies-67.Rproj), paths are relative to it
# - Before sourcing, football_combined_nested_tune.R must be set to only load:
#       every RERUN switch FALSE, QUICK = FALSE, and every results/combined/<wflow_id>.rds present
#   otherwise this script stops instead of starting a refit that overwrites results/combined/*.rds
# - To change a model or refit: edit and run football_combined_nested_tune.R directly, not from here
# - After sourcing you have its objects (see the README at its top): euro, folds, nested_res,
#       selections, preds, wdl, scores
#   e.g. filter(scores, model_class == "rf", covariates == "cov"). Section 0.1 shows what each looks like
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
  # scores: one row per metric per workflow (wflow_id = <covariates>_<model_class>)
  #   e.g. filter(scores, metric == "rps") or pivot_wider(names_from = wflow_id, values_from = value)
  print("scores object:")
  glimpse(scores)

  # wdl: one row per match per workflow. p_win / p_draw / p_loss from Skellam, actual result,
  #   and the predicted result for argmax / the tuned draw thresholds (pred_t_draw, pred_t_range)
  print("wdl object:")
  glimpse(wdl)

  # preds: one row per held-out team-game per workflow. Goals vs the predicted lambda (.pred)
  print("preds object:")
  glimpse(preds)

  # selections: hyperparameters each outer year (held-out tournament) chose, per workflow
  print("selections object")
  glimpse(selections)

  # nested_res: one row per workflow, the nestedtune result in a list column
  print(nested_res)
  #   each result has one row per outer year: splits, .metrics, .selected, .predictions, ...
  print(nested_res$result[[which(nested_res$wflow_id == "base_pois")]])
  #   summary() says what the run means: failed folds, what each year selected, the estimate
  print(summary(nested_res$result[[which(nested_res$wflow_id == "base_pois")]]))
  #   all workflows at once: collect_all(<helper>) gives one flat table with a wflow_id column
  print(collect_all(collect_metrics))
  print(collect_all(collect_predictions))
  #   Helpers for one result r (e.g. r = nested_res$result[[2]]), all work in collect_all() too:
  #   nestedtune (methods for a nested result):
  #     summary(r)                  what the run means: failed folds, selections, the estimate
  #     collect_metrics(r)          the nested estimate: mean and std_err across the outer years
  #                                 summarize = FALSE: one row per outer year, type = "wide": a column per metric
  #     collect_predictions(r)      held-out predictions, one row per team-game (.row = row in euro)
  #     augment(r)                  the same predictions joined onto the data columns
  #     collect_selections(r)       hyperparameters each outer year picked (what `selections` is built from)
  #     agreement(r)                how often each hyperparameter setting was picked across the outer years
  #     collect_inner_metrics(r)    every candidate's inner-CV score in every outer year (type = "wide" too)
  #     collect_notes(r)            warnings / errors from fitting
  #     collect_extracts(r)         anything saved with control_grid(extract = ...) (we don't save any)
  #     compute_metrics(r, metric_set(...))  rescore the saved predictions with other metrics, no refit
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
  View(selections)
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

# Section 1: Plots----
# e.g. save_plot(ggplot(scores, aes(...)) + geom_point(), "model_comparison")

# Section 2: Tables----
# one column per workflow, lower is better except accuracy and macro F1
scores |>
  select(wflow_id, metric, value) |>
  pivot_wider(names_from = wflow_id, values_from = value) |>
  print(width = Inf)

# draw thresholds chosen for each held-out year
wdl |> distinct(wflow_id, Year, overwrite_t_range) |> pivot_wider(names_from = Year, values_from = overwrite_t_range)
wdl |> distinct(wflow_id, Year, overwrite_t_draw) |> pivot_wider(names_from = Year, values_from = overwrite_t_draw)

# confusion matrix of one workflow (rows = predicted, columns = actual)
with(filter(wdl, wflow_id == "base_pois"), table(predicted = pred_t_draw, actual = actual))
with(filter(wdl, wflow_id == "base_pois"), table(predicted = pred_t_range, actual = actual))

# results for threshold method comparison
save_table(scores |>
             select(wflow_id, metric, value) |>
             pivot_wider(names_from = wflow_id, values_from = value) |>
             filter(metric %in% c("draw_recall_argmax", "draw_recall_t_draw", "draw_recall_t_range")) |>
             select(-starts_with("cov")),
           "draw_recall_by_threshold_method", digits = 2)

save_table(wdl |>
             distinct(wflow_id, Year, overwrite_t_range, overwrite_t_draw) |>
             pivot_wider(names_from = Year, values_from = c(overwrite_t_range, overwrite_t_draw)) |>
             rowwise() |>
             summarise(wflow_id = wflow_id,
                       mean_t_range = mean(c_across(starts_with("overwrite_t_range"))),
                       sd_t_range =   sd(c_across(starts_with("overwrite_t_range"))),
                       mean_t_draw = mean(c_across(starts_with("overwrite_t_draw"))),
                       sd_t_draw   =   sd(c_across(starts_with("overwrite_t_draw")))) |>
             filter(!str_detect(wflow_id, "cov_")),
           "draw_threshold_mean_sd")

# explain why rf did better with p(draw)
save_table(wdl |>
             distinct(wflow_id, Year, overwrite_t_range) |>
             pivot_wider(names_from = Year, values_from = c(overwrite_t_range)) |>
             filter(wflow_id %in% c("base_xg", "base_nnet")),
           "t_range_by_year_xg_nnet")
