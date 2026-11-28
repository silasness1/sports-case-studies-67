# README: shared results, tables and plots for the group presentation
# - Loads the saved fits from football_combined_nested_tune.R, it never fits anything itself
# - Run from the project root (open sports-case-studies-67.Rproj), paths are relative to it
# - Before sourcing, football_combined_nested_tune.R must be set to only load:
#       every RERUN switch FALSE, QUICK = FALSE, and every results/combined/<wflow_id>.rds present
#   otherwise this script stops instead of starting a refit that overwrites results/combined/*.rds
# - To change a model or refit: edit and run football_combined_nested_tune.R directly, not from here
# - After sourcing you have its objects (see the README at its top): euro, folds, nested_res,
#       inner_scores, choices, fits, preds, wdl, scores, seed_scores, seed_summary (the default scores:
#       seed means) and finals (the final models)
#   and the expanding-window results as expanding$wdl, expanding$scores, ... (its Section 14)
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
SHOW_OBJECTS = FALSE
VIEW_OBJECTS = FALSE # TRUE = also open the flat tables in RStudio's viewer

if (SHOW_OBJECTS) {
  # an algorithm = workflow + selection target + decision rule, e.g. "base_rf | mae | range"
  # seed_summary: the default scores. One row per algorithm per metric: the mean, sd, min and max over
  #   the refit seeds (rf / xg / nnet: 6 seeds; the deterministic ones: their one value, sd 0)
  #   e.g. filter(seed_summary, metric == "accuracy")
  print("seed_summary object:")
  glimpse(seed_summary)

  # seed_scores: what seed_summary averages. One row per algorithm per seed (rf / xg / nnet only),
  #   every score in its own column. Seed 6769 is the main run
  print("seed_scores object:")
  glimpse(seed_scores)

  # scores: the main seed only (6769), one row per metric per algorithm, every algorithm
  #   e.g. filter(scores, metric == "macro_f1") or pivot_wider(names_from = metric, values_from = value)
  print("scores object:")
  glimpse(scores)

  # finals: the final models, fitted on all 6 tournaments (Section 15 of the nested tune file), by name
  #   each has $settings (the algorithm), $params, $t, $tuning_macro_f1 (optimistic) and $fit (a workflow)
  #   predict_final(new_data, finals$pois): one row per match with lambdas, probabilities and the call
  print("finals object:")
  for (name in names(finals)) {
    cat(name, ": ", paste(finals[[name]]$settings, collapse = " | "), ", t = ", finals[[name]]$t, "\n", sep = "")
    print(finals[[name]]$params)
  }

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
  #   metric: the mean over the 5 inner folds. Metrics: mae, poisson_log_loss, rps, f1_<rule>_<t>
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
  View(seed_summary)
  View(seed_scores)
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

# Section 0.4: Why nested leave one tournament out: the structure of the data----
# - tournament_table: one obs per {tournament}: its size (16 teams up to 2012, 24 from 2016), goals,
#     draw rate and the spread (sd) of two covariates. The covariates are team minus opponent, so their
#     mean is 0 in every tournament, but their scale changes (e.g. UefaPoints)
# - variance_levels: how the variance of a team's goals in a match splits into between tournaments,
#     between teams within a tournament, and within a team's matches (nested sums of squares)
#   tournaments barely differ in goals (about 1%), but a team's matches in one tournament are alike (about
#   25%): they share the team and its covariates. A random split of the rows would put some of a team's
#   matches in training and the rest in testing; holding out whole tournaments doesn't, and it's also the
#   real task (predict a new tournament, with its own format and covariate scale)
match_draws = tibble(Tournament = euro$Year[odd], draw = euro$Goals[odd] == euro$Goals[odd + 1]) # one per match
tournament_table = euro |>
  group_by(Tournament = Year) |>
  summarise(Matches          = n() / 2,
            Teams            = n_distinct(Team),
            `Goals per team` = mean(Goals),
            `SD MarketValue` = sd(MarketValue),
            `SD UefaPoints`  = sd(UefaPoints)) |>
  left_join(match_draws |> group_by(Tournament) |> summarise(`Draw rate` = mean(draw)), by = "Tournament") |>
  relocate(`Draw rate`, .after = `Goals per team`)
print(tournament_table)
save_table(tournament_table, "tournament_structure", digits = c(0, 0, 0, 2, 3, 2, 0))

goal_levels = anova(lm(Goals ~ factor(Year) / factor(Team), data = euro))
variance_levels = tibble(Level = c("between tournaments", "between teams within a tournament",
                                   "within a team's matches"),
                         `Share of goal variance` = goal_levels[["Sum Sq"]] / sum(goal_levels[["Sum Sq"]]),
                         p = c(goal_levels[["Pr(>F)"]][1:2], NA))
print(variance_levels)
save_table(variance_levels, "variance_levels", digits = c(0, 3, 3))

# Section 1: Headline: model class x decision rule x selection target----
# One tile per algorithm on base covariates. Facets = selection target, so one picture shows the best
# algorithm and whether the best rule depends on how the hyperparameters were picked
# ("none" = algorithms with nothing to tune: the null model and the historical frequency baseline)
# Every bar is the mean over the 6 refit seeds (seed_summary, from the nested tune file's Section 13)

target_order = c("none", "mae", "poisson_log_loss", "rps", "macro_f1")
model_order  = c("pois", "rf", "xg", "nnet", "null", "historical") # top to bottom in the plots
headline = seed_summary |>
  filter(covariates %in% c("base", "none")) |>
  mutate(target = factor(target, target_order),
         rule = factor(rule, c("argmax", "range", "p_draw")),
         model_class = factor(model_class, rev(model_order))) # ggplot draws the first level at the bottom

plot_headline = function(metric_name, label, digits = 3) {
  
  selected_data = headline |> 
    filter(metric == metric_name, target != "none") |>
    arrange(desc(mean)) |> 
    mutate(
      rank = row_number(),
      top10 = if_else(rank <= 10, factor(rank), NA)
    ) |>
    group_by(model_class, target) |>
    mutate(target_rule_winner = mean == max(mean, na.rm = TRUE)) |>
    ungroup()
  
  ggplot(selected_data, aes(x = rule, y = mean, fill = top10)) +
    geom_col() +
    geom_col(
      data = filter(selected_data, target_rule_winner),
      fill = NA,
      color = "red",
      linewidth = 1
    ) +
    geom_errorbar(
      aes(
        ymin = mean - 3*sd,
        ymax = mean + 3*sd
      ),
      width = 0.2,
      linewidth = 0.5
    ) +
    scale_fill_viridis_d(
      option = "C",
      direction = -1,
      na.value = "grey85",
      name = "Global top 10"
    ) +
    scale_y_continuous(
      expand = expansion(mult = c(0, 0.15))
    ) +
    facet_grid(
      model_class ~ target,
      labeller = label_both
    ) +
    labs(
      title = label,
      x = "Model",
      y = metric_name
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.background = element_rect(fill = "grey90", color = NA),
      strip.text = element_text(face = "bold"),
      legend.position = "right"
    )
}

# saved as headline_<metric>.png (save_plot only prints them when running interactively)
save_plot(plot_headline("macro_f1", "MacroF1"), "headline_macro_f1", width = 10, height = 8)
save_plot(plot_headline("draw_recall", "Draw Recall"), "headline_draw_recall", width = 10, height = 8)
save_plot(plot_headline("accuracy", "Accuracy"), "headline_accuracy", width = 10, height = 8)
save_plot(plot_headline("draw_f1", "Draw F1"), "headline_draw_f1", width = 10, height = 8)
save_plot(plot_headline("draw_precision", "Draw Precision"), # no bar: no draws called (argmax), so no precision
          "headline_draw_precision", width = 10, height = 8)
save_plot(plot_headline("draws_called", "Draws Called"),     # out of 246 matches, 72 of them draws
          "headline_draws_called", width = 10, height = 8)
save_plot(plot_headline("macruracy_mean", "MacroF1 + Accuracy Avg"), "headline_macruracy_mean", width = 10, height = 8)

# Appendix tables: every algorithm on base covariates, the seed means of the scores side by side
# - one table per model class (12 rows, one slide each), one for the two baselines, and one with
#     everything (for a written report)
# - Draws called: one decimal when it's a seed mean that isn't whole (rf / xg / nnet average 6 seeds)
# - in the slides: \rtable{appendix_pois} etc. (booktabs tabulars, no float, so they fit \rtable)
target_labels = c(none = "--", mae = "MAE", poisson_log_loss = "Poisson NLL", rps = "RPS",
                  macro_f1 = "macro F1 (joint)")
appendix_table = headline |>
  filter(metric %in% c("draws_called", "accuracy", "draw_recall", "draw_precision", "draw_f1", "macro_f1", "rps",
                       "brier", "mae", "rmse", "poisson_nll")) |>
  select(model_class, target, rule, metric, mean) |>
  pivot_wider(names_from = metric, values_from = mean) |>
  mutate(model_class = factor(model_class, model_order)) |>
  arrange(model_class, target, rule) |>
  transmute(Model            = as.character(model_class),
            Target           = target_labels[as.character(target)],
            Rule             = as.character(rule),
            `Draws called`   = if_else(draws_called == round(draws_called), sprintf("%.0f", draws_called),
                                       sprintf("%.1f", draws_called)), # a seed mean can be e.g. 0.2
            Accuracy         = accuracy,
            `Draw recall`    = draw_recall,
            `Draw precision` = draw_precision,
            `Draw F1`        = draw_f1,
            `Macro F1`       = macro_f1,
            RPS              = rps,
            Brier            = brier,
            MAE              = mae,          # goal-level scores: historical predicts no goals, so "--"
            RMSE             = rmse,
            `Poisson NLL`    = poisson_nll) |>
  # scores as text with 3 decimals, so every table prints them alike (kable would print a column of
  #   zeros as "0"). NA stays NA, printed as "--"
  mutate(across(Accuracy:`Poisson NLL`, \(x) if_else(is.na(x), NA, sprintf("%.3f", x))))

# save_appendix(df, name): text columns left-aligned, numbers right-aligned
save_appendix = function(df, name) {
  save_table(df, name, align = if_else(names(df) %in% c("Model", "Target", "Rule"), "l", "r"))
}
for (m in c("pois", "rf", "xg", "nnet")) {
  save_appendix(filter(appendix_table, Model == m) |> select(-Model), paste0("appendix_", m))
}
save_appendix(filter(appendix_table, Model %in% c("null", "historical")) |> select(-Target),
              "appendix_baselines")
save_appendix(appendix_table, "appendix_all")
print(appendix_table, n = Inf, width = Inf)

# Section 1.1: Pareto fronts: draw recall vs accuracy, and macro F1 vs accuracy (the model choice)----
# Every algorithm on base covariates (seed means), with the historical baseline for reference
# - a front: the algorithms no other algorithm beats on both of its two scores. There are two fronts:
#     draw recall and accuracy (the main one), and macro F1 and accuracy (what macro F1 alone would keep)
#     Choosing a point on a front is a preference (how much accuracy a caught draw is worth), decided as a
#     team, not a statistical test
# - draw_precision: of the draws called, the share that were draws (compare with the draw base rate)
# - the plots have equal axes (0.1 is as long on both), so a front's slope = accuracy lost per unit gained.
#     Front points are numbered by id, the same id as in that front's table
n_matches = n_test_rows / 2                                            # one call per match: 246
n_draws   = sum(distinct(wdl, Year, match_id, actual)$actual == "draw")

# on_front(x, y): TRUE for the points no other point beats on both x and y (higher = better on both)
on_front = function(x, y) map2_lgl(x, y, \(xi, yi) !any(x >= xi & y >= yi & (x > xi | y > yi)))

# front_data: one obs per {algorithm}, the seed means of the front's scores side by side
#   (draw_precision is NA for algorithms that call no draws; draw_f1 is 0 for them)
front_data = headline |>
  filter(metric %in% c("accuracy", "draw_recall", "draws_called", "draw_precision", "draw_f1", "macro_f1", "rps"),
         model_class != "null") |>
  select(algorithm, model_class, target, rule, metric, mean) |>
  pivot_wider(names_from = metric, values_from = mean)

# make_front(x): the front on score x and accuracy, one obs per algorithm on it, from fewest to most
#   draws called (id = its label in the plot)
make_front = function(x) {
  front_data |>
    filter(model_class != "historical") |>
    filter(on_front(.data[[x]], accuracy)) |>
    arrange(draws_called) |>
    mutate(id = row_number()) |>
    select(id, model_class, target, rule, draws_called, accuracy, draw_recall, draw_precision, draw_f1,
           macro_f1, rps)
}
front_draw_recall = make_front("draw_recall")
front_macro_f1    = make_front("macro_f1")
front_draw_f1     = make_front("draw_f1")
print(front_draw_recall, width = Inf)
print(front_macro_f1, width = Inf)
print(front_draw_f1, width = Inf)
cat("draw base rate:", round(n_draws / n_matches, 3), "\n")
save_table(front_draw_recall, "pareto_front")
save_table(front_macro_f1, "pareto_front_macro_f1")
save_table(front_draw_f1, "pareto_front_draw_f1")

# plot_front(front, x, color): accuracy against score x, coloured by score color (column names as strings)
#   front: the front on these axes (from make_front(x)), numbered by its id. Grey: every other algorithm
#   labels: the front's points are too close together to label in place, so the labels go in a staircase
#     below the curve (label_y), each joined to its point by a line. They end at their point's x
#     (hjust = 1), unless the point is near the left edge (then they start there)
#   lims: the range of both axes (the same on both, so 0.1 is as long on each). The label staircase is
#     spaced relative to it (zoom = how much smaller the range is than 0 to 0.6)
#   historical: FALSE leaves out the historical baseline (e.g. when zoomed in past it)
axis_names = c(draw_recall    = "draw recall (share of actual draws called a draw)",
               macro_f1       = "macro F1 (mean of the win / draw / loss F1s)",
               draw_f1        = "draw F1 (harmonic mean of draw precision and draw recall)")
color_names = c(draw_recall = "draw recall", macro_f1 = "macro F1", draw_f1 = "draw F1", draw_precision = "draw precision")
historical_point = filter(front_data, model_class == "historical")

plot_front = function(front, x, color, table_name, lims = c(0, 0.6), historical = TRUE) {
  zoom = diff(lims) / 0.6
  others = front_data |>
    filter(model_class != "historical") |>
    anti_join(front, by = c("model_class", "target", "rule"))
  labels = front |>
    mutate(label   = paste0(id, ": ", model_class, " | ", sub("poisson_log_loss", "NLL", target), " | ", rule,
                            " (", round(draws_called), " draws)"),
           label_x = .data[[x]],
           label_y = min(accuracy) - (0.05 + 0.04 * (id - 1)) * zoom,
           hjust   = if_else(label_x < lims[1] + 0.2 * zoom, 0, 1))
  ggplot(front, aes(.data[[x]], accuracy)) +
    geom_point(data = others, color = "grey75") +
    geom_line(color = "grey40") +
    geom_point(aes(color = .data[[color]]), size = 3) +
    geom_segment(data = labels, aes(xend = label_x, yend = label_y + 0.012 * zoom), color = "grey60",
                 linewidth = 0.3) +
    geom_text(data = labels, aes(x = label_x, y = label_y, label = label, hjust = hjust), size = 3) +
    geom_point(data = if (historical) historical_point else historical_point[0, ], shape = 4, size = 3) +
    scale_color_viridis_c(option = "C", name = color_names[[color]]) +
    coord_fixed(ratio = 1, xlim = lims, ylim = lims) +
    labs(x = axis_names[[x]],
         y = "accuracy (share of matches called right)",
         title = paste("Pareto front:", color_names[[x]], "vs accuracy"),
         subtitle = paste0("numbered: the front (id in the ", table_name, " table); grey: dominated algorithms;",
                           "\n", if (historical) "x: historical frequencies; ", "base covariates, seed means"))
}
save_plot(plot_front(front_draw_recall, "draw_recall", "macro_f1", "pareto_front"),
          "pareto_front", width = 8, height = 7)
save_plot(plot_front(front_macro_f1, "macro_f1", "draw_recall", "pareto_front_macro_f1", lims = c(0.35, 0.55),
                     historical = FALSE),
          "pareto_front_macro_f1", width = 8, height = 7)
save_plot(plot_front(front_draw_f1, "draw_f1", "draw_recall", "pareto_front_draw_f1"),
          "pareto_front_draw_f1", width = 8, height = 7)

# Draw precision vs draw recall: which matches are draws, not just how many
# - one point per algorithm that calls at least MIN_DRAWS draws (with fewer, precision rests on a
#     handful of calls, e.g. one lucky call = precision 1), coloured by accuracy
# - the line at the draw base rate: the precision of calling draws at random
# - numbered: the draw recall front (ids of the pareto_front table)
MIN_DRAWS = 10
pr_data = filter(front_data, model_class != "historical", draws_called >= MIN_DRAWS)
pr_front = semi_join(front_draw_recall, pr_data, by = c("model_class", "target", "rule"))
plot_draw_pr = ggplot(pr_data, aes(draw_recall, draw_precision)) +
  geom_hline(yintercept = n_draws / n_matches, linetype = "dashed", color = "grey40") +
  annotate("text", x = 0, y = n_draws / n_matches, label = "draw base rate (random draw calls)",
           hjust = 0, vjust = 1.6, size = 3) +
  geom_point(aes(color = accuracy), size = 2.5) +
  geom_text(data = pr_front, aes(label = id), vjust = -1, size = 3.5) +
  scale_color_viridis_c(option = "C", name = "accuracy") +
  coord_fixed(ratio = 1, xlim = c(0, 0.6), ylim = c(0, 0.6)) +
  labs(x = axis_names[["draw_recall"]],
       y = "draw precision (share of draws called that were draws)",
       title = "Draw precision vs draw recall",
       subtitle = paste0("algorithms calling at least ", MIN_DRAWS, " draws; numbered: the draw recall front ",
                         "(id in the pareto_front table)\nbase covariates, seed means"))
save_plot(plot_draw_pr, "draw_precision_recall", width = 8, height = 7)


# Section 1.2: Factor effects in the core grid----
# The core grid: model class (4) x target (4) x rule (3) = 48 algorithms on base covariates, LOTO, one
#   seed mean per algorithm (a cell). It's fully crossed, so a factor's effect can be read off matched
#   cells: algorithms that differ only in that factor
# Scores: accuracy, draw recall, draw precision, macro F1 and draws called. Draw precision only exists with
#   a threshold rule (argmax calls no draws), so it's left out wherever argmax is involved
# No significance tests: the cells share the same 246 matches. For scale, one algorithm's accuracy or
#   macro F1 has an SE of about 0.03 over those matches
effect_metrics = c("accuracy", "draw_recall", "draw_precision", "macro_f1", "draws_called")
effect_labels  = c(accuracy = "Accuracy", draw_recall = "Draw recall", draw_precision = "Draw precision",
                   macro_f1 = "Macro F1", draws_called = "Draws called")

# core_cells: one obs per {model class} x {target} x {rule} x {score}
core_cells = headline |>
  filter(target != "none", metric %in% effect_metrics) |>
  transmute(model_class = as.character(model_class), target = as.character(target),
            rule = as.character(rule), metric, value = mean) |>
  # argmax: a few seeds of rf / xg / nnet call 1 or 2 draws, which gives a meaningless precision
  mutate(value = if_else(metric == "draw_precision" & rule == "argmax", NA, value))
stopifnot(nrow(distinct(core_cells, model_class, target, rule)) == 48)

# save_effects(df, name): text columns left-aligned, the score columns right-aligned
save_effects = function(df, name) {
  save_table(df, name, align = if_else(names(df) %in% c("Term", "Model class", "Contrast"), "l", "r"))
}

# Variance decomposition: how much of the differences among the 48 cell means each factor and interaction
#   accounts for
# - total = the sum of squared deviations of the cell means from their overall mean. The grid is complete
#     and balanced, so it splits exactly into one part per factor, per two-way interaction, and a remainder
#     (the three-way interaction: each cell has one value, so there's no separate noise term)
# - share = a part / the total. The last row, the SD of the cell means, gives each score's scale: a big
#     share of a tiny spread is still a tiny effect
# - draw precision: the 32 cells with a threshold rule (its rule term = range vs p_draw only)
term_labels = c(model_class = "model class", target = "target", rule = "rule",
                `model_class:target` = "model class x target", `model_class:rule` = "model class x rule",
                `target:rule` = "target x rule", Residuals = "model class x target x rule (remainder)")
decompose = function(m) {
  cells = filter(core_cells, metric == m, m != "draw_precision" | rule != "argmax")
  parts = anova(lm(value ~ (model_class + target + rule)^2, data = cells))
  tibble(Term = unname(term_labels[rownames(parts)]), metric = m,
         share = parts[["Sum Sq"]] / sum(parts[["Sum Sq"]]), spread = sd(cells$value))
}
variance_parts = bind_rows(map(effect_metrics, decompose))
variance_table = bind_rows(
  variance_parts |>
    mutate(entry = sprintf("%.0f%%", 100 * share)) |>
    select(Term, metric, entry),
  variance_parts |>
    distinct(metric, spread) |>
    mutate(Term = "SD of the cell means", entry = sprintf(if_else(metric == "draws_called", "%.1f", "%.3f"), spread)) |>
    select(Term, metric, entry)) |>
  pivot_wider(names_from = metric, values_from = entry) |>
  rename(any_of(setNames(effect_metrics, effect_labels)))
print(variance_table, width = Inf)
save_effects(variance_table, "variance_decomposition")

# effects(factor, ref, by): the factor's levels against its reference level ref, within matched cells
#   (the same levels of the other factors), and with by: separately per level of by (an interaction:
#   does the factor's effect depend on by?)
#   each entry: the mean difference over the matched cells, and (k/n) = how many of the n were higher
#   (higher is better for every score except draws called, which only says how many)
effects = function(factor, ref, by = NULL) {
  others    = setdiff(c("model_class", "target", "rule"), factor)
  reference = core_cells |>
    filter(.data[[factor]] == ref) |>
    select(all_of(others), metric, ref_value = value)
  core_cells |>
    filter(.data[[factor]] != ref) |>
    inner_join(reference, by = c(others, "metric")) |>
    mutate(delta = value - ref_value) |>
    filter(!is.na(delta)) |>                    # draw precision against argmax
    group_by(across(all_of(c(by, factor))), metric) |>
    # round first (+ 0 turns -0 into 0), so a mean of e.g. -0.04 draws prints as +0.0, not -0.0
    summarise(entry = if (first(metric) == "draws_called") {
                sprintf("%+.1f (%d/%d)", round(mean(delta), 1) + 0, sum(delta > 0), n())
              } else {
                sprintf("%+.3f (%d/%d)", round(mean(delta), 3) + 0, sum(delta > 0), n())
              },
              .groups = "drop") |>
    mutate(Contrast = paste(.data[[factor]], "vs", ref)) |>
    pivot_wider(names_from = metric, values_from = entry) |>
    select(any_of(by), Contrast, any_of(effect_metrics)) |>
    rename(any_of(c(`Model class` = "model_class")), any_of(setNames(effect_metrics, effect_labels)))
}

# main effects: a threshold rule vs argmax (16 matched cells each), range vs p_draw (16, the one rule
#   contrast with a draw precision: argmax calls no draws, so it has none), each target vs mae (12 each),
#   each model class vs pois (12 each)
effects_main = bind_rows(effects("rule", "argmax"),
                         effects("rule", "p_draw") |> filter(Contrast == "range vs p_draw"),
                         effects("target", "mae"),
                         effects("model_class", "pois")) |>
  select(Contrast, any_of(unname(effect_labels))) # the same column order as the other tables
print(effects_main, width = Inf)
save_effects(effects_main, "effects_main")

# interactions: does the rule's effect depend on the model class (4 matched cells each)?
#   and does the target's effect depend on the model class (3 matched cells each)?
effects_rule_by_model   = effects("rule", "argmax", by = "model_class")
effects_target_by_model = effects("target", "mae", by = "model_class")
print(effects_rule_by_model, n = Inf, width = Inf)
print(effects_target_by_model, n = Inf, width = Inf)
save_effects(effects_rule_by_model, "effects_rule_by_model")
save_effects(effects_target_by_model, "effects_target_by_model")

# Draw precision at a fixed number of draws called: range vs p_draw on the same probabilities
# - with a two-stage target the three rules share the goal model's predictions, only the rule differs
# - each rule ranks the matches by how draw-like they look: p_draw by P(draw), range by how close the three
#     probabilities are (smallest max - min first). Calling the top k a draw = calling exactly k draws
# - precision at k: the share of those k matches that were draws. At the same k both rules call the same
#     number of draws, so a gap between the curves is about which matches they pick, not how many
# - pooled over the 6 held-out years (one ranking of all 246 matches), averaged over the seeds and the
#     three two-stage targets
draw_ks = seq(10, 150, by = 5)
precision_at_k = bind_rows(wdl, seed_wdl) |>
  filter(covariates == "base", target %in% c("mae", "poisson_log_loss", "rps"), rule == "argmax") |>
  mutate(range_score = pmax(p_win, p_draw, p_loss) - pmin(p_win, p_draw, p_loss)) |>
  group_by(model_class, target, seed) |>
  reframe(k      = draw_ks,
          p_draw = map_dbl(draw_ks, \(k) mean(actual[order(-p_draw)][1:k] == "draw")),
          range  = map_dbl(draw_ks, \(k) mean(actual[order(range_score)][1:k] == "draw"))) |>
  pivot_longer(c(p_draw, range), names_to = "rule", values_to = "precision") |>
  group_by(model_class, rule, k) |>
  summarise(precision = mean(precision), .groups = "drop")

# per model class: range minus p_draw precision, averaged over k = 25 to 125 draws called (the range the
#   tuned algorithms actually call), and at how many of those k p_draw was more precise
precision_gap = precision_at_k |>
  filter(between(k, 25, 125)) |>
  pivot_wider(names_from = rule, values_from = precision) |>
  group_by(`Model class` = model_class) |>
  summarise(`Range minus p_draw` = sprintf("%+.3f", mean(range - p_draw)),
            `p_draw more precise` = paste0(sum(p_draw > range), "/", n(), " k"))
print(precision_gap)
save_effects(precision_gap, "precision_range_vs_p_draw")

plot_precision_k = ggplot(precision_at_k, aes(k, precision, color = rule)) +
  geom_hline(yintercept = n_draws / n_matches, linetype = "dashed", color = "grey40") +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ model_class) +
  labs(x = "draws called (the k most draw-like matches)", y = "draw precision",
       title = "Draw precision at the same number of draws called: range vs p_draw",
       subtitle = paste("same goal-model probabilities for both rules (two-stage targets, seed means);",
                        "dashed: draw base rate"))
save_plot(plot_precision_k, "precision_at_k", width = 9, height = 6)

# Section 2: Paired comparisons: two algorithms on the same 246 matches (used in Section 3)----
# wdl_seeds: wdl plus the rf / xg / nnet picks refit with the 5 other seeds (Section 13 of the nested
#   tune file): one obs per {algorithm} x {seed} x {match}. pois, null and historical have one seed only
wdl_seeds = bind_rows(wdl, seed_wdl)

# compare_once(rows_a, rows_b): one paired comparison of two algorithms' calls (one seed each)
#   only matches both have are compared (so expanding vs LOTO is on 2012-2024 only)
#   delta_macro_f1 / delta_accuracy / delta_draw_recall: a minus b, pooled over all matches
#   years_better: in how many of the held-out years a had the higher macro F1
compare_once = function(rows_a, rows_b) {
  both = inner_join(rows_a |> select(Year, match_id, actual, call_a = call),
                    rows_b |> select(Year, match_id, call_b = call),
                    by = c("Year", "match_id"))
  per_year = both |>
    group_by(Year) |>
    summarise(f1_a = macro_f1(actual, call_a), f1_b = macro_f1(actual, call_b))
  draws = both$actual == "draw"
  tibble(delta_macro_f1    = macro_f1(both$actual, both$call_a) - macro_f1(both$actual, both$call_b),
         delta_accuracy    = mean(both$call_a == both$actual) - mean(both$call_b == both$actual),
         delta_draw_recall = mean(both$call_a[draws] == "draw") - mean(both$call_b[draws] == "draw"),
         years_better   = sum(per_year$f1_a > per_year$f1_b),
         n_years        = nrow(per_year))
}

# compare_algorithms(a, b): algorithm a (the change) against b (what it's compared to), over the seeds
#   wdl_a / wdl_b: where to find each one, e.g. expanding$wdl for the expanding-window results (Section 4)
#   each seed is one replicate: two random algorithms are compared at the same seed, and an algorithm
#   with one seed (pois, null, historical, everything in expanding$wdl) is reused at every seed
#   delta_* and years_better: the mean over the seeds, *_sd: their sd over the seeds (NA for one seed)
compare_algorithms = function(a, b, wdl_a = wdl_seeds, wdl_b = wdl_seeds) {
  rows_a = filter(wdl_a, algorithm == a)
  rows_b = filter(wdl_b, algorithm == b)
  has_seeds = function(rows) n_distinct(rows$seed) > 1
  seeds = if (has_seeds(rows_a) && has_seeds(rows_b)) intersect(unique(rows_a$seed), unique(rows_b$seed))
          else if (has_seeds(rows_a)) unique(rows_a$seed)
          else if (has_seeds(rows_b)) unique(rows_b$seed)
          else NA
  at_seed = function(rows, s) if (has_seeds(rows)) filter(rows, seed == s) else rows
  per_seed = map(seeds, \(s) compare_once(at_seed(rows_a, s), at_seed(rows_b, s))) |> bind_rows()
  tibble(a = a, b = b,
         n_seeds           = nrow(per_seed),
         delta_macro_f1    = mean(per_seed$delta_macro_f1),
         delta_macro_f1_sd = sd(per_seed$delta_macro_f1),
         delta_accuracy    = mean(per_seed$delta_accuracy),
         delta_accuracy_sd = sd(per_seed$delta_accuracy),
         delta_draw_recall = mean(per_seed$delta_draw_recall),
         years_better      = mean(per_seed$years_better),
         n_years           = per_seed$n_years[1])
}

# Section 3: The chosen model: pois | macro_f1 | p_draw (point 4 of the draw recall front)----
# - why it: the knee of the draw recall front (the most draws caught before draw precision falls), the best
#     macro F1 and draw F1, and deterministic (no seed noise)
# - its honest scores are its outer ones (each held-out year predicted by a model that never saw it). The
#     final model refit on all 6 tournaments (finals$pois, Section 15 of the nested tune file) is what
#     would predict 2028
CHOSEN     = "base_pois | macro_f1 | p_draw"
chosen_wdl = filter(wdl, algorithm == CHOSEN)

# algorithm_id(): the algorithm's name from its parts, as in preds / wdl ("base_pois | macro_f1 | p_draw")
algorithm_id = function(wflow_id, target, rule) paste(wflow_id, target, rule, sep = " | ")

# chosen_by_year: one obs per {held-out tournament}: how it did there, and the t and candidate it picked
chosen_by_year = chosen_wdl |>
  group_by(Year) |>
  summarise(Matches          = n(),
            `Actual draws`   = sum(actual == "draw"),
            `Draws called`   = sum(call == "draw"),
            Accuracy         = mean(call == actual),
            `Draw recall`    = sum(call == "draw" & actual == "draw") / sum(actual == "draw"),
            `Draw precision` = sum(call == "draw" & actual == "draw") / sum(call == "draw"),
            `Macro F1`       = macro_f1(actual, call)) |>
  left_join(choices |>
              filter(algorithm_id(wflow_id, target, rule) == CHOSEN) |>
              transmute(Year = as.numeric(sub("test ", "", id)), t, Hyperparameters = params_label),
            by = "Year")
print(chosen_by_year, width = Inf)
save_table(chosen_by_year, "chosen_by_year")

# chosen_confusion: rows = what it called, columns = what happened (all 246 held-out matches)
chosen_confusion = chosen_wdl |>
  count(Called = call, actual, .drop = FALSE) |>
  pivot_wider(names_from = actual, values_from = n, names_prefix = "Actual ")
print(chosen_confusion)
save_table(chosen_confusion, "chosen_confusion")

# chosen_vs_front: the chosen model against the other algorithms on the draw recall front and the
#   historical baseline, on the same matches (seed means for rf / xg / nnet, see compare_algorithms)
front_algorithms = with(front_draw_recall, algorithm_id(paste0("base_", model_class), target, rule))
chosen_vs_front = c(setdiff(front_algorithms, CHOSEN), "historical | none | argmax") |>
  map(\(b) compare_algorithms(CHOSEN, b)) |>
  bind_rows() |>
  transmute(`Compared with`    = b,
            `Delta accuracy`    = delta_accuracy,
            `Delta draw recall` = delta_draw_recall,
            `Delta macro F1`    = delta_macro_f1,
            `Years better`      = paste0(round(years_better, 1), "/", n_years))
print(chosen_vs_front, width = Inf)
save_table(chosen_vs_front, "chosen_vs_front")

# chosen_final_model: the final model's lasso coefficients (finals$pois). The predictors are standardized,
#   so a coefficient is the change in log expected goals per sd of that predictor, and its exp() the
#   factor expected goals are multiplied by. Draws are called when P(draw) > t
if (!is.null(finals$pois)) {
  chosen_final_model = tidy(extract_fit_parsnip(finals$pois$fit)) |>
    transmute(Term = term, Coefficient = estimate, `Goals multiplier per sd` = exp(estimate))
  print(chosen_final_model)
  cat("final model: penalty =", finals$pois$params$penalty, " mixture =", finals$pois$params$mixture,
      " rule: call a draw when P(draw) >", finals$pois$t, "\n")
  save_table(chosen_final_model, "chosen_final_model")
}

# Section 4: Sub-experiments: seed noise and the AgeFitDiff covariate----
# rf / xg / nnet are random: refit with another seed (Section 13 of the nested tune file, nothing
#   re-tuned), the same pick scores a little differently. pois and null are deterministic
# seed_noise: one obs per {model class}: over its algorithms with a threshold rule (argmax calls almost no
#   draws, so it has no draw precision to vary), the median seed sd and, in brackets, the median range
#   (max - min over the 6 seeds), from seed_summary. Compare with the differences between algorithms
#   (e.g. the effects tables): a gap smaller than the seed range can come from the seed alone
seed_noise = seed_summary |>
  filter(covariates == "base", model_class %in% c("rf", "xg", "nnet"), rule != "argmax",
         metric %in% effect_metrics) |>
  group_by(`Model class` = model_class, metric) |>
  summarise(entry = sprintf(if (first(metric) == "draws_called") "%.1f (%.1f)" else "%.3f (%.3f)",
                            median(sd, na.rm = TRUE), median(max - min, na.rm = TRUE)),
            .groups = "drop") |>
  pivot_wider(names_from = metric, values_from = entry) |>
  select(`Model class`, any_of(effect_metrics)) |>
  rename(any_of(setNames(effect_metrics, effect_labels)))
print(seed_noise, width = Inf)
save_effects(seed_noise, "seed_noise")

# AgeFitDiff: does adding the age covariate help? It only ran with target mae (Section 9 of the nested
#   tune file), so each cov algorithm is compared with the same model class and rule on base covariates
#   (seed means): 3 matched pairs per model class, 12 in all
# agefitdiff_effect: one obs per {model class} (+ all): the mean difference cov minus base, and (k/n) = in
#   how many pairs it went up. The scores of the effects tables (Section 1.2), plus MAE: a covariate
#   changes the goal model first, and MAE measures that directly. Up is better for every score except MAE
#   (lower = better) and draws called (only says how many). Draw precision: threshold rules only (argmax
#   calls ~no draws)
age_metrics = c(effect_metrics, "mae")
age_labels  = c(effect_labels, mae = "MAE")
age_pairs = seed_summary |>
  filter(target == "mae", model_class %in% c("pois", "rf", "xg", "nnet"), metric %in% age_metrics) |>
  mutate(mean = if_else(metric == "draw_precision" & rule == "argmax", NA, mean)) |>
  select(covariates, model_class, rule, metric, mean) |>
  pivot_wider(names_from = covariates, values_from = mean) |>
  mutate(delta = cov - base) |>
  filter(!is.na(delta))
agefitdiff_effect = bind_rows(age_pairs, mutate(age_pairs, model_class = "all")) |>
  group_by(`Model class` = model_class, metric) |>
  summarise(entry = if (first(metric) == "draws_called") {
              sprintf("%+.1f (%d/%d)", round(mean(delta), 1) + 0, sum(delta > 0), n())
            } else {
              sprintf("%+.3f (%d/%d)", round(mean(delta), 3) + 0, sum(delta > 0), n())
            },
            .groups = "drop") |>
  pivot_wider(names_from = metric, values_from = entry) |>
  arrange(factor(`Model class`, c(model_order, "all"))) |>
  select(`Model class`, any_of(age_metrics)) |>
  rename(any_of(setNames(age_metrics, age_labels)))
print(agefitdiff_effect, width = Inf)
save_effects(agefitdiff_effect, "agefitdiff_effect")


