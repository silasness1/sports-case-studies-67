# README: every model class (poisson, random forest, xgboost, neural net), each with and without the
# AgeDifference and _____ covariates, run through ONE nested leave-one-tournament-out design
#
# Important intermediates
#       euro
#       folds
#       predictor_sets
#       wf_sets
#       selections
#       
# Results live in tables with one column per choice, so use filters to get specific:
#   nested_res   one row per workflow (wflow_id = <covariates>_<model_class>), the nestedtune results
#   preds        one row per team-game per workflow: Goals and the predicted lambda (.pred)
#   wdl          one row per match per workflow: Skellam win/draw/loss probabilities + draw-threshold call
#   scores       one row per metric per workflow
# e.g. filter(scores, model_class == "rf", covariates == "cov")

# Section 0: Settings----
# Which workflows to rerun. TRUE = run its nested tuning and overwrite <RESULTS_DIR>/<wflow_id>.rds,
# FALSE = load that file. A workflow with no saved file runs anyway
# Set TRUE after changing anything a workflow depends on (its grid, model, predictors, data or folds):
# a FALSE workflow loads its old result without checking
RERUN = c(
  base_null = FALSE,
  base_pois = FALSE, base_rf = FALSE, base_xg = FALSE, base_nnet = FALSE,
  cov_pois  = FALSE, cov_rf  = FALSE, cov_xg  = FALSE, cov_nnet  = FALSE
)
QUICK = FALSE  # TRUE = 3 candidates per grid, for checking the script runs (saved separately)
RESULTS_DIR = if (QUICK) "results_debug/combined" else "results/combined"

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(rsample)
library(recipes)
library(parsnip)
library(poissonreg)
library(workflows)
library(workflowsets)
library(tune)
library(dials)
library(yardstick)
library(nestedtune)
library(skellam)

# Section 1: Data----
# combine existing: EuroTest = 2024, EuroTraining = 2004-2020
euro_test  = read_csv("raw-data/EuroTest.csv")
euro_train = read_csv("raw-data/EuroTraining.csv")
euro = rbind(euro_test, euro_train)

# Age covariate: Eurodata_with_Age.csv has the same rows in the same order, plus the age columns
# Only AgeDifference (team age - opponent age) is used: it's a difference like the other predictors
ages = read_csv("raw-data/Eurodata_with_Age_New.csv")
stopifnot(all(ages$Team == euro$Team), all(ages$Opponent == euro$Opponent), all(ages$Year == euro$Year))
euro$AgeFitDiff = ages$AgeFitDiff

# 2024 fix: the original data has GroupStage = 0 for every 2024 game, but the first 36 matches
rows_2024 = which(euro$Year == 2024)
euro$GroupStage[rows_2024[1:72]] = 1

# match_id: the two rows of a match are next to each other (team's view, then opponent's view)
euro$match_id = rep(1:(nrow(euro) / 2), each = 2)
odd = seq(1, nrow(euro), by = 2)
stopifnot(all(euro$Team[odd] == euro$Opponent[odd + 1])) # checks if Team matches Opponent of nextline

# Section 2: Nested leave-one-tournament-out splits----
# outer: hold out one tournament (honest test), inner: hold out one of the other five (tuning)
set.seed(67676769)
folds = nested_resamples(euro,
                         outside = group_vfold_cv(group = Year),
                         inside  = group_vfold_cv(group = Year))

# Name each fold by its held-out year instead of Resample1, Resample2, ... (which aren't in year order)
#   outer: "test 2016", inner: "test 2016 / eval 2004" (2016 is the outer test year, 2004 the inner one)
held_out_year = function(split) unique(assessment(split)$Year)
rename_folds = function(rset, ids) {
  rset$id = ids
  for (i in seq_along(rset$splits)) {
    rset$splits[[i]]$id$id = ids[i] # the split keeps its own copy of the id
  }
  return (rset)
}
test_years = map_dbl(folds$splits, held_out_year)
folds = rename_folds(folds, paste("test", test_years))
for (i in seq_along(folds$inner_resamples)) {      #renaming inner folds
  eval_years = map_dbl(folds$inner_resamples[[i]]$splits, held_out_year)
  folds$inner_resamples[[i]] = rename_folds(folds$inner_resamples[[i]],
                                            paste0("test ", test_years[i], " / eval ", eval_years))
}

# Weight each fold by its number of held-out rows, so averaging the 5 inner folds' mae equals the mae
# pooled over all their rows (otherwise a 62-row year counts as much as a 102-row year)
assessment_weights = function(rset) {
  n = map_int(rset$splits, \(split) nrow(assessment(split)))
  n / sum(n)
}
folds = tune::add_resample_weights(folds, assessment_weights(folds))
for (i in seq_along(folds$inner_resamples)) {
  folds$inner_resamples[[i]] = add_resample_weights(folds$inner_resamples[[i]],
                                                    assessment_weights(folds$inner_resamples[[i]]))
}

# Section 3: Preprocessing (the covariate choice)----
# base = the original predictors, cov = base + the new covariates (add new ones to new_covariates)
base_predictors = c("GroupStage", "GDP", "MarketValue", "FifaRank", "UefaPoints", "CLPlayers")
new_covariates  = c("AgeFitDiff")
predictor_sets  = list(base = base_predictors,
                       cov  = c(base_predictors, new_covariates))

# Predictors are standardized with the training rows' means and sds: the neural net needs it
# (UefaPoints is in the 10,000s, GDP around 1), and it doesn't change trees or glmnet's fit
make_recipe = function(predictors) {
  recipe(reformulate(predictors, response = "Goals"), data = euro) |>
    step_normalize(all_numeric_predictors())
}

# Section 4: Models (the model class choice)----
# null: no predictors, predicts the training mean for every row (the baseline)
null_spec = parsnip::null_model(mode = "regression") |> set_engine("parsnip")

# pois: lasso / ridge / elastic-net Poisson regression
pois_spec = parsnip::poisson_reg(penalty = tune(), 
                                 mixture = tune()) |> set_engine("glmnet")

# rf: ranger random forest. sample.fraction = share of rows each tree is grown on
rf_spec = parsnip::rand_forest(mtry = tune(), 
                               min_n = tune(), 
                               trees = 1000) |>
  set_engine("ranger", 
             sample.fraction = tune(), 
             num.threads = 1) |>
  set_mode("regression")

# xg: xgboost with a Poisson objective, so predictions are expected goals (> 0)
#    min_n = xgboost's min_child_weight, sample_size = subsample, learn_rate = eta,
#    mtry = 0.8 with counts = FALSE = colsample_bytree 0.8 (as in xgboost_with_skellam.R)
xg_spec = parsnip::boost_tree(trees = tune(),
                              tree_depth = tune(),
                              learn_rate = tune(),
                              sample_size = tune(), 
                              min_n = tune(), 
                              mtry = 0.8) |>
  set_engine("xgboost", 
             objective = "count:poisson", 
             counts = FALSE, 
             nthread = 1) |>
  set_mode("regression")

# nnet: one hidden layer, linear output. penalty = weight decay
nnet_spec =  parsnip::mlp(hidden_units = tune(), 
                          penalty = tune(), 
                          epochs = 1000) |>
  set_engine("nnet") |>
  set_mode("regression")

# Section 5: Grids----
# n_pred = number of predictors, so rf's mtry goes up to all of them in each predictor set
make_grid = function(model_class, n_pred) {
  switch(model_class,
    pois = dials::grid_regular(penalty(), 
                               mixture(), 
                               levels = 7),
    rf   = expand.grid(mtry = 1:n_pred, 
                       min_n = c(3, 5, 10, 15), 
                       sample.fraction = c(0.7, 0.85, 1.0)),
    # tune fits each combination once at 1000 trees and predicts at fewer, so trees is nearly free
    xg   = expand.grid(trees = c(100, 250, 500, 1000), 
                       tree_depth = c(1, 2, 3),
                       learn_rate = c(0.01, 0.05, 0.1), 
                       sample_size = c(0.5, 0.8),
                       min_n = c(1, 5, 10)),
    nnet = expand.grid(hidden_units = 1:5, 
                       penalty = c(0.5, 0.7, 0.9, 1.0, 1.1, 2.0))
  )
}

# Section 6: Workflow set: every preprocessing x every model----
#     ids are <covariates>_<model_class>, e.g. base_rf, cov_xg. cov_null is dropped 
#     null model ignores predictors, so it would be identical to base_null
wf_set = workflowsets::workflow_set(
  preproc = map(predictor_sets, make_recipe),
  models  = list(null = null_spec, 
                 pois = pois_spec, 
                 rf = rf_spec, 
                 xg = xg_spec, 
                 nnet = nnet_spec)) |> 
  filter(wflow_id != "cov_null")

# keep the held-out predictions. The null model has nothing to tune, so nestedtune runs it through
# nested_fit_resamples, which ignores the shared control: it needs its own
keep_preds = control_grid(save_pred = TRUE)
wf_set = option_add(wf_set, control = keep_preds, id = "base_null")

# attach each workflow's grid
for (id in setdiff(wf_set$wflow_id, "base_null")) {
  covariates  = sub("_.*", "", id) # e.g. "cov_pois" -> "cov"
  model_class = sub(".*_", "", id) # e.g. "cov_pois" -> "pois"
  grid = make_grid(model_class, n_pred = length(predictor_sets[[covariates]]))
  if (QUICK) grid = grid[round(seq(1, nrow(grid), length.out = 3)), ] # shorten for debug
  wf_set = option_add(wf_set, grid = grid, id = id)
}

# Section 7: Run----
# - For each workflow and each outer year: tune on the other 5 years (inner folds, best mae),
#     refit the winner on those 5 years, and predict the held-out year
# - Each workflow is saved to its own file, so only the ones marked in RERUN have to run again
# - Mixing fresh and saved results gives the same numbers as one run of everything: nested_workflow_map
#     starts every workflow from the same random state after set.seed(), so a workflow's result doesn't
#     depend on which other workflows ran with it
stopifnot(setequal(names(RERUN), wf_set$wflow_id)) # one switch per workflow, no typos
result_file = function(id) file.path(RESULTS_DIR, paste0(id, ".rds"))
to_run = wf_set$wflow_id[RERUN[wf_set$wflow_id] | !file.exists(result_file(wf_set$wflow_id))]

# LOAD_ONLY = TRUE is set by scripts that source this one (e.g. community_plots_result_pres.R)
# get0 checks for LOAD_ONLY's existence and stop instead of refitting or loading the QUICK debug fits
if (isTRUE(get0("LOAD_ONLY")) && (length(to_run) > 0 || QUICK)) {
  stop("LOAD_ONLY: this would refit ", paste(to_run, collapse = ", "), if (QUICK) " (QUICK = TRUE)",
       "\n  set RERUN to FALSE / QUICK = FALSE, or run football_combined_nested_tune.R directly")
}

if (length(to_run) > 0) {
  mirai::daemons(2) # sets up parallelization based on 2 physical cores
  set.seed(6769)
  fresh = nestedtune::nested_workflow_map(
    filter(wf_set, wflow_id %in% to_run),
    fn        = "nested_tune_grid",
    resamples = folds,
    metrics   = metric_set(mae, rmse),             # mae first = the selection metric
    select    = selection_rule(rule = "best"),
    control   = keep_preds,
    verbose   = TRUE
  )
  mirai::daemons(0)
  dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
  for (i in seq_len(nrow(fresh))) saveRDS(fresh$result[[i]], result_file(fresh$wflow_id[i]))
}

# nested_res: one row per workflow, with its nested results (fresh or saved) in the result column
nested_res = tibble(wflow_id = wf_set$wflow_id,
                    result   = map(wflow_id, \(id) readRDS(result_file(id))))

# hyperparameters each outer year chose, per workflow
selections = nested_res |>
  mutate(chosen = map(result, collect_selections)) |>
  select(wflow_id, chosen) |>
  unnest(chosen)

# Section 8: Goal predictions----
# - One row per held-out team-game per workflow
# - ".row" is the row in euro, which gives back Year and match_id
# - The neural net can predict slightly negative goals; lambda must be > 0 for Skellam, so clamp
#     (moving a negative prediction to ~0 can only shrink its error, since Goals >= 0)
preds = nested_res |>
  mutate(pred = map(result, collect_predictions)) |>
  select(wflow_id, pred) |>
  unnest(pred) |>
  separate_wider_delim(wflow_id, "_", names = c("covariates", "model_class"), cols_remove = FALSE) |>
  mutate(.pred = pmax(.pred, 1e-6),
         Year = euro$Year[.row], match_id = euro$match_id[.row]) |>
  select(wflow_id, covariates, model_class, Year, match_id, .row, Goals, .pred)

stopifnot(all(count(preds, wflow_id)$n == nrow(euro))) # every row predicted exactly once per workflow

# Section 9: Win/draw/loss----
# team 1 = first row of the match. Goal difference G1 - G2 ~ Skellam(lambda1, lambda2)
lvls = c("win", "draw", "loss")

convert_preds_to_wdl = function(preds) {
  preds |>
    arrange(.row) |>
    group_by(Year, match_id) |>
    summarise(lambda1 = first(.pred), lambda2 = last(.pred),
              goals1  = first(Goals), goals2  = last(Goals), .groups = "drop") |>
    mutate(
      p_win   = pskellam(0, lambda1, lambda2, lower.tail = FALSE),  # G1 - G2 >= 1
      p_draw  = dskellam(0, lambda1, lambda2),                      # G1 - G2 = 0
      p_loss  = pskellam(-1, lambda1, lambda2),                     # G1 - G2 <= -1
      p_range = pmax(p_win, p_draw, p_loss) - pmin(p_win, p_draw, p_loss),
      pred_argmax = factor(lvls[max.col(cbind(p_win, p_draw, p_loss), ties.method = "first")], lvls),
      actual = factor(case_when(goals1 > goals2 ~ "win", goals1 == goals2 ~ "draw", TRUE ~ "loss"), lvls)
    )
}

# Draw rule: call a draw when the three probabilities are within t of each other, otherwise argmax
# (Skellam almost never makes draw the single most likely outcome, so argmax alone never calls draws)
pred_at_t_range = function(wdl, t) {
  factor(if_else(wdl$p_range < t, "draw", as.character(wdl$pred_argmax)), lvls)
}

# Alternative: Call a draw when P_draw is above a threshod
pred_at_t_draw = function(wdl, t) {
  factor(if_else(wdl$p_draw > t, "draw", as.character(wdl$pred_argmax)), lvls)
}

# Macro F1: F1 of each class, then the plain average, so draws count as much as wins
macro_f1 = function(truth, estimate) {
  mean(sapply(lvls, function(k) {
    tp = sum(estimate == k & truth == k)
    fp = sum(estimate == k & truth != k)
    fn = sum(estimate != k & truth == k)
    if (2 * tp + fp + fn == 0) 0 else 2 * tp / (2 * tp + fp + fn)
  }))
}

# t is tuned leave-one-tournament-out: each year's draws are called with the t that had the best
# macro F1 on the OTHER five years, so the score at that t is honest. Ties go to the smallest t
T_GRID_RANGE = seq(0, 0.30, by = 0.025)
T_GRID_DRAW  = seq(.1, 0.40, by = 0.0125)

add_draw_threshold = function(wdl) {
  best_t_draw  = function(w) T_GRID_DRAW[which.max(sapply(T_GRID_DRAW,   \(t) macro_f1(w$actual, pred_at_t_draw(w, t))))]
  best_t_range = function(w) T_GRID_RANGE[which.max(sapply(T_GRID_RANGE, \(t) macro_f1(w$actual, pred_at_t_range(w, t))))]
  years = sort(unique(wdl$Year))
  t_draw_by_year  = sapply(years, \(y) best_t_draw(filter(wdl, Year != y)))
  t_range_by_year = sapply(years, \(y) best_t_range(filter(wdl, Year != y)))
  wdl |>
    mutate(overwrite_t_draw = t_draw_by_year[match(Year, years)],
           overwrite_t_range = t_range_by_year[match(Year, years)],
           pred_t_draw = pred_at_t_draw(wdl, overwrite_t_draw), 
           pred_t_range = pred_at_t_range(wdl, overwrite_t_range))
}

wdl = preds |>
  group_by(wflow_id, covariates, model_class) |>
  group_modify(\(p, key) p |> convert_preds_to_wdl() |> add_draw_threshold()) |>
  ungroup()

stopifnot(all(count(wdl, wflow_id)$n == nrow(euro) / 2)) # one row per match per workflow

# Section 10: Scores----
# Every score is pooled: computed once over all held-out rows / matches, not averaged per year
#   goal level:  mae, rmse, poisson_nll (-log P(actual goals))
#   match level: rps, brier, log_loss use only the probabilities (no threshold involved)
#                accuracy / macro_f1 for argmax and for the tuned draw threshold (_t)
score_workflow = function(p, w) {
  y = cbind(w$actual == "win", w$actual == "draw", w$actual == "loss") * 1
  P = cbind(w$p_win, w$p_draw, w$p_loss)
  tibble(
    metric = c("mae", 
               "rmse", 
               "poisson_nll", 
               "rps", 
               "brier", 
               "log_loss",
               "accuracy_argmax", 
               "accuracy_t_draw", 
               "accuracy_t_range", 
               "macro_f1_argmax", 
               "macro_f1_t_draw",
               "macro_f1_t_range",
               "draw_recall_argmax",
               "draw_recall_t_draw",
               "draw_recall_t_range"),
    value = c(
      mean(abs(p$Goals - p$.pred)),
      sqrt(mean((p$Goals - p$.pred)^2)),
      -mean(dpois(p$Goals, p$.pred, log = TRUE)),
      # RPS: squared errors of the cumulative probabilities P(win), P(win or draw), / 2
      mean(((P[, 1] - y[, 1])^2 + (P[, 1] + P[, 2] - y[, 1] - y[, 2])^2) / 2),
      mean(rowSums((P - y)^2)), #brier
      -mean(log(rowSums(P * y))), #log_loss
      mean(w$pred_argmax == w$actual),
      mean(w$pred_t_draw == w$actual),
      mean(w$pred_t_range == w$actual),
      macro_f1(w$actual, w$pred_argmax),
      macro_f1(w$actual, w$pred_t_draw),
      macro_f1(w$actual, w$pred_t_range),
      sum((w$pred_argmax == "draw")  & (w$actual == "draw")) / sum(w$actual == "draw"), 
      sum((w$pred_t_draw == "draw")  & (w$actual == "draw")) / sum(w$actual == "draw"), 
      sum((w$pred_t_range == "draw") & (w$actual == "draw")) / sum(w$actual == "draw")
    )
  )
}

scores = wdl |>
  distinct(wflow_id, covariates, model_class) |>
  mutate(score = map(wflow_id, \(id) score_workflow(filter(preds, wflow_id == id),
                                                    filter(wdl, wflow_id == id)))) |>
  unnest(score)

