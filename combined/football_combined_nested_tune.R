# README: every model class (poisson, random forest, xgboost, neural net), each with and without the
# AgeDifference and _____ covariates, run through ONE nested leave-one-tournament-out design
#
# An algorithm = workflow (covariates + model class) + selection target + decision rule:
#   goal model -> Skellam win/draw/loss probabilities -> decision rule (argmax, range t, p_draw t)
#   selection target = the inner score that picks the hyperparameters: "mae" / "poisson_log_loss"
#   (two-stage: t picked afterwards on inner macro F1) or "macro_f1" (joint: hyperparameters and t together)
#   algorithm ids look like "base_rf | mae | range"
#
# Important intermediates
#       euro, folds, predictor_sets, wf_set
#       nested_res     one row per workflow (wflow_id = <covariates>_<model_class>), the nestedtune results
#       inner_scores   one row per workflow x outer year x candidate x metric (inner-fold means)
#       choices        one row per workflow x outer year x target x rule: the picked hyperparameters and t
#       fits           one row per distinct pick (workflow x outer year x candidate): its outer predictions
#
# Results live in tables with one column per choice, so use filters to get specific:
#   preds        one row per team-game per algorithm: Goals and the predicted lambda (.pred)
#   wdl          one row per match per algorithm: Skellam probabilities, the actual result and the call
#   scores       one row per metric per algorithm
# e.g. filter(scores, model_class == "rf", rule == "range", metric == "macro_f1")

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
#    colsample_bytree = 0.8: each tree sees a random 80% of the predictors (as in xgboost_with_skellam.R)
#    (not mtry = 0.8: parsnip passes mtry to xgboost as colsample_bynode, which re-samples at every split)
xg_spec = parsnip::boost_tree(trees = tune(),
                              tree_depth = tune(),
                              learn_rate = tune(),
                              sample_size = tune(),
                              min_n = tune()) |>
  set_engine("xgboost",
             objective = "count:poisson",
             colsample_bytree = 0.8,
             counts = FALSE,   # 0.8 is a share of the predictors, not a count
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
    # penalty: 11 values evenly spaced on the log scale from 0.001 to 0.32 (4 per factor of 10),
    #   since the earlier grid picked 0.02 almost every year and its other values were near 0 or 1
    pois = expand.grid(penalty = 10^seq(-3, -0.5, by = 0.25),
                       mixture = c(0, 0.25, 0.5, 0.75, 1)),
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

# attach each workflow's grid
for (id in setdiff(wf_set$wflow_id, "base_null")) {
  covariates  = sub("_.*", "", id) # e.g. "cov_pois" -> "cov"
  model_class = sub(".*_", "", id) # e.g. "cov_pois" -> "pois"
  grid = make_grid(model_class, n_pred = length(predictor_sets[[covariates]]))
  if (QUICK) grid = grid[round(seq(1, nrow(grid), length.out = 3)), ] # shorten for debug
  wf_set = option_add(wf_set, grid = grid, id = id)
}

# Section 7: Match-level helpers and the macro F1 metrics----
# The fits run on parallel workers (mirai), which don't see this script's global environment or its
# loaded packages. The macro F1 metrics run on those workers, so everything they use is defined inside
# local() (functions keep the environment they were made in, and it travels with them to the workers)
# and package functions are called as package::function
match_helpers = local({
  # team 1 = first row of a match. Goal difference G1 - G2 ~ Skellam(lambda1, lambda2)
  lvls = c("win", "draw", "loss")

  # win / draw / loss probabilities for team 1. Returns a tibble, so inside mutate() it adds 3 columns
  wdl_probs = function(lambda1, lambda2) {
    tibble::tibble(p_win  = skellam::pskellam(0, lambda1, lambda2, lower.tail = FALSE),  # G1 - G2 >= 1
                   p_draw = skellam::dskellam(0, lambda1, lambda2),                      # G1 - G2 = 0
                   p_loss = skellam::pskellam(-1, lambda1, lambda2))                     # G1 - G2 <= -1
  }

  # the actual result from team 1's view
  actual_result = function(goals1, goals2) {
    factor(dplyr::case_when(goals1 > goals2 ~ "win", goals1 == goals2 ~ "draw", TRUE ~ "loss"), lvls)
  }

  # Decision rules: turn the three probabilities into a call
  #   argmax: the most likely outcome (Skellam almost never makes draw the most likely, so ~no draws)
  #   range:  draw when the three probabilities are within t of each other, otherwise argmax
  #   p_draw: draw when P(draw) > t, otherwise argmax
  call_result = function(p_win, p_draw, p_loss, rule, t) {
    argmax  = lvls[max.col(cbind(p_win, p_draw, p_loss), ties.method = "first")]
    p_range = pmax(p_win, p_draw, p_loss) - pmin(p_win, p_draw, p_loss)
    factor(dplyr::case_when(rule == "range"  & p_range < t ~ "draw",
                            rule == "p_draw" & p_draw > t  ~ "draw",
                            TRUE ~ argmax), lvls)
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

  # macro F1 of one fold's held-out rows under one rule and t
  #   tune hands a metric each fold's rows in data order, and folds hold out whole tournaments, so
  #   rows 1-2 are one match, rows 3-4 the next, ... (checked on a test run)
  fold_macro_f1 = function(goals, pred, rule, t) {
    stopifnot(length(goals) %% 2 == 0)
    team1  = seq(1, length(goals), by = 2)
    team2  = team1 + 1
    lambda = pmax(pred, 1e-6) # lambda must be > 0 for Skellam (the neural net can predict < 0)
    p = wdl_probs(lambda[team1], lambda[team2])
    macro_f1(actual_result(goals[team1], goals[team2]),
             call_result(p$p_win, p$p_draw, p$p_loss, rule, t))
  }

  # Turn one (rule, t) into a yardstick metric that tune computes for every fold and every candidate
  #   numeric_metric_summarizer() keeps tune's grouping, so there's one value per candidate
  make_f1_metric = function(name, rule, t) {
    metric = function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
      yardstick::numeric_metric_summarizer(
        name = name,
        fn = function(truth, estimate, ...) fold_macro_f1(truth, estimate, rule, t),
        data = data, truth = !!rlang::enquo(truth), estimate = !!rlang::enquo(estimate),
        na_rm = na_rm, case_weights = !!rlang::enquo(case_weights))
    }
    yardstick::new_numeric_metric(metric, direction = "maximize")
  }

  environment() # local() returns the environment holding the functions above
})
list2env(as.list(match_helpers), envir = globalenv()) # so the rest of the script can call them by name

# The decision rules and thresholds tried: one row per (rule, t), argmax has no t
rules = bind_rows(tibble(rule = "argmax", t = NA),
                  tibble(rule = "range",  t = seq(0, 0.30, by = 0.025)),
                  tibble(rule = "p_draw", t = seq(0.10, 0.40, by = 0.025))) |>
  mutate(metric_name = if_else(rule == "argmax", "f1_argmax", sprintf("f1_%s_%.3f", rule, t)))

f1_metrics = pmap(list(rules$metric_name, rules$rule, rules$t), make_f1_metric)
names(f1_metrics) = rules$metric_name

# Every metric computed on every inner fold for every candidate (stored in the results)
#   mae first = the metric nestedtune itself selects on. The others are for the other selection targets
run_metrics = do.call(metric_set, c(list(mae = mae, poisson_log_loss = poisson_log_loss), f1_metrics))

# Section 8: Run----
# - For each workflow and each outer year: tune on the other 5 years (inner folds, best mae),
#     refit the winner on those 5 years, and predict the held-out year
# - Each workflow is saved to its own file, so only the ones marked in RERUN have to run again
# - Every workflow starts from the same random state (set.seed before each), so a workflow's result
#     doesn't depend on which other workflows ran with it
stopifnot(setequal(names(RERUN), wf_set$wflow_id)) # one switch per workflow, no typos
result_file = function(id) file.path(RESULTS_DIR, paste0(id, ".rds"))

# a saved result from before a metric was added (or changed), or one whose folds all failed, can't be
# used, so it runs again too (base_null has no inner folds, so it has no inner metrics to check)
lacks_metrics = function(id) {
  if (id == "base_null" || !file.exists(result_file(id))) return(FALSE)
  saved_metrics = unique(unlist(map(readRDS(result_file(id))$.inner_metrics, ".metric")))
  !all(rules$metric_name %in% saved_metrics)
}
to_run = wf_set$wflow_id[RERUN[wf_set$wflow_id] | !file.exists(result_file(wf_set$wflow_id)) |
                         map_lgl(wf_set$wflow_id, lacks_metrics)]

# LOAD_ONLY = TRUE is set by scripts that source this one (e.g. community_plots_result_pres.R)
# get0 checks for LOAD_ONLY's existence and stop instead of refitting or loading the QUICK debug fits
if (isTRUE(get0("LOAD_ONLY")) && (length(to_run) > 0 || QUICK)) {
  stop("LOAD_ONLY: this would refit ", paste(to_run, collapse = ", "), if (QUICK) " (QUICK = TRUE)",
       "\n  set RERUN to FALSE / QUICK = FALSE, or run football_combined_nested_tune.R directly")
}

if (length(to_run) > 0) {
  mirai::daemons(2) # sets up parallelization based on 2 physical cores
  dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
  for (id in to_run) {
    set.seed(6769)
    elapsed = system.time({
      fresh = nestedtune::nested_workflow_map(
        filter(wf_set, wflow_id == id),
        fn        = "nested_tune_grid",
        resamples = folds,
        metrics   = run_metrics,
        select    = selection_rule(rule = "best"),
        verbose   = TRUE
      )
    })[["elapsed"]]
    result = fresh$result[[1]]
    attr(result, "elapsed_sec") = elapsed # how long the run took, to plan the next ones
    saveRDS(result, result_file(id))       # saved right away, so a crash keeps the finished ones
    cat(id, "took", round(elapsed / 60, 1), "min\n")
  }
  mirai::daemons(0)
}

# nested_res: one row per workflow, with its nested results (fresh or saved) in the result column
nested_res = tibble(wflow_id = wf_set$wflow_id,
                    result   = map(wflow_id, \(id) readRDS(result_file(id))))

# Section 9: Selections: which hyperparameters and t each algorithm uses----
# An algorithm = workflow + selection target + decision rule. For each outer year it picks, using
# only that year's inner folds:
#   target "mae" or "poisson_log_loss" (two-stage): the candidate with the best inner goal score,
#       then the t with the best inner macro F1 for that candidate
#   target "macro_f1" (joint): the (candidate, t) with the best inner macro F1, picked together
# Ties go to the smallest t, then to the first candidate in the grid
# base_null has nothing to tune, so it gets one row per outer year: target "none", rule argmax
# cov workflows only get target "mae": they're compared with base under that target only

# inner_scores: one row per workflow x outer year x candidate x metric, the metric's mean over the
#   5 inner folds. The models have different hyperparameters, so they're packed into one list
#   column (params), and params_label shows them as text, e.g. "penalty=0.0215, mixture=1"
inner_scores_of = function(result) {
  scores = collect_inner_metrics(result)
  param_cols = setdiff(names(scores), c("id", ".metric", ".estimator", "mean", "n", "std_err", ".config"))
  scores |>
    nest(params = all_of(param_cols)) |>
    mutate(params_label = map_chr(params, \(p) paste(names(p), signif(unlist(p), 3), sep = "=", collapse = ", ")))
}
inner_scores = nested_res |>
  filter(wflow_id != "base_null") |>
  mutate(scores = map(result, inner_scores_of)) |>
  select(wflow_id, scores) |>
  unnest(scores) |>
  left_join(select(rules, metric_name, rule, t), by = c(".metric" = "metric_name")) # NA for mae / NLL

f1_scores = inner_scores |> filter(!is.na(rule)) |> arrange(t)       # arranged so ties -> smallest t

# two-stage: the best candidate on the goal score (lowest), then that candidate's best t
pick_two_stage = function(target) {
  best_candidate = inner_scores |>
    filter(.metric == target) |>
    group_by(wflow_id, id) |>
    slice_min(mean, n = 1, with_ties = FALSE) |>
    ungroup() |>
    select(wflow_id, id, .config)
  f1_scores |>
    semi_join(best_candidate, by = c("wflow_id", "id", ".config")) |>
    group_by(wflow_id, id, rule) |>
    slice_max(mean, n = 1, with_ties = FALSE) |>
    ungroup() |>
    mutate(target = target)
}

# joint: the best (candidate, t) on macro F1, for each rule
pick_joint = f1_scores |>
  group_by(wflow_id, id, rule) |>
  slice_max(mean, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(target = "macro_f1")

# choices: one row per workflow x outer year x target x rule, with the picked candidate and t
#   inner_f1 = the inner macro F1 of that pick (what the joint target maximized)
null_choices = tibble(wflow_id = "base_null", id = folds$id, target = "none", rule = "argmax",
                      t = NA, .config = "none", params = list(tibble()), params_label = "", mean = NA)
choices = bind_rows(pick_two_stage("mae"), pick_two_stage("poisson_log_loss"), pick_joint, null_choices) |>
  separate_wider_delim(wflow_id, "_", names = c("covariates", "model_class"), cols_remove = FALSE) |>
  filter(covariates == "base" | target %in% c("mae", "none")) |>
  select(wflow_id, covariates, model_class, id, target, rule, t, .config, params, params_label,
         inner_f1 = mean)

# check: target "mae" must pick exactly what nestedtune itself picked (it selects on inner mae)
nestedtune_picks = nested_res |>
  filter(wflow_id != "base_null") |>
  mutate(chosen = map(result, collect_selections)) |>
  select(wflow_id, chosen) |>
  unnest(chosen)
stopifnot(nrow(anti_join(distinct(filter(choices, target == "mae"), wflow_id, id, .config),
                         nestedtune_picks, by = c("wflow_id", "id", ".config"))) == 0)

# Section 10: Outer predictions for every algorithm----
# choices -> fits -> preds:
# - every pick in choices is refit by hand: fit the picked candidate on the 5 outer-training years and
#     predict the held-out year (what nestedtune does after its own pick, done here for every target)
# - several algorithms often share a pick (e.g. argmax / range / p_draw under mae), so each is fit once
# - the fits are saved to RESULTS_DIR/outer_fits.rds, so sourcing this script again doesn't refit them

# one refit: one row per held-out team-game
refit_outer = function(wflow_id, outer_id, params) {
  split = folds$splits[[which(folds$id == outer_id)]]
  wf = extract_workflow(wf_set, wflow_id) |> finalize_workflow(params)
  set.seed(6769)
  fitted = fit(wf, analysis(split))
  tibble(.row  = as.integer(split, data = "assessment"),
         Goals = assessment(split)$Goals,
         .pred = predict(fitted, assessment(split))$.pred)
}

# fits: one row per distinct (workflow, outer year, candidate) that some algorithm uses
fits = choices |>
  distinct(wflow_id, id, .config, .keep_all = TRUE) |>
  select(wflow_id, id, .config, params)

# reuse the saved fits only if they are for exactly these picks and no workflow was just rerun,
# otherwise refit them all (a couple of minutes)
FITS_FILE = file.path(RESULTS_DIR, "outer_fits.rds")
saved_fits = if (file.exists(FITS_FILE)) readRDS(FITS_FILE)
if (length(to_run) == 0 && !is.null(saved_fits) && identical(select(saved_fits, -pred), fits)) {
  fits = saved_fits
} else {
  if (isTRUE(get0("LOAD_ONLY"))) stop("LOAD_ONLY: the outer fits are missing or out of date, run this script directly")
  cat("refitting", nrow(fits), "picks\n")
  fits$pred = pmap(list(fits$wflow_id, fits$id, fits$params), refit_outer)
  saveRDS(fits, FITS_FILE)
}

# preds: one row per held-out team-game per algorithm
# - ".row" is the row in euro, which gives back Year and match_id
# - lambda must be > 0 for Skellam, so clamp (the neural net can predict slightly negative goals;
#     moving a negative prediction to ~0 can only shrink its error, since Goals >= 0)
preds = choices |>
  select(-params) |>
  left_join(select(fits, -params), by = c("wflow_id", "id", ".config")) |>
  unnest(pred) |>
  mutate(.pred = pmax(.pred, 1e-6),
         Year = euro$Year[.row], match_id = euro$match_id[.row],
         algorithm = paste(wflow_id, target, rule, sep = " | ")) |>
  select(algorithm, wflow_id, covariates, model_class, target, rule, id, Year, match_id, .row,
         t, params_label, Goals, .pred)

stopifnot(all(count(preds, algorithm)$n == nrow(euro))) # every row predicted exactly once per algorithm

# Section 11: Win/draw/loss----
# wdl: one row per match per algorithm, with the call under that algorithm's rule and t
wdl = preds |>
  arrange(.row) |>
  group_by(algorithm, wflow_id, covariates, model_class, target, rule, Year, match_id) |>
  summarise(t = first(t), lambda1 = first(.pred), lambda2 = last(.pred),
            goals1  = first(Goals), goals2  = last(Goals), .groups = "drop") |>
  mutate(wdl_probs(lambda1, lambda2),
         actual = actual_result(goals1, goals2),
         call   = call_result(p_win, p_draw, p_loss, rule, t))

# historical frequency baseline: each match gets the win / draw / loss shares of the other 5 years
matches = wdl |> distinct(Year, match_id, goals1, goals2, actual)
shares_without = function(year) prop.table(table(filter(matches, Year != year)$actual))
baseline_wdl = matches |>
  mutate(p_win  = map_dbl(Year, \(y) shares_without(y)[["win"]]),
         p_draw = map_dbl(Year, \(y) shares_without(y)[["draw"]]),
         p_loss = map_dbl(Year, \(y) shares_without(y)[["loss"]]),
         call   = call_result(p_win, p_draw, p_loss, "argmax", NA),
         algorithm = "historical | none | argmax", wflow_id = "historical", covariates = "none",
         model_class = "historical", target = "none", rule = "argmax")
wdl = bind_rows(wdl, baseline_wdl)

stopifnot(all(count(wdl, algorithm)$n == nrow(euro) / 2)) # one row per match per algorithm

# Section 12: Scores----
# Every score is pooled: computed once over all held-out rows / matches, not averaged per year
#   goal level:  mae, rmse, poisson_nll (-log P(actual goals)). Not for the historical baseline
#   match level: rps, brier, log_loss use only the probabilities (the rule doesn't change them)
#                accuracy, macro_f1, draw_recall, draws_called use the call
goal_scores = preds |>
  group_by(algorithm, wflow_id, covariates, model_class, target, rule) |>
  summarise(mae         = mean(abs(Goals - .pred)),
            rmse        = sqrt(mean((Goals - .pred)^2)),
            poisson_nll = -mean(dpois(Goals, .pred, log = TRUE)),
            .groups = "drop")

match_scores = wdl |>
  group_by(algorithm, wflow_id, covariates, model_class, target, rule) |>
  summarise(
    # RPS: squared errors of the cumulative probabilities P(win), P(win or draw), / 2
    rps          = mean(((p_win - (actual == "win"))^2 +
                         (p_win + p_draw - (actual != "loss"))^2) / 2),
    brier        = mean((p_win - (actual == "win"))^2 + (p_draw - (actual == "draw"))^2 +
                        (p_loss - (actual == "loss"))^2),
    log_loss     = -mean(log(case_when(actual == "win" ~ p_win, actual == "draw" ~ p_draw, TRUE ~ p_loss))),
    accuracy     = mean(call == actual),
    macro_f1     = macro_f1(actual, call),
    draw_recall  = sum(call == "draw" & actual == "draw") / sum(actual == "draw"),
    draws_called = sum(call == "draw"),
    .groups = "drop")

# scores: one row per metric per algorithm
scores = full_join(goal_scores, match_scores,
                   by = c("algorithm", "wflow_id", "covariates", "model_class", "target", "rule")) |>
  pivot_longer(-c(algorithm, wflow_id, covariates, model_class, target, rule),
               names_to = "metric", values_to = "value")
