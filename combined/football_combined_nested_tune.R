# Experimental design:
# - Main grid (loto, base): {model class} x {target} x {decision rule}, fully crossed
#     model class:   pois, rf, xg, nnet (+ baselines null, historical)
#     target:        mae, poisson_log_loss, rps (two-stage) or macro_f1 (joint)
#     decision rule: argmax, range, p_draw (t for range / p_draw always picked on inner macro F1)
# - Sensitivity checks against the main grid:
#     cov:       + AgeFitDiff, target mae only (every model class and decision rule)
#     expanding: expanding instead of loto, base only (full {target} x {decision rule} grid)
#     seeds:     rf / xg / nnet picks refit with 5 other seeds (outer refit only, no re-tuning)

# Keywords
# - "base" / "cov": the original covariates vs base + the new AgeFitDiff covariate
# - model classes:
#     "null":       predicts the training mean goals for every team-match
#     "historical": skips goals, uses the training years' win / draw / loss shares
#     "pois":       Poisson elastic net (glmnet)
#     "rf":         random forest (ranger)
#     "xg":         gradient boosted trees (xgboost, Poisson objective)
#     "nnet":       one-hidden-layer neural net
# - "loto" / "expanding": nested leave one tournament out / nested expanding window CV
# - "wflow_id": <covariates>_<model_class>, e.g. base_rf, cov_pois, base_null
# - "team-match": one team's row of a match (each match has two, adjacent in euro)
# - "outer fold": one held-out test year
# - "inner fold": one tuning split within an outer fold's training years
# - "candidate" (.config): one hyperparameter combination from a workflow's grid
# - "decision rule" or "rule": turns P(win, draw, loss) into a "call":
#     argmax, or range / p_draw with a specified threshold t
# - "target": the inner score that picks the candidate
#     mae, poisson_log_loss, rps (two-stage: candidate first, then t on macro F1);
#     macro_f1 (joint: candidate and t together); none (null / historical)
# - "algorithm": wflow_id + target + decision rule, e.g. "base_rf | mae | range"
# - "pick(s)": the candidate and t an algorithm chooses for one outer fold (using only its inner folds)

# Important objects (in order of appearance):
# - euro:        one obs per {team-match}
# - folds:       one obs per {outer fold}, each with its inner folds
# - wf_set:      one obs per {wflow_id}, with its grid
# - run_metrics: every metric computed on the inner folds
# - nested_res:  one obs per {wflow_id}, the saved nestedtune results
# - choices:     one obs per {wflow_id} x {outer fold} x {target} x {decision rule}: the pick
# - fits:        one obs per {wflow_id} x {outer fold} x {candidate}: the outer predictions
#                (shared by every pick with that candidate)
# - preds:       one obs per {algorithm} x {team-match}: Goals and predicted lambda
# - wdl:         one obs per {algorithm} x {match}: probabilities, actual result, call
# - scores:      one obs per {algorithm} x {metric} (pooled over all held-out matches)
# - seed_scores:  one obs per {algorithm} x {seed}: every score (rf / xg / nnet, the main seed + 5 others)
# - seed_summary: one obs per {algorithm} x {metric}: mean, sd, min and max over the seeds (the default)
# - expanding:    environment holding the expanding run's objects (expanding$scores, ...)
# - finals:       the final models by name (finals$pois, finals$rf), for predict_final()

# Section 0: Settings----

# Designed to save run time: 
# - RERUN controls at workflow level whether the saved .rds is overwritten (otherwise read from existing file)
# - Notes:
#       - a workflow will rerun anyway if its .rds is missing or a metric for an existing .rds is missing
#       - RERUN doesn't detect changed grids, models, predictors or folds
RERUN = c(
  base_null = FALSE,
  base_pois = FALSE, base_rf = FALSE, base_xg = FALSE, base_nnet = FALSE,
  cov_pois  = FALSE, cov_rf  = FALSE, cov_xg  = FALSE, cov_nnet  = FALSE
)

# Quick is another alternative to save time and debug which lowers hyperparam grid sizes (saves to results_debug/; LOAD_ONLY refuses QUICK results)
QUICK = FALSE

# Controls section 14 which reruns this script using an expanding window CV structure
RUN_EXPANDING = TRUE

# sets default (if not specified) to leave one tournament out rather than expanding window
if (!exists("RESAMPLING")) RESAMPLING = "loto"
stopifnot(RESAMPLING %in% c("loto", "expanding"))

# logic on where to save results
RESULTS_DIR = file.path(if (QUICK) "results_debug" else "results",
                        if (RESAMPLING == "loto") "combined" else "combined_expanding")

# libraries
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

# Section 1: Cleaning "euro" dataset----
euro = rbind(read_csv("raw-data/EuroTest.csv"), 
             read_csv("raw-data/EuroTraining.csv"))

# ages was created in a separate script using data from https://fbref.com/en/comps/676/stats/UEFA-Euro-Stats
ages = read_csv("raw-data/Eurodata_with_Age_New.csv")
stopifnot(all(ages$Team == euro$Team), all(ages$Opponent == euro$Opponent), all(ages$Year == euro$Year))
euro$AgeFitDiff = ages$AgeFitDiff # note flips sign across team for a match 

# fixes an error in the groupstage variable from 2024 tournament
rows_2024 = which(euro$Year == 2024)
euro$GroupStage[rows_2024[1:72]] = 1

# note that in the euro dataset two rows of the same match are adjacent
euro$match_id = rep(1:(nrow(euro) / 2), each = 2)
odd = seq(1, nrow(euro), by = 2)
stopifnot(all(euro$Team[odd] == euro$Opponent[odd + 1]))

# Section 2: Nested splits----

# helper for the nested expanding window 
expanding_splits = function(data, skip) {
  test_years = sort(unique(data$Year))[-(1:skip)]
  splits = map(test_years, \(year) make_splits(list(analysis   = which(data$Year < year),
                                                     assessment = which(data$Year == year)), data = data))
  manual_rset(splits, ids = paste0("Resample", seq_along(test_years)))
}

set.seed(67676769)

# chooses nested leave one tournament out or nested expanding window based on flag
if (RESAMPLING == "loto") {
  folds = nested_resamples(euro, outside = group_vfold_cv(group = Year), inside = group_vfold_cv(group = Year))
} else {
  folds = nested_resamples(euro, outside = expanding_splits(skip = 2), inside = expanding_splits(skip = 1))
}

# extra code for readability of rsample objects looping and labeling which eval and test holdout years
held_out_year = function(split) unique(assessment(split)$Year)
rename_folds = function(rset, ids) {
  rset$id = ids
  for (i in seq_along(rset$splits)) {
    rset$splits[[i]]$id$id = ids[i]
  }
  return (rset)
}
test_years = map_dbl(folds$splits, held_out_year)
folds = rename_folds(folds, paste("test", test_years))
for (i in seq_along(folds$inner_resamples)) {
  eval_years = map_dbl(folds$inner_resamples[[i]]$splits, held_out_year)
  folds$inner_resamples[[i]] = rename_folds(folds$inner_resamples[[i]],
                                            paste0("test ", test_years[i], " / eval ", eval_years))
}

# adding metric weights based on size of eval and test hold outs to achieve pooled estimates (technically macro_f1 a little off)
assessment_weights = function(rset) {
  n = map_int(rset$splits, \(split) nrow(assessment(split)))
  n / sum(n)
}
folds = tune::add_resample_weights(folds, assessment_weights(folds))
for (i in seq_along(folds$inner_resamples)) {
  folds$inner_resamples[[i]] = add_resample_weights(folds$inner_resamples[[i]],
                                                    assessment_weights(folds$inner_resamples[[i]]))
}

# Section 3: choosing covariates & including standardization----
base_predictors = c("GroupStage", "GDP", "MarketValue", "FifaRank", "UefaPoints", "CLPlayers")
new_covariates  = c("AgeFitDiff")
predictor_sets  = list(base = base_predictors,
                       cov  = c(base_predictors, new_covariates))

# Function to add predictor demeaning and division by sd using appropriate training data as part of the workflow 
make_recipe = function(predictors) {
  recipe(reformulate(predictors, response = "Goals"), data = euro) |>
    step_normalize(all_numeric_predictors())
}

# Section 4: Models (the model class choice)----

# parsnip models are "workflow" compatible layer of abstraction of package implementation 
#      e.g. rpart and partykit both called from decision_tree() with standardized hyperparams

# null: predicts the training mean goals
null_spec = parsnip::null_model(mode = "regression") |> set_engine("parsnip")

# poisson elastic net
pois_spec = parsnip::poisson_reg(penalty = tune(),     # penalty: overall regularization
                                 mixture = tune()) |>  # mixture: 1 = lasso, 0 = ridge
  set_engine("glmnet")

# random forest
rf_spec = parsnip::rand_forest(mtry = tune(),    # mtry: # of columns randomly selected at split
                               min_n = tune(),   # min_n: min # of data for a split
                               trees = 1000) |>  # trees: total trees in ensemble
  set_engine("ranger",
             sample.fraction = tune(),  # sample.fraction: Fraction of obs to sample for each tree (1.0 is ordinary boostrap)
             num.threads = 1) |> 
  set_mode("regression")
# translate(rf_spec) # for more details about the call to ranger

# xgboost
xg_spec = parsnip::boost_tree(trees = tune(),       # trees: total # in ensemble
                              tree_depth = tune(),  # tree_depth: max depth of tree
                              learn_rate = tune(),  # learn_rate: rate which boosting algorithm adapts btw iterations
                              sample_size = tune(), # sample_size: proportion of data for fitting in each iteration
                              min_n = tune()) |>
  set_engine("xgboost",
             objective = "count:poisson", # poisson nll with log link; if there were no regularization, ideal leaf update optimizes loss by log(observed_count/predicted_count)
             colsample_bytree = 0.8,      # colsample_bytree: proportion of columns subsampled during construction of tree
             counts = FALSE,              # makes colsample_by tree interpreted btw 0 and 1
             nthread = 1) |>
  set_mode("regression")

# neural net
nnet_spec =  parsnip::mlp(hidden_units = tune(), # hidden_units: # of nodes in mlp's one hidden layer
                          penalty = tune(),      # penalty: ridge-like penalty for both linear combo of sigmoids and linear combos of X's being fed to sigmoids
                          epochs = 1000) |>      # epochs: max number of steps of BFGS algorithm for penalized weight optimization
  set_engine("nnet") |>
  set_mode("regression")

# Section 5: Grids----

# one function to provide a grid based on what model_class you call
make_grid = function(model_class, n_pred) {
  switch(model_class,
    pois = expand.grid(penalty = 10^seq(-3, -0.5, by = 0.25),
                       mixture = c(0, 0.25, 0.5, 0.75, 1)),
    rf   = expand.grid(mtry = 1:n_pred,
                       min_n = c(3, 5, 10, 15),
                       sample.fraction = c(0.7, 0.85, 1.0)),
    xg   = expand.grid(trees = c(100, 250, 500, 1000),       # I wonder if xg does artificially well bc its grid is relatively larger
                       tree_depth = c(1, 2, 3),
                       learn_rate = c(0.01, 0.05, 0.1),
                       sample_size = c(0.5, 0.8),
                       min_n = c(1, 5, 10)),
    nnet = expand.grid(hidden_units = 1:5,
                       penalty = c(0.5, 0.7, 0.9, 1.0, 1.1, 2.0))
  )
}

# Section 6: Workflow set: {standardized-base, standardized-cov} x {null, pois, rf, xg, nnet}----

# wf_set is a core object to this script that contains all the model specifications inclusive of formula
wf_set = workflowsets::workflow_set(
  preproc = map(predictor_sets, make_recipe),
  models  = list(null = null_spec,
                 pois = pois_spec,
                 rf = rf_spec,
                 xg = xg_spec,
                 nnet = nnet_spec)) |>
  filter(wflow_id != "cov_null")
if (RESAMPLING == "expanding") wf_set = filter(wf_set, startsWith(wflow_id, "base_")) # no new age covariate for expanding window

# tacking on grid column to the wf_set object via option_add
for (id in setdiff(wf_set$wflow_id, "base_null")) {
  covariates  = sub("_.*", "", id)
  model_class = sub(".*_", "", id)
  grid = make_grid(model_class, n_pred = length(predictor_sets[[covariates]]))
  if (QUICK) grid = grid[round(seq(1, nrow(grid), length.out = 3)), ] # QUICK mode reduces each grid size to 3
  wf_set = option_add(wf_set, grid = grid, id = id)                   # here you could save predictions but we recalculate by hand later
}

# Section 7: Match-level helpers and the macro F1 / RPS metrics----

# Context: 
# - we use mirai package for parallelization to shorten runtimes
# - the mirai "daemons" don't have access to the global environment
# - here we package lots of function definitions into a variable scope they're able to access
match_helpers = local({
  lvls = c("win", "draw", "loss")
  
  # uses skellam to convert to win, draw, loss
  wdl_probs = function(lambda1, lambda2) {
    tibble::tibble(p_win  = skellam::pskellam(0, lambda1, lambda2, lower.tail = FALSE),
                   p_draw = skellam::dskellam(0, lambda1, lambda2),
                   p_loss = skellam::pskellam(-1, lambda1, lambda2))
  }

  # which team actually won
  actual_result = function(goals1, goals2) {
    factor(dplyr::case_when(goals1 > goals2 ~ "win", goals1 == goals2 ~ "draw", TRUE ~ "loss"), lvls)
  }

  # replace argmax decision with draws using a draw threshold rule \in {"range", "p_draw"} and a threshold values, t
  call_result = function(p_win, p_draw, p_loss, rule, t) {
    argmax  = lvls[max.col(cbind(p_win, p_draw, p_loss), ties.method = "first")]
    p_range = pmax(p_win, p_draw, p_loss) - pmin(p_win, p_draw, p_loss)
    factor(dplyr::case_when(rule == "range"  & p_range < t ~ "draw",
                            rule == "p_draw" & p_draw > t  ~ "draw",
                            TRUE ~ argmax), lvls)
  }

  # macro_f1: arithmetic mean across classes of harmonic mean of class precision & class recall
  macro_f1 = function(truth, estimate) {
    mean(sapply(lvls, function(k) {
      tp = sum(estimate == k & truth == k)
      fp = sum(estimate == k & truth != k)
      fn = sum(estimate != k & truth == k)
      if (2 * tp + fp + fn == 0) 0 else 2 * tp / (2 * tp + fp + fn)
    }))
  }

  # ranked probability score: square penalty for cumulative class probabilities 
  mean_rps = function(p_win, p_draw, actual) {
    mean(((p_win - (actual == "win"))^2 + (p_win + p_draw - (actual != "loss"))^2) / 2)
  }

  # takes team-match level vec and generates match-level df of p_win, p_draw, p_loss, actual
  fold_matches = function(goals, pred) {
    stopifnot(length(goals) %% 2 == 0)
    team1  = seq(1, length(goals), by = 2) # team1 is first appearance in row order
    team2  = team1 + 1
    lambda = pmax(pred, 1e-6)
    matches = wdl_probs(lambda[team1], lambda[team2])
    matches$actual = actual_result(goals[team1], goals[team2])
    matches
  }

  # returns macro F1 of one fold given team-match level vecs (goals, pred) for a given decision rule and t
  fold_macro_f1 = function(goals, pred, rule, t) {
    m = fold_matches(goals, pred)
    wdl_call_after_threshold = call_result(m$p_win, m$p_draw, m$p_loss, rule, t)
    macro_f1(truth = m$actual, wdl_call_after_threshold)
  }
  
  # returns ranked prob score averaged over match-level based on team-match level vecs
  fold_rps = function(goals, pred) {
    m = fold_matches(goals, pred)
    mean_rps(m$p_win, m$p_draw, m$actual)
  }

  # context: 
  # - yardstick is a package that contains metrics to help tune workflows
  # - for a given candidate hyperparameter, simple metrics like mae can be calculated based on a single obs
  # - because we want to tune our hyperparams using rps and macro_f1 we had to find a workaround by implementing a custom yardstick metric
  # - think about make_match_metric as a function factory
  make_match_metric = function(name, fold_score, direction) {
    metric = function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
      yardstick::numeric_metric_summarizer(
        name = name,
        fn = function(truth, estimate, ...) fold_score(truth, estimate),
        data = data, truth = !!rlang::enquo(truth), estimate = !!rlang::enquo(estimate), # supports the tidyverse way of passing column names without quotations
        na_rm = na_rm, case_weights = !!rlang::enquo(case_weights))
    }
    yardstick::new_numeric_metric(metric, direction = direction)
  }
  
  # another function factory to make one metric per (decision rule, t)
  #    motivated by doing our grid search of t by using metrics rather than rerunning directly 
  make_f1_metric = function(name, rule, t) {
    make_match_metric(name, function(goals, pred) fold_macro_f1(goals, pred, rule, t), "maximize")
  }
  
  # rps custom yardstick-package-compatible metric
  rps_metric = make_match_metric("rps", fold_rps, "minimize")

  environment()
})
list2env(as.list(match_helpers), envir = globalenv())

# grid of draw threshold decision rules to be compared
rules = bind_rows(tibble(rule = "argmax", t = NA),
                  tibble(rule = "range",  t = seq(0, 0.30, by = 0.025)),
                  tibble(rule = "p_draw", t = seq(0.10, 0.40, by = 0.025))) |>
  mutate(metric_name = if_else(rule == "argmax", "f1_argmax", sprintf("f1_%s_%.3f", rule, t)))

# a list of yardstick metrics, one for each (decision rule, t) in rules
f1_metrics = pmap(list(rules$metric_name, rules$rule, rules$t), make_f1_metric)
names(f1_metrics) = rules$metric_name

# combines all the metrics we want to test hyperparameter tuning on into one yardstick::metric_set
run_metrics = do.call(metric_set, c(list(mae = mae, poisson_log_loss = poisson_log_loss, rps = rps_metric),
                                    f1_metrics))

# Section 8: Run----

# safeguarding using RERUN flags
stopifnot(all(wf_set$wflow_id %in% names(RERUN)))
result_file = function(id) file.path(RESULTS_DIR, paste0(id, ".rds"))

# add to the list of workflows to rerun if saved .rds isn't compatible with changes in the metric_set 
lacks_metrics = function(id) {
  if (id == "base_null" || !file.exists(result_file(id))) return(FALSE)
  saved_metrics = unique(unlist(map(readRDS(result_file(id))$.inner_metrics, ".metric")))
  !all(c("rps", rules$metric_name) %in% saved_metrics)
}

# rerun a wflow_id if...
to_run = wf_set$wflow_id[
  RERUN[wf_set$wflow_id] |                     # our RERUN flag is on
  !file.exists(result_file(wf_set$wflow_id)) | # if no existing .rds 
  map_lgl(wf_set$wflow_id, lacks_metrics)      # if there are missing metrics in existing .rds
]

# Prevent the main loop of refit if this file is being sourced from community_plots_result_pres.R
if (isTRUE(get0("LOAD_ONLY")) && (length(to_run) > 0 || QUICK)) {
  stop("LOAD_ONLY: this would refit ", paste(to_run, collapse = ", "), if (QUICK) " (QUICK = TRUE)",
       "\n  set RERUN to FALSE / QUICK = FALSE, or run football_combined_nested_tune.R directly")
}

# the main loop! 
if (length(to_run) > 0) {
  
  # set one worker per physical CPU core
  mirai::daemons(2)
  dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
  for (id in to_run) {
    set.seed(6769)
    
    # times the call
    elapsed = system.time({
      
      # context: 
      # - nestedtune is a package to hyperparameter tuning using nested CV
      # - nested_tune_grid: runs & organizes the tuning results into highly structured tibbles
      # - nested_workflow_map: for each workflow id runs a nested_tune_grid 
      # - we borrow this high level code for its organized output and saved intermediates 
      #   but end up having to recalculate the predictions, outer loop hyperparam selections, etc in section 9 and 10
      # - for example, we're 1) doing hyperparameter tuning across the outputted metrics, and 2) want to compare the interaction 
      #   between a sequential goal-based then draw threshold performance model vs directly optimizing hyperparameters based on draw macro F1
      fresh = nestedtune::nested_workflow_map(
        filter(wf_set, wflow_id == id),
        fn        = "nested_tune_grid",           
        resamples = folds,                         # expanding window or leave one tournament out, depending
        metrics   = run_metrics,                   # macro F1 for every (decision rule, t), rps, mae, poisson_log_loss
        select    = selection_rule(rule = "best"), # results discarded: chooses "best" candidate for an outer fold based on first metric in run_metrics (mae) 
        verbose   = TRUE
      )
    })[["elapsed"]]
    result = fresh$result[[1]]
    attr(result, "elapsed_sec") = elapsed
    saveRDS(result, result_file(id))
    cat(id, "took", round(elapsed / 60, 1), "min\n")
  }
  mirai::daemons(0)
}

# nested_res contains all computationally heavy results and everything to follow is derived from this object
nested_res = tibble(wflow_id = wf_set$wflow_id,                 
                    result   = map(wflow_id, \(id) readRDS(result_file(id)))) # combines the saved .rds and rerun-ed workflows

# Section 9: Selections: which hyperparameters and t each algorithm uses----

# returns averaged inner fold metrics (loto: n=5, expanding: n in [1,4]) to extract one obs per {outer fold} x {candidate} x {metric}
inner_scores_of = function(result) {
  
  # provides a table where metrics have been averaged over inner splits
  scores = collect_inner_metrics(result)
  
  # collapse all the hyperparameter columns into one column of dfs
  param_cols = setdiff(names(scores), c("id", ".metric", ".estimator", "mean", "n", "std_err", ".config"))
  scores |>
    nest(params = all_of(param_cols)) |>
    mutate(params_label = map_chr(params, \(p) paste(names(p), signif(unlist(p), 3), sep = "=", collapse = ", ")))
}

# averaged inner fold metric (loto: n=5, expanding: n in [1,4]) one obs per {wflow_id} x {outer fold} x {candidate} x {metric}
inner_scores = nested_res |>
  filter(wflow_id != "base_null") |>
  mutate(scores = map(result, inner_scores_of)) |>  
  select(wflow_id, scores) |>
  unnest(scores) |>
  left_join(select(rules, metric_name, rule, t), by = c(".metric" = "metric_name"))

# subset of just macro F1 metrics
f1_scores = inner_scores |> filter(!is.na(rule)) |> arrange(t) # smaller t's break ties

# choose the best candidate per outer fold using a two-stage decision
pick_two_stage = function(target) {
  
  # (first stage: goal prediction) plays role of selection_rule("best") in nested_tune_grid for the "target" metric
  best_candidate = inner_scores |>
    filter(.metric == target) |>
    group_by(wflow_id, id) |>
    slice_min(mean, n = 1, with_ties = FALSE) |>
    ungroup() |>
    select(wflow_id, id, .config)
  
  # (second stage: draw threshold) after best_candidate on goal pred, select best t, for each decision rule
  f1_scores |>
    semi_join(best_candidate, by = c("wflow_id", "id", ".config")) |>
    group_by(wflow_id, id, rule) |>
    slice_max(mean, n = 1, with_ties = FALSE) |>
    ungroup() |>
    mutate(target = target)
}

# choose best (candidate, t) per decision rule per outer fold directly based on macro F1
pick_joint = f1_scores |>
  group_by(wflow_id, id, rule) |>
  slice_max(mean, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(target = "macro_f1")

# make a compatible df to fill in for null model
null_choices = tibble(wflow_id = "base_null", id = folds$id, target = "none", rule = "argmax",
                      t = NA, .config = "none", params = list(tibble()), params_label = "", mean = NA)

# the picks: one obs per {wflow_id} x {outer fold} x {target} x {decision rule} (+ null_choices)
choices = bind_rows(pick_two_stage("mae"), 
                    pick_two_stage("poisson_log_loss"), 
                    pick_two_stage("rps"),
                    pick_joint, 
                    null_choices) |>
  # only comparing new AgeFitDiff covariate using the two-stage (stage one = mae) run 
  separate_wider_delim(wflow_id, "_", names = c("covariates", "model_class"), cols_remove = FALSE) |>
  filter(covariates == "base" | target %in% c("mae", "none")) |>
  select(wflow_id, covariates, model_class, id, target, rule, t, .config, params, params_label,
         inner_f1 = mean)

# sanity check using mae results from nested_tune_grid to make sure choices is created properly
nestedtune_picks = nested_res |>
  filter(wflow_id != "base_null") |>
  mutate(chosen = map(result, collect_selections)) |>
  select(wflow_id, chosen) |>
  unnest(chosen)
stopifnot(nrow(anti_join(distinct(filter(choices, target == "mae"), wflow_id, id, .config),
                         nestedtune_picks, by = c("wflow_id", "id", ".config"))) == 0)

# Section 10: Outer predictions for every algorithm----

# context: 
# - we're refitting every pick in "choices"
# - normally this kind of thing would be handled within nested_workflow_map
# - we could theoretically run nested_workflow_map multiple times for each target
# - here instead we avoid rerunning the whole program by refitting each target's picks ourselves
refit_outer = function(wflow_id, outer_id, params, seed = 6769) {
  split = folds$splits[[which(folds$id == outer_id)]]
  wf = extract_workflow(wf_set, wflow_id)
  if (ncol(params) > 0) wf = finalize_workflow(wf, params)
  set.seed(seed)
  fitted = fit(wf, analysis(split))
  tibble(.row  = as.integer(split, data = "assessment"),
         Goals = assessment(split)$Goals,
         .pred = predict(fitted, assessment(split))$.pred)
}

# all the models to refit: one obs per {wflow_id} x {outer fold} x {candidate}
fits = choices |>  # in comparison to "choices": target col - just leads to a candidate; rule col - applied after fitting
  distinct(wflow_id, id, .config, .keep_all = TRUE) |>
  select(wflow_id, id, .config, params)

# only refit if a saved version isn't adequate
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


# essentially duplicates results of refit_outer for each rule and target by joining to choices
to_preds = function(fits) {
  
  # start with every pick
  choices |>
    select(-params) |>
    
    # join with fits that now includes fits$pred created by refit_outer
    inner_join(select(fits, -params), 
               by = c("wflow_id", "id", ".config"), 
               relationship = "many-to-many") |>
    
    # duplicate all columns per row in each pred nested df
    unnest(pred) |>
    
    # add year, match_id, and "algorithm" informational columns
    mutate(.pred = pmax(.pred, 1e-6), # cut off any negative expected goals (nnet)
           Year = euro$Year[.row], 
           match_id = euro$match_id[.row],
           algorithm = paste(wflow_id, target, rule, sep = " | ")) |>
    
    select(algorithm, wflow_id, covariates, model_class, target, rule, id, Year, match_id, .row,
           t, params_label, seed, Goals, .pred)
}

# Goal prediction per team-match for each algorithm: one obs per {algorithm} x {team-match}
preds = to_preds(mutate(fits, seed = 6769))

# sanity check: all team-match combos represented per algorithm
n_test_rows = sum(map_int(folds$splits, \(split) nrow(assessment(split))))
stopifnot(all(count(preds, algorithm)$n == n_test_rows))

# Section 11: Win/draw/loss----
to_wdl = function(preds) {
  preds |>
    arrange(.row) |>
    # prepare match_level expected goals for team1 and team2 for skellam inside wdl_probs
    group_by(algorithm, wflow_id, covariates, model_class, target, rule, seed, Year, match_id) |>
    summarise(t = first(t), 
              lambda1 = first(.pred), 
              lambda2 = last(.pred),
              goals1  = first(Goals), 
              goals2  = last(Goals), 
              .groups = "drop") |>
    mutate(wdl_probs(lambda1, lambda2), # skellam calculation of p_win, p_draw, p_loss
           actual = actual_result(goals1, goals2),
           call   = call_result(p_win, p_draw, p_loss, rule, t)) # call: is based on decision rule
}

# Shows wdl prediction and actual for each algorithm, one obs per {algorithm} x {match}
wdl = to_wdl(preds)

# context: the two baselines (adding historical to wdl)
# - algorithm = "base_null | none | argmax"  (created earlier and incorporated in choices)
#      - uses grand mean as expected goals and then skellam transformation predicts team1 wins every time
# - algorithm = "historical | none | argmax"
#      - use the observed W/D/L ratios in the available training data
all_matches = tibble(Year = euro$Year[odd], actual = actual_result(euro$Goals[odd], euro$Goals[odd + 1]))
training_shares = function(year) {
  split = folds$splits[[which(folds$id == paste("test", year))]]
  prop.table(table(filter(all_matches, Year %in% analysis(split)$Year)$actual))
}
matches = wdl |> distinct(Year, match_id, goals1, goals2, actual)
baseline_wdl = matches |>
  mutate(p_win  = map_dbl(Year, \(y) training_shares(y)[["win"]]),
         p_draw = map_dbl(Year, \(y) training_shares(y)[["draw"]]),
         p_loss = map_dbl(Year, \(y) training_shares(y)[["loss"]]),
         call   = call_result(p_win, p_draw, p_loss, "argmax", NA),
         algorithm = "historical | none | argmax", wflow_id = "historical", covariates = "none",
         model_class = "historical", target = "none", rule = "argmax")
wdl = bind_rows(wdl, baseline_wdl)

stopifnot(all(count(wdl, algorithm)$n == n_test_rows / 2))

# Section 12: Scores----

# Calculate mae, rmse, poisson_nll for goal predictions (one obs per {algorithm}, except historical)
goal_scores = preds |>
  group_by(algorithm, wflow_id, covariates, model_class, target, rule) |>
  summarise(mae         = mean(abs(Goals - .pred)),
            rmse        = sqrt(mean((Goals - .pred)^2)),
            poisson_nll = -mean(dpois(Goals, .pred, log = TRUE)),
            .groups = "drop")

# calculate WDL or probability based metrics (one obs per {algorithm})
match_scores = wdl |>
  group_by(algorithm, wflow_id, covariates, model_class, target, rule) |>
  summarise(
    rps          = mean_rps(p_win, p_draw, actual),
    brier        = mean((p_win - (actual == "win"))^2 + (p_draw - (actual == "draw"))^2 +
                        (p_loss - (actual == "loss"))^2),
    log_loss     = -mean(log(case_when(actual == "win" ~ p_win, actual == "draw" ~ p_draw, TRUE ~ p_loss))),
    accuracy     = mean(call == actual),
    macro_f1     = macro_f1(actual, call),
    macruracy_mean = (accuracy + macro_f1)/2,
    macruracy_h_mean = 2*(accuracy*macro_f1)/(accuracy+macro_f1),
    draw_recall  = sum(call == "draw" & actual == "draw") / sum(actual == "draw"),
    draws_called = sum(call == "draw"),
    draw_f1      = 2 * sum(call == "draw" & actual == "draw") / (draws_called + sum(actual == "draw")), # 0 if no draw called right
    draw_precision = if (draws_called > 0) sum(call == "draw" & actual == "draw") / draws_called else NA_real_, # NA: no draws called
    .groups = "drop")

# scores contains the summary metrics with which to compare: one obs per {algorithm} x {metric}
scores = full_join(goal_scores, match_scores,
                   by = c("algorithm", "wflow_id", "covariates", "model_class", "target", "rule")) |>
  pivot_longer(-c(algorithm, wflow_id, covariates, model_class, target, rule),
               names_to = "metric", values_to = "value")

# Section 13: Seed noise (sensitivity, LOTO only)----

# context: 
# - rf, nnet, and xgboost all change their fits based on seed (poisson and null deterministic)
# - here, we try to approximate how sensitive our results are to outer refit seed changes
#   by rerunning the fits in the leave one tournament out case
# - seed changes during tuning (and their impacts on the picks in choice) aren't being taken into account
SEEDS = 1:5
if (RESAMPLING == "loto") {
  
  # make a copy of fits for each in SEEDS
  seed_fits = fits |>
    filter(!endsWith(wflow_id, "_pois"), wflow_id != "base_null") |>
    select(wflow_id, id, .config, params) |>
    crossing(seed = SEEDS)

  # only refit seeds if a saved version isn't adequate
  SEED_FILE = file.path(RESULTS_DIR, "seed_fits.rds")
  saved_seed_fits = if (file.exists(SEED_FILE)) readRDS(SEED_FILE)
  if (length(to_run) == 0 && !is.null(saved_seed_fits) && identical(select(saved_seed_fits, -pred), seed_fits)) {
    seed_fits = saved_seed_fits
  } else {
    if (isTRUE(get0("LOAD_ONLY"))) stop("LOAD_ONLY: the seed fits are missing or out of date, run this script directly")
    cat("refitting", nrow(seed_fits), "picks with other seeds\n")
    seed_fits$pred = pmap(list(seed_fits$wflow_id, seed_fits$id, seed_fits$params, seed_fits$seed), refit_outer)
    saveRDS(seed_fits, SEED_FILE)
  }

  # reuse to_preds() and to_wdl() to get scores object for each seed
  seed_preds = to_preds(seed_fits)
  seed_wdl   = to_wdl(seed_preds)

  # goal-level scores per seed (as goal_scores in Section 12), joined onto the match-level ones below
  seed_goal_scores = bind_rows(preds, seed_preds) |>
    filter(model_class %in% c("rf", "xg", "nnet")) |>
    group_by(algorithm, wflow_id, covariates, model_class, target, rule, seed) |>
    summarise(mae         = mean(abs(Goals - .pred)),
              rmse        = sqrt(mean((Goals - .pred)^2)),
              poisson_nll = -mean(dpois(Goals, .pred, log = TRUE)),
              .groups = "drop")

  seed_scores = bind_rows(wdl, seed_wdl) |>
    filter(model_class %in% c("rf", "xg", "nnet")) |>
    group_by(algorithm, wflow_id, covariates, model_class, target, rule, seed) |>
    summarise(    rps          = mean_rps(p_win, p_draw, actual),
                  brier        = mean((p_win - (actual == "win"))^2 + (p_draw - (actual == "draw"))^2 +
                                        (p_loss - (actual == "loss"))^2),
                  log_loss     = -mean(log(case_when(actual == "win" ~ p_win, actual == "draw" ~ p_draw, TRUE ~ p_loss))),
                  accuracy     = mean(call == actual),
                  macro_f1     = macro_f1(actual, call),
                  macruracy_mean = (accuracy + macro_f1)/2,
                  macruracy_h_mean = 2*(accuracy*macro_f1)/(accuracy+macro_f1),
                  draw_recall  = sum(call == "draw" & actual == "draw") / sum(actual == "draw"),
                  draws_called = sum(call == "draw"),
                  draw_f1      = 2 * sum(call == "draw" & actual == "draw") / (draws_called + sum(actual == "draw")),
                  draw_precision = if (draws_called > 0) sum(call == "draw" & actual == "draw") / draws_called else NA_real_,
              .groups  = "drop") |>
    left_join(seed_goal_scores, by = c("algorithm", "wflow_id", "covariates", "model_class", "target", "rule", "seed"))
  stopifnot(all(count(seed_scores, algorithm)$n == length(SEEDS) + 1), !anyNA(seed_scores$mae))
}

# seed_summary: one obs per {algorithm} x {metric}: the mean, sd, min and max over the refit seeds
# - rf / xg / nnet (LOTO): over the 6 seeds of seed_scores. Everything else, and every algorithm of the
#     expanding run (no other seeds there): its one value from scores, sd 0
# - draw_precision is NA at a seed that called no draws: those seeds are left out (NA if no seed called one)
# - the default scores to report: seed means are less noisy than the main seed alone
id_cols = c("algorithm", "wflow_id", "covariates", "model_class", "target", "rule")
seed_values = if (RESAMPLING == "loto") {
  bind_rows(seed_scores |> pivot_longer(-all_of(c(id_cols, "seed")), names_to = "metric"),
            scores |> filter(!model_class %in% c("rf", "xg", "nnet")))
} else {
  scores
}
seed_summary = seed_values |>
  group_by(across(all_of(id_cols)), metric) |>
  summarise(mean    = if (all(is.na(value))) NA_real_ else mean(value, na.rm = TRUE),
            sd      = if (sum(!is.na(value)) > 1) sd(value, na.rm = TRUE) else 0,
            min     = if (all(is.na(value))) NA_real_ else min(value, na.rm = TRUE),
            max     = if (all(is.na(value))) NA_real_ else max(value, na.rm = TRUE),
            n_seeds = n(),
            .groups = "drop")

# Section 14: Expanding-window sub-experiment----

# runs the whole code again if expanding window setting is on
if (RESAMPLING == "loto" && RUN_EXPANDING) {
  expanding = new.env()
  local({
    RESAMPLING = "expanding"
    source("combined/football_combined_nested_tune.R", local = TRUE)
  }, envir = expanding)
}

# Section 15: Final models (LOTO only)----

# context:
# - the outer folds estimated how well each algorithm's *procedure* works (tune, then predict a new
#   tournament). The final model runs the chosen procedure once more on all 6 tournaments, like
#   nestedtune::nested_final_fit(), except that the tuning also picks the decision rule's t
# - tuning resamples: the 6 outer folds (leave one tournament out over all the data), weighted by size
# - target "macro_f1" (joint): the (candidate, t) with the best macro F1 over the 6 held-out years
#   two-stage targets (mae, poisson_log_loss, rps): the best candidate on the target, then its best t
#   ties go to the smallest t, then the first candidate (as in Section 9)
# - tuning_macro_f1 is the score the pick was chosen on, so it's optimistic: the honest estimate of
#   how the procedure does on a new tournament is the algorithm's outer score (scores, Section 12)
# - FINALS: the algorithms to fit a final model for, by name. Each is saved to
#   RESULTS_DIR/final_fit_<name>.rds and refit only if its settings, its grid or the folds change
#     pois: the chosen algorithm (point 4 of the draw recall front: most draws caught before draw
#           precision falls, best macro F1 and draw F1, deterministic)
#     rf:   the more consistent alternative (point 3: best worst year, best in the expanding window)
FINALS = list(pois = list(wflow_id = "base_pois", target = "macro_f1", rule = "p_draw"),
              rf   = list(wflow_id = "base_rf",   target = "macro_f1", rule = "range"))

# fit_final(settings, final_wf, final_grid, key): tune one algorithm on the 6 outer folds and refit its
#   pick on all of euro. Returns its settings, params, t, the scores it was picked on and the fitted workflow
fit_final = function(settings, final_wf, final_grid, key) {
  # the 6 outer splits as one plain resampling set, weighted by held-out rows (as in Section 2)
  final_folds = manual_rset(folds$splits, folds$id)
  final_folds = add_resample_weights(final_folds, assessment_weights(final_folds))

  # metrics: macro F1 at every t of the rule, plus the target if it's two-stage
  two_stage     = settings$target != "macro_f1"
  rule_metrics  = filter(rules, rule == settings$rule)
  final_metrics = do.call(metric_set, c(
    if (two_stage) list(mae = mae, poisson_log_loss = poisson_log_loss, rps = rps_metric)[settings$target],
    f1_metrics[rule_metrics$metric_name]))

  set.seed(6769)
  final_tuning = tune_grid(final_wf, resamples = final_folds, grid = final_grid, metrics = final_metrics)

  # one obs per {candidate} x {metric}: the weighted mean over the 6 held-out years
  final_scores = collect_metrics(final_tuning)
  final_f1 = final_scores |>
    inner_join(select(rule_metrics, metric_name, t), by = c(".metric" = "metric_name")) |>
    arrange(t) # so ties -> smallest t
  if (two_stage) {
    best_config = final_scores |>
      filter(.metric == settings$target) |>
      slice_min(mean, n = 1, with_ties = FALSE) |>
      pull(.config)
    final_f1 = filter(final_f1, .config == best_config)
  }
  best        = slice_max(final_f1, mean, n = 1, with_ties = FALSE)
  best_params = select(best, all_of(extract_parameter_set_dials(final_wf)$id))

  set.seed(6769) # rf is random: the same seed as the outer refits (Section 10)
  list(key             = key,
       settings        = settings,
       params          = best_params,
       t               = best$t,
       tuning_macro_f1 = best$mean,
       tuning_scores   = final_scores,
       fit             = fit(finalize_workflow(final_wf, best_params), euro))
}

# predict_final(new_data, final_model): one obs per {match}: both lambdas, the win / draw / loss
#   probabilities and the call under the model's rule and t, e.g. predict_final(new_data, finals$pois)
#   new_data: like euro, the two rows of a match adjacent (the team's view, then the opponent's view)
predict_final = function(new_data, final_model) {
  stopifnot(nrow(new_data) %% 2 == 0)
  lambda = pmax(predict(final_model$fit, new_data)$.pred, 1e-6)
  team1  = seq(1, nrow(new_data), by = 2)
  p      = wdl_probs(lambda[team1], lambda[team1 + 1])
  tibble(Team = new_data$Team[team1], Opponent = new_data$Opponent[team1],
         lambda1 = lambda[team1], lambda2 = lambda[team1 + 1], p,
         call = call_result(p$p_win, p$p_draw, p$p_loss, final_model$settings$rule, final_model$t))
}

if (RESAMPLING == "loto") {
  # finals: one final model per entry of FINALS (by the same name), loaded if up to date, else fitted.
  #   Under LOAD_ONLY a missing or stale one is NULL (not needed to load the results, so no stop)
  finals = imap(FINALS, \(settings, name) {
    stopifnot(settings$wflow_id %in% wf_set$wflow_id, settings$rule %in% rules$rule,
              settings$target %in% c("mae", "poisson_log_loss", "rps", "macro_f1"))
    file       = file.path(RESULTS_DIR, paste0("final_fit_", name, ".rds"))
    final_wf   = extract_workflow(wf_set, settings$wflow_id)
    final_grid = wf_set$option[[which(wf_set$wflow_id == settings$wflow_id)]]$grid
    key        = list(settings, final_grid, folds$id)
    saved      = if (file.exists(file)) readRDS(file)
    if (length(to_run) == 0 && !is.null(saved) && identical(saved$key, key)) return(saved)
    if (isTRUE(get0("LOAD_ONLY"))) {
      message("LOAD_ONLY: the final model ", name, " is missing or out of date, run this script directly to fit it")
      return(NULL)
    }
    cat("fitting the final model:", name, "\n")
    final = fit_final(settings, final_wf, final_grid, key)
    saveRDS(final, file)
    final
  })

  for (name in names(finals)) {
    final = finals[[name]]
    if (is.null(final)) next
    cat("\nfinal model ", name, ": ", paste(final$settings, collapse = " | "), "\n", sep = "")
    print(final$params)
    cat("t =", final$t, "  macro F1 it was picked on (6 held-out years, optimistic):",
        round(final$tuning_macro_f1, 3), "\n")
    # what each outer fold picked for the same algorithm (Section 9): how stable the picks were
    choices |>
      filter(wflow_id == final$settings$wflow_id, target == final$settings$target, rule == final$settings$rule) |>
      select(id, params_label, t, inner_f1) |>
      print()
  }
}
