# README: now that we've each tried to get some results about nnet, random forest, xgboost, and poisson 
# separately, this script tries to replicate it using a more streamlined/organized high-level R packages
# namely parsnip models, workflows, and nestedtune


# HIGHLEVEL 

# Important objects: 
#   - resample_struct
#   - metrics_sel_mse, metrics_sel_rps, metrics_sel_skel (Section 4.3)
#   - RUN_* switches and the naming scheme (Section 9)
#   - <model_class>_<select>_nested / _final (Section 10)
#   - pois_results (Section 13)

# Section 0: Global settings-----

SIMPLE = F

# Section 1: load libraries---------
library(tune)
library(readr)
library(rsample)
library(poissonreg)
library(yardstick)
library(purrr)
library(dials)
library(recipes)
library(workflows)
library(tidyr)
library(dplyr)
library(knitr)
# install.packages("pak")
# pak::pak("tidymodels/nestedtune")
library(nestedtune)
library(lubridate)
library(skellam)
library(mirai)
library(tailor)
library(caret)
library(ggplot2)

# Section 2: Read in data ----
euro_2004 = read_csv("raw-data/euro_2004.csv")
euro_2008 = read_csv("raw-data/euro_2008.csv")
euro_2012 = read_csv("raw-data/euro_2012.csv")
euro_2016 = read_csv("raw-data/euro_2016.csv")
euro_2020 = read_csv("raw-data/euro_2020.csv")
euro_2024 = read_csv("raw-data/euro_2024_fixed.csv")

euro = rbind(euro_2004,
             euro_2008,
             euro_2012,
             euro_2016,
             euro_2020,
             euro_2024) 

# Section 3: Form splits----
# NOTE: Look for where resample_struct is defined. That's the ultimate split that's used below
set.seed(67676769)


# Section 3.1 nested_leave_one_year object----
nested_leave_one_year = nested_resamples(data = euro,
                                         outside = group_vfold_cv(v = NULL, 
                                                                  group = Year,
                                                                  repeats = 1,
                                                                  strata = NULL),
                                         inside = group_vfold_cv(v = NULL, 
                                                                 group = Year,
                                                                 repeats = 1,
                                                                 strata = NULL))

all_years <- seq(2004, 2024, 4)

# Rename an rset and its underlying split objects
rename_rset_ids <- function(x, new_ids) {
  stopifnot(length(new_ids) == nrow(x))
  
  x$id <- new_ids
  
  for (i in seq_along(x$splits)) {
    x$splits[[i]]$id$id <- new_ids[i]
  }
  
  x
}

get_resample_ids <- function(resamples, all_years, outer_test_year = NULL) {
  
  purrr::map_chr(resamples$splits, function(split) {
    
    missing_years <- setdiff(
      all_years,
      unique(analysis(split)$Year)
    )
    
    if (is.null(outer_test_year)) {
      stopifnot(length(missing_years) == 1)
      paste("test", missing_years)
      
    } else {
      eval_year <- setdiff(missing_years, outer_test_year)
      
      stopifnot(length(eval_year) == 1)
      
      paste0("test ", outer_test_year, " / eval ", eval_year)
    }
  })
}

# Rename outer resamples
nested_leave_one_year <- rename_rset_ids(
  nested_leave_one_year,
  get_resample_ids(nested_leave_one_year, all_years)
)

# Rename inner resamples
for (i in seq_along(nested_leave_one_year$inner_resamples)) {
  
  outer_test_year <- setdiff(
    all_years,
    unique(analysis(nested_leave_one_year$splits[[i]])$Year)
  )
  
  inner <- nested_leave_one_year$inner_resamples[[i]]
  
  inner_names <- get_resample_ids(
    inner,
    all_years,
    outer_test_year
  )
  
  nested_leave_one_year$inner_resamples[[i]] <- rename_rset_ids(
    inner,
    inner_names
  )
}

# Resample weights so fold averages equal pooled values (metric computed once on all held-out rows)
# Weight = number of HELD-OUT (assessment) rows. NOT calculate_resample_weights(): that weights by
# training-set size, which gives the small 62-row years MORE weight than the 102-row years
assessment_weights = function(rset) {
  n = purrr::map_int(rset$splits, \(split) nrow(assessment(split)))
  n / sum(n)
}

nested_leave_one_year = add_resample_weights(
  rset = nested_leave_one_year, 
  weights = assessment_weights(nested_leave_one_year)
) 

for (i in 1:6) { # inner weights are used by tune when averaging inner metrics for selection
  nested_leave_one_year$inner_resamples[[i]] = add_resample_weights(
    rset = nested_leave_one_year$inner_resamples[[i]], 
    weights = assessment_weights(nested_leave_one_year$inner_resamples[[i]])
  ) 
}

# Section 3.2 nested expanding window (sliding period)----
euro %>% mutate(Year = lubridate::make_date(year = Year, 
                                            month = 1, 
                                            day = 1)) -> euro_date

nested_expaning_window = nested_resamples(
  data = euro_date,
  outside = sliding_period(
    lookback = Inf, # makes expanding window
    period = "year", 
    index = Year,
    every = 4
  ),
  inside = sliding_period(
    lookback = Inf, # makes expanding window
    period = "year", 
    index = Year,
    every = 4
  )
) %>% filter(id != "Slice1") # takes out first instance where no validation available

# Section 3.3 Choose the splitting method----
resample_struct = nested_leave_one_year

# Section 4: Defining custom groupwise metrics ----

# Section 4.1: skellam negative log likelihood metric-----
skellam_nll <- function(data, truth, estimate, na_rm = TRUE, ...) {
  
  # defined inside so it travels to mirai daemons with the metric
  skellam_nll_vec <- function(truth, estimate, na_rm = TRUE, ...) {
    if (length(truth) != 2 || length(estimate) != 2) {
      stop("Expected exactly two observations per match.")
    }
    if (na_rm && anyNA(c(truth, estimate))) {
      return(NA_real_)
    }
    
    # negative log likelihood
    -skellam::dskellam(
      x = truth[1] - truth[2],# actual skellam value
      lambda1 = estimate[1], # probability of the truth, given our estimate
      lambda2 = estimate[2],
      log = TRUE
    )
  }
  
  yardstick::numeric_metric_summarizer(
    name = "skellam_nll",
    fn = skellam_nll_vec,
    data = data,
    truth = !!rlang::enquo(truth),
    estimate = !!rlang::enquo(estimate),
    na_rm = na_rm
  )
}

# have skellam_nll inherit properties of a yardstick metric
skellam_nll <- yardstick::new_numeric_metric(
  skellam_nll,
  direction = "minimize"
)

# we tell yardstick how our custom skellam_nll works across groups
skellam_nll_by_match <- yardstick::new_groupwise_metric(
  fn = skellam_nll,
  name = "skellam_nll_by_match", # must equal the object name, or selecting on it fails
  aggregate = function(x) mean(x$.estimate, na.rm = TRUE),
  direction = "minimize"
)

# tells new_groupwise_metric which variable to group_by
skellam_nll_by_match <- skellam_nll_by_match(match_id)

# Section 4.1.1: why we tune on yardstick::mse instead of rmse-----
  # Pooled vs fold-weighted: nestedtune/tune average fold metrics weighted by the resample weights
  # (assessment-row weights, Section 3.1).
  # For metrics that are a mean over rows (mse, mae, poisson_log_loss, skellam_nll, wdl_*)
  # that weighted average IS the pooled value (metric computed once on all held-out rows).
  # rmse isn't: the sqrt is taken per fold before averaging. mse is a mean over rows, so it
  # pools exactly, and argmin pooled mse == argmin pooled rmse (sqrt is monotone).
  # Pooled rmse is reported after the fact from the outer predictions (goal_scores, Section 11).



# Section 4.2: Implementing yardstick compatible classification metrics all at once----
  # Basically, "tune" package didn't like I was using poisson "regression" based parsnip model 
  # and mixing with classification based metrics (so here had to make my own)
  # WDL metrics: each sees one match at a time (2 rows, grouped by match_id)
  #   truth    = c(goals team 1, goals team 2)
  #   estimate = c(lambda team 1, lambda team 2)
  # WDL is from the first row's perspective

# helpers live in the make_wdl_vec def so they get shipped to mirai daemons
# concept is that make_wdl_vec is a function factory to reduce repeated code among metric defs
make_wdl_vec <- function(score = c("accuracy", "brier", "rps"), draw_range = 0) {
  score <- match.arg(score) # makes sure score is one of provided arguments (defaults to accuracy)
  force(draw_range) # storing draw_range at time of factory make_wdl_vec call rather than it getting manipulated

  # P(win), P(draw), P(loss) for team 1 from a Skellam(lambda1, lambda2)
  skellam_wdl_probs <- function(lambda1, lambda2) {
    p <- c(
      win  = skellam::pskellam(q = 0,  lambda1 = lambda1, lambda2 = lambda2, lower.tail = FALSE),
      draw = skellam::dskellam(x = 0,  lambda1 = lambda1, lambda2 = lambda2),
      loss = skellam::pskellam(q = -1, lambda1 = lambda1, lambda2 = lambda2, lower.tail = TRUE)
    )
    p / sum(p)
  }

  actual_wdl <- function(goals) {
    if (goals[1] > goals[2]) "win" else if (goals[1] == goals[2]) "draw" else "loss"
  }

  function(truth, estimate, na_rm = TRUE, ...) {
    
    # catch bad inputs
    if (length(truth) != 2 || length(estimate) != 2) {
      stop("Expected exactly two observations per match.")
    }
    if (na_rm && anyNA(c(truth, estimate))) {
      return(NA_real_)
    }

    p      <- skellam_wdl_probs(estimate[1], estimate[2])
    actual <- actual_wdl(truth)
    y      <- as.numeric(names(p) == actual)   #True/False (1/0) if correctly predicted or not

    # based on what score argument calculate the desired metric
    switch(
      score,
      
      # argmax pick, or "draw" when the three probabilities are within draw_range
      accuracy = {
        pred_wdl <- names(p)[which.max(p)]
        # TODO: range implemented accuracy (change if we want threshold or raw)
        if (max(p) - min(p) < draw_range) pred_wdl <- "draw"
        as.numeric(pred_wdl == actual)
      },
      
      # multiclass brier: sum of squared errors over win/draw/loss
      brier = sum((p - y)^2),
      
      # ranked probability score: respects win > draw > loss ordering
      # RPS essentially avg square penalized errors in cumulative probabilities (final)
      # imagine "p <- c(win = 0.6, draw = 0.3, loss = 0.1)"
      # cum_win_v_draw_draw_v_loss = (cumsum(p) - cumsum(y))[1:2] 
      rps = sum((cumsum(p) - cumsum(y))[1:2]^2) / 2
    )
  }
}

# builds the yardstick metric + groupwise-by-match version in one go
make_wdl_metric <- function(name, score, direction, draw_range = 0) {

  vec_fn <- make_wdl_vec(score, draw_range)

  metric_fn <- function(data, truth, estimate, na_rm = TRUE, ...) {
    # https://yardstick.tidymodels.org/reference/metric_summarizer.html
    yardstick::numeric_metric_summarizer(
      name = name,
      fn = vec_fn,
      data = data,
      truth = !!rlang::enquo(truth),   #tidyverse way of accepting column names and !! evals Goals now
      estimate = !!rlang::enquo(estimate),
      na_rm = na_rm
    )
  }
  metric_fn <- yardstick::new_numeric_metric(metric_fn, direction = direction)

  by_match <- yardstick::new_groupwise_metric(
    fn = metric_fn,
    name = name,
    aggregate = function(x) mean(x$.estimate, na.rm = TRUE),
    direction = direction
  )
  by_match(match_id)
}

# object names must equal the metric name: nestedtune checks selection_rule()
# against object names, tune matches on the .metric column
wdl_accuracy       = make_wdl_metric("wdl_accuracy",       "accuracy", "maximize")
wdl_accuracy_range = make_wdl_metric("wdl_accuracy_range", "accuracy", "maximize", draw_range = .15)
wdl_brier          = make_wdl_metric("wdl_brier",          "brier",    "minimize")
wdl_rps            = make_wdl_metric("wdl_rps",            "rps",      "minimize")

# Section 4.3: Declare metrics for nested_grid_tune----
# selection_rule("best") ranks on the FIRST metric of the set, so each set leads with its selection metric
# (selection_rule("desirability", minimize(wdl_rps)) errors: desirability can't see the custom groupwise metrics)
# Sets are lean on purpose: grouped metrics add run time, and all WDL / skellam reporting is done
# afterwards from the saved predictions (Sections 11-13)
# mse rather than mae: Skellam needs each team's MEAN goals (lambda); mse is minimized by the mean,
# mae by the median. mse rather than rmse so fold-weighted averaging is exactly pooled (Section 4.1.1)
metrics_sel_mse = yardstick::metric_set(
  mse,
  mae,
  poisson_log_loss
)

metrics_sel_rps = yardstick::metric_set(
  wdl_rps,
  mae,
  mse,
  poisson_log_loss
)

# skellam_nll_by_match first (selection), plus wdl_rps and mse so this one run has all three
# selection criteria computed on exactly the same fits (Section 15.1)
metrics_sel_skel = yardstick::metric_set(
  skellam_nll_by_match,
  wdl_rps,
  mse,
  mae,
  poisson_log_loss
)

# Full sets, e.g. for compute_metrics() on a finished run
metrics_wdl = yardstick::metric_set(
  wdl_rps,
  wdl_brier,
  wdl_accuracy,
  wdl_accuracy_range
)

metrics_goals = yardstick::metric_set(
  mse,
  mae,
  poisson_log_loss,
  skellam_nll_by_match
)

# Section 5: Poisson Workflow & Grid----
pois_tune <- poisson_reg(engine = "glmnet", 
                             penalty = tune(), 
                             mixture = tune())

pois_grid = grid_regular(penalty(), 
                         mixture(), 
                         levels = 7)

pois_rec = recipe(
  data = euro, 
  Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers + match_id) |>
  update_role(match_id, new_role = "id")

pois_wf = workflow() |>
  add_model(pois_tune) |> 
  add_recipe(pois_rec)

# Section 6: Defining Neural Net Workflow & Grid -----

nnet_tune = mlp(engine = "nnet", 
                mode = "regression",
                penalty = tune(), 
                hidden_units = tune()
)

nnet_grid = expand.grid(
  penalty = c(0.5, 
              0.7,
              0.9,
              1.0,
              1.1,
              2.0), # TODO think about whether to include much smaller values
  hidden_units = 1:5
)
  
nnet_wf = workflow() |>
  add_model(nnet_tune) |> 
  add_formula(Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers)

# Section 7: Defining Random Forest Workflow & Grid -----

# Settings
target <- "Goals"
predictors <- c(
  "GroupStage",
  "GDP",
  "MarketValue",
  "FifaRank",
  "UefaPoints",
  "CLPlayers"
)
N_TREES = 1000


rf_formula <- as.formula(
  paste(target, "~", paste(predictors, collapse = " + "))
)

rf_tune = rand_forest(
  min_n = tune(), 
  mtry = tune(), 
  trees= !!N_TREES
) %>%
  set_engine(
    "ranger", 
    sample.fraction = tune(), 
    num.threads = 1) %>%
  set_mode("regression")


# FULL
rf_grid <- expand.grid(
  mtry            = 1:length(predictors),
  min_n           = c(3, 5, 10, 15),
  sample.fraction = c(0.7, 0.85, 1.0)
)

# option to reduce runtime when debugging (overrides FULL)
# NOTE: N_TREES here comes after rf_tune is built, so it doesn't change the trees
if (SIMPLE == T) {
  N_TREES = 100
  
  rf_grid <- expand.grid(
    mtry            = 3:length(predictors),
    min_n           = c(3, 5), 
    sample.fraction = c(0.7, 0.85)
  )
}

rf_wf = workflow() |>
  add_model(rf_tune) |> 
  add_formula(rf_formula)

# Section 8: Defining XGBoost Workflow & Grid-----

xg_tune <- boost_tree(
  tree_depth = tune(), 
  learn_rate = tune(), 
  sample_size = tune(), 
  min_n = tune()) |>                           # similar-ish to min_child_weight
  set_engine("xgboost",
             counts = FALSE,                   # changes colsample_bytree
             # min_child_weight = tune(),      # not possible
             eval_metric = "poisson-nloglik", 
             objective =  "count:poisson",
             colsample_bytree = 0.8) |> 
  set_mode("regression")

xg_grid <- expand.grid(
  # FULL
  tree_depth    = c(1, 2, 3),
  learn_rate    = c(0.01, 0.05, 0.1), # assuming equivalent to eta
  sample_size   = c(0.5, 0.8),        # assuming equivalent to subsample
  min_n         = c(1, 5, 10)
)

if (SIMPLE){
  xg_grid <- expand.grid(
    tree_depth       = c(1, 2),
    learn_rate       = c(0.01, 0.05), # assuming equivalent to eta
    sample_size      = c(0.5, 0.8),   # assuming equivalent to subsample
    min_n            = c(1, 5, 10)
  )
}

xg_wf = workflow() |>
  add_model(xg_tune) |> 
  add_formula(Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers)

# Section 9: Experiment settings----

# Naming scheme: <model_class>_<selection metric>_<stage>
#   pois_mse_nested        nested_tune_grid result, hyperparameters selected on mse (old pois_tuned-style object)
#   pois_skel_nested       same, selected on the skellam negative log likelihood of each match
#   pois_rps_final         nested_final_fit: tuning pass on all 6 years + fit on all data (for 2028)
#   pois_mse_preds_outer   goal predictions from the nested outer folds (honest)
#   pois_mse_preds_tuning  goal predictions from the tuning pass at the chosen config (optimistic,
#                          comparable to Random_Forest_LOTOCV.Rmd and xgboost_with_skellam.R)
#   pois_mse_wdl_outer     match-level win/draw/loss built from preds_outer, with tuned draw threshold
# To add rf / xg / nnet: copy a Section 10 block and rename pois -> rf etc.

# TRUE = refit and overwrite results/<name>.rds, FALSE = load the last saved fit (no refit)
RUN_POIS_MSE = FALSE
RUN_POIS_RPS = FALSE
RUN_POIS_SKEL = FALSE

# debug runs (SIMPLE = T) save elsewhere so they can never overwrite the full-grid fits
# frozen copy of the full run: results_final_2026-11-11 (read-only)
RESULTS_DIR = if (SIMPLE) "results_debug" else "results"
PLOT_DIR    = file.path(RESULTS_DIR, "plots")
dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

# save a plot to results/plots/<name>.png, and show it when running interactively (RStudio)
# (printing under Rscript would dump every plot into a stray Rplots.pdf instead)
save_plot = function(plot, name, width = 8, height = 5) {
  if (interactive()) print(plot)
  ggsave(file.path(PLOT_DIR, paste0(name, ".png")), plot, width = width, height = height, bg = "white")
  invisible(plot)
}

if (SIMPLE) {
  pois_grid = grid_regular(penalty(), mixture(), levels = 2) # 4 candidates for quick debugging
}

# Shared by every nested run
nested_control = control_grid(
  verbose = FALSE,
  extract = NULL,   # can save other intermediates from the nested cv & call collect_extracts()
  save_pred = TRUE  # needed for every pooled / WDL result below. The inner run's predictions are still discarded
)

# Wrappers: run with 2 mirai daemons and record when it started and how long it took
nested_tune_grid_mirai = function(...) {
  mirai::daemons(2) # 0 vs 2 makes a difference 3,4, no real difference.
  on.exit(mirai::daemons(0))

  started_at = Sys.time()
  elapsed = system.time({tuned = nested_tune_grid(...)})[["elapsed"]]
  cat("nested_tune_grid elapsed:", round(elapsed), "s\n")

  attr(tuned, "started_at")  = started_at
  attr(tuned, "elapsed_sec") = elapsed
  return(tuned)
}

nested_final_fit_mirai = function(...) {
  mirai::daemons(2)
  on.exit(mirai::daemons(0))

  started_at = Sys.time()
  elapsed = system.time({final = nested_final_fit(...)})[["elapsed"]]
  cat("nested_final_fit elapsed:", round(elapsed), "s\n")

  attr(final, "started_at")  = started_at
  attr(final, "elapsed_sec") = elapsed
  return(final)
}

#Explanation of nested_final_fit:
# for context nested_tune_grid scored the whole procedure:
    #- across all combos of hyperparams
    #- using our selection rule
    #- using our model class & formula
    # nested_tune_grid produced 6 model fits and produced the score (e.g. collect_metrics(pois_mse_nested))
    # the reason we do this is so we can compare across model_classes
        # (and potentially selection rules determining hyperparameters)

# nested_final_fit:
  # We're running our procedure on the outer fold structure only to find the best
  # hyperparameters for the final model we'd use to predict 2028
      # `results` argument is telling us the grid & eval metrics & data
      # `object` argument is telling us the model_class & formula
  # Other members of the group refer to this sometimes as "tuning pass" of the final model
  # (!!) Finally, the function uses those hyperparameters to fit on all the data available
  # Its tuning-pass metrics are optimistically biased: every year helped choose the winner

# Section 10: Running the experiments----

# Section 10.1: pois, hyperparameters selected on mse----
if (RUN_POIS_MSE) {
  pois_mse_nested = nested_tune_grid_mirai(          # full run: ~121s
    object     = pois_wf,
    resamples  = resample_struct,
    grid       = pois_grid,
    metrics    = metrics_sel_mse,
    select     = selection_rule(rule = "best"),     # best on mse (first in metrics_sel_mse)
    control    = nested_control
  )
  pois_mse_final = nested_final_fit_mirai(object = pois_wf, results = pois_mse_nested)

  saveRDS(list(nested = pois_mse_nested, final = pois_mse_final),
          file.path(RESULTS_DIR, "pois_mse.rds"))
} else {
  pois_mse_saved  = readRDS(file.path(RESULTS_DIR, "pois_mse.rds"))
  pois_mse_nested = pois_mse_saved$nested
  pois_mse_final  = pois_mse_saved$final
}

# Section 10.2: pois, hyperparameters selected on wdl_rps----
if (RUN_POIS_RPS) {
  pois_rps_nested = nested_tune_grid_mirai(          # full run: ~125s (old pois_wdl with 4 wdl metrics: ~240s)
    object     = pois_wf,
    resamples  = resample_struct,
    grid       = pois_grid,
    metrics    = metrics_sel_rps,
    select     = selection_rule(rule = "best"),     # best on wdl_rps (first in metrics_sel_rps)
    control    = nested_control
  )
  pois_rps_final = nested_final_fit_mirai(object = pois_wf, results = pois_rps_nested)

  saveRDS(list(nested = pois_rps_nested, final = pois_rps_final),
          file.path(RESULTS_DIR, "pois_rps.rds"))
} else {
  pois_rps_saved  = readRDS(file.path(RESULTS_DIR, "pois_rps.rds"))
  pois_rps_nested = pois_rps_saved$nested
  pois_rps_final  = pois_rps_saved$final
}

# Section 10.3: pois, hyperparameters selected on skellam negative log likelihood----
if (RUN_POIS_SKEL) {
  pois_skel_nested = nested_tune_grid_mirai(         # full run: ~170s
    object     = pois_wf,
    resamples  = resample_struct,
    grid       = pois_grid,
    metrics    = metrics_sel_skel,
    select     = selection_rule(rule = "best"),     # best on skellam_nll_by_match (first in metrics_sel_skel)
    control    = nested_control
  )
  pois_skel_final = nested_final_fit_mirai(object = pois_wf, results = pois_skel_nested)

  saveRDS(list(nested = pois_skel_nested, final = pois_skel_final),
          file.path(RESULTS_DIR, "pois_skel.rds"))
} else {
  pois_skel_saved  = readRDS(file.path(RESULTS_DIR, "pois_skel.rds"))
  pois_skel_nested = pois_skel_saved$nested
  pois_skel_final  = pois_skel_saved$final
}

# Section 10.4: Run times----
# (selections, final hyperparameters and nested metrics are reported in Sections 13-15)
pois_timing = tibble(
  experiment  = c("pois_mse_nested", "pois_mse_final", "pois_rps_nested", "pois_rps_final",
                  "pois_skel_nested", "pois_skel_final"),
  started_at  = do.call(c, map(list(pois_mse_nested, pois_mse_final, pois_rps_nested, pois_rps_final,
                                    pois_skel_nested, pois_skel_final),
                               \(x) attr(x, "started_at"))),
  elapsed_sec = map_dbl(list(pois_mse_nested, pois_mse_final, pois_rps_nested, pois_rps_final,
                             pois_skel_nested, pois_skel_final),
                        \(x) attr(x, "elapsed_sec"))
)
pois_timing

# example use 2028
# pois_final_wf = extract_workflow(pois_rps_final)
# predict(pois_final_wf, new_data = euro_2028)

# Section 11: Helper functions for predictions & win/draw/loss----

# Tuning-pass predictions of the config the final fit actually chose
# (the tuning pass keeps predictions for every candidate in the grid)
get_tuning_preds = function(final) {
  extract_tune_results(final) |>
    collect_predictions() |>
    semi_join(final$selected, by = ".config")
}

# Goal predictions -> one row per match with Skellam win/draw/loss probabilities
# collect_predictions drops match_id and Year, so they're recovered from euro via .row
# team 1 = first row of the match, same perspective as the wdl_* metrics
convert_preds_to_wdl = function(preds) {
  lvls = c("win", "draw", "loss")
  wdl = preds |>
    mutate(match_id = euro$match_id[.row], Year = euro$Year[.row]) |>
    arrange(.row) |>
    group_by(Year, match_id) |>
    summarise(
      n_rows  = n(),
      lambda1 = first(.pred), lambda2 = last(.pred),
      goals1  = first(Goals), goals2  = last(Goals),
      .groups = "drop"
    ) |>
    mutate(
      p_win   = skellam::pskellam(q = 0,  lambda1, lambda2, lower.tail = FALSE),
      p_draw  = skellam::dskellam(x = 0,  lambda1, lambda2),
      p_loss  = skellam::pskellam(q = -1, lambda1, lambda2, lower.tail = TRUE),
      p_range = pmax(p_win, p_draw, p_loss) - pmin(p_win, p_draw, p_loss),
      pred_wdl_raw = factor(lvls[max.col(cbind(p_win, p_draw, p_loss), ties.method = "first")],
                            levels = lvls),
      actual_wdl = factor(case_when(
        goals1 >  goals2 ~ "win",
        goals1 == goals2 ~ "draw",
        TRUE             ~ "loss"
      ), levels = lvls)
    )
  stopifnot(all(wdl$n_rows == 2), nrow(wdl) == nrow(euro) / 2) # one row per match, no duplicates
  wdl
}

# Call a draw whenever the three probabilities are within t of each other, otherwise argmax
# t can be a single value or one per match
pred_at_threshold = function(wdl, t) {
  factor(if_else(wdl$p_range < t, "draw", as.character(wdl$pred_wdl_raw)),
         levels = levels(wdl$actual_wdl))
}

# Macro F1: F1 for each class (win, draw, loss), then the plain average, so draws count as much
# as wins. A class that is never predicted gets F1 = 0
macro_f1 = function(truth, estimate) {
  f1 = sapply(levels(truth), function(k) {
    tp = sum(estimate == k & truth == k)
    fp = sum(estimate == k & truth != k)
    fn = sum(estimate != k & truth == k)
    if (2 * tp + fp + fn == 0) 0 else 2 * tp / (2 * tp + fp + fn)
  })
  mean(f1)
}

DRAW_GRID = seq(0, 0.30, by = 0.025)

# Threshold with the highest macro F1 on the given matches (which.max = smallest t on ties)
best_draw_t = function(wdl, grid = DRAW_GRID) {
  f1 = sapply(grid, function(t) macro_f1(wdl$actual_wdl, pred_at_threshold(wdl, t)))
  grid[which.max(f1)]
}

# Leave-one-tournament-out threshold: each year's draws are called with the t that was best
# on the OTHER years, so the score at that t is honest (unlike the Rmd's section 14)
tune_draw_range = function(wdl, grid = DRAW_GRID) {
  years = sort(unique(wdl$Year))
  t_by_year = sapply(years, function(y) best_draw_t(filter(wdl, Year != y), grid))
  names(t_by_year) = years

  wdl |>
    mutate(
      draw_t     = unname(t_by_year[as.character(Year)]),
      pred_wdl_t = pred_at_threshold(wdl, draw_t)
    )
}

# Pooled win/draw/loss scores over all matches
#   rps, brier, log_loss: use the probabilities, so they don't depend on the threshold
#   skellam_nll: -log P(actual goal difference), the skellam_nll_by_match metric pooled
#                (NA for the baseline, which has no goal predictions)
#   *_raw: plain argmax,  *_t: leave-one-tournament-out threshold from tune_draw_range
wdl_scores = function(wdl) {
  y_win  = as.numeric(wdl$actual_wdl == "win")
  y_draw = as.numeric(wdl$actual_wdl == "draw")
  y_loss = as.numeric(wdl$actual_wdl == "loss")
  p_true = wdl$p_win * y_win + wdl$p_draw * y_draw + wdl$p_loss * y_loss
  skel_nll = if ("lambda1" %in% names(wdl)) {
    -mean(skellam::dskellam(wdl$goals1 - wdl$goals2, wdl$lambda1, wdl$lambda2, log = TRUE))
  } else NA_real_

  tibble(
    metric = c("rps", "brier", "log_loss", "skellam_nll",
               "accuracy_raw", "accuracy_t", "macro_f1_raw", "macro_f1_t"),
    value  = c(
      # same RPS as wdl_rps: squared errors of the two cumulative probabilities, / 2
      mean(((wdl$p_win - y_win)^2 + (wdl$p_win + wdl$p_draw - y_win - y_draw)^2) / 2),
      mean((wdl$p_win - y_win)^2 + (wdl$p_draw - y_draw)^2 + (wdl$p_loss - y_loss)^2),
      -mean(log(p_true)),
      skel_nll,
      mean(wdl$pred_wdl_raw == wdl$actual_wdl),
      mean(wdl$pred_wdl_t   == wdl$actual_wdl),
      macro_f1(wdl$actual_wdl, wdl$pred_wdl_raw),
      macro_f1(wdl$actual_wdl, wdl$pred_wdl_t)
    )
  )
}

# Pooled goal-level scores: computed once over all held-out rows rather than averaged per fold,
# so rmse here is the true pooled rmse (see Section 4.1.1). No std_err: one pooled number
goal_scores = function(preds, metrics = metric_set(rmse, mae, poisson_log_loss), check_rows = TRUE) {
  if (check_rows) stopifnot(nrow(preds) == nrow(euro)) # one prediction per row, no duplicates
  metrics(preds, truth = Goals, estimate = .pred) |>
    select(metric = .metric, value = .estimate)
}

# Baseline: no model. For each held-out year, P(win), P(draw), P(loss) = how often team 1
# won / drew / lost in the OTHER years' matches. Every match in that year gets the same probabilities.
# Returns the same columns as convert_preds_to_wdl, so tune_draw_range and wdl_scores work on it
# team 1 = first row of the match in euro, same as everywhere else
# leave_out = FALSE uses all years' frequencies (in-sample, to sit beside the tuning-pass numbers)
baseline_wdl = function(leave_out = TRUE) {
  lvls = c("win", "draw", "loss")
  matches = euro |>
    mutate(.row = row_number()) |>
    arrange(.row) |>
    group_by(Year, match_id) |>
    summarise(goals1 = first(Goals), goals2 = last(Goals), .groups = "drop") |>
    mutate(actual_wdl = factor(case_when(
      goals1 >  goals2 ~ "win",
      goals1 == goals2 ~ "draw",
      TRUE             ~ "loss"
    ), levels = lvls))

  wdl = map(sort(unique(matches$Year)), function(y) {
    train_freq = prop.table(table(matches$actual_wdl[matches$Year != y | !leave_out]))
    matches |>
      filter(Year == y) |>
      mutate(p_win  = train_freq[["win"]],
             p_draw = train_freq[["draw"]],
             p_loss = train_freq[["loss"]])
  }) |>
    bind_rows() |>
    mutate(
      p_range = pmax(p_win, p_draw, p_loss) - pmin(p_win, p_draw, p_loss),
      pred_wdl_raw = factor(lvls[max.col(cbind(p_win, p_draw, p_loss), ties.method = "first")],
                            levels = lvls)
    )
  stopifnot(nrow(wdl) == nrow(euro) / 2)
  wdl
}

# Goal-level baseline to match: predict the mean goals of the other years for every row
baseline_preds = function() {
  euro |>
    mutate(.row = row_number()) |>
    group_by(Year) |>
    mutate(.pred = map_dbl(Year, \(y) mean(euro$Goals[euro$Year != y]))) |>
    ungroup() |>
    select(Goals, .pred, .row)
}

# Section 12: pois predictions & win/draw/loss----

# Section 12.1: Goal predictions----
pois_mse_preds_outer  = collect_predictions(pois_mse_nested)
pois_mse_preds_tuning = get_tuning_preds(pois_mse_final)
pois_rps_preds_outer  = collect_predictions(pois_rps_nested)
pois_rps_preds_tuning = get_tuning_preds(pois_rps_final)
pois_skel_preds_outer  = collect_predictions(pois_skel_nested)
pois_skel_preds_tuning = get_tuning_preds(pois_skel_final)

# Section 12.2: Win/draw/loss with leave-one-tournament-out draw threshold----
pois_mse_wdl_outer  = pois_mse_preds_outer  |> convert_preds_to_wdl() |> tune_draw_range()
pois_mse_wdl_tuning = pois_mse_preds_tuning |> convert_preds_to_wdl() |> tune_draw_range()
pois_rps_wdl_outer  = pois_rps_preds_outer  |> convert_preds_to_wdl() |> tune_draw_range()
pois_rps_wdl_tuning = pois_rps_preds_tuning |> convert_preds_to_wdl() |> tune_draw_range()
pois_skel_wdl_outer  = pois_skel_preds_outer  |> convert_preds_to_wdl() |> tune_draw_range()
pois_skel_wdl_tuning = pois_skel_preds_tuning |> convert_preds_to_wdl() |> tune_draw_range()

# thresholds chosen per held-out year
pois_mse_wdl_outer |> distinct(Year, draw_t)
pois_rps_wdl_outer |> distinct(Year, draw_t)
pois_skel_wdl_outer |> distinct(Year, draw_t)

# Section 12.3: Final draw threshold for the 2028 rule----
# best t on all pooled outer matches. This is a decision rule, not a performance estimate
pois_mse_t_final = best_draw_t(pois_mse_wdl_outer)
pois_rps_t_final = best_draw_t(pois_rps_wdl_outer)
pois_skel_t_final = best_draw_t(pois_skel_wdl_outer)

# Section 12.4: Baseline (no model)----
baseline_preds_outer = baseline_preds()
baseline_wdl_outer   = baseline_wdl() |> tune_draw_range()
baseline_t_final     = best_draw_t(baseline_wdl_outer)
baseline_wdl_insample = baseline_wdl(leave_out = FALSE) |> tune_draw_range()

baseline_wdl_outer |> distinct(Year, p_win, p_draw, p_loss, draw_t)

# Section 13: pois results----

# Section 13.1: Score table----
# preds = nested_outer is the honest estimate, tuning_pass is optimistic (comparable to teammates)
pois_results = bind_rows(
  bind_rows(goal_scores(pois_mse_preds_outer),  wdl_scores(pois_mse_wdl_outer))  |> mutate(select = "mse", preds = "nested_outer"),
  bind_rows(goal_scores(pois_mse_preds_tuning), wdl_scores(pois_mse_wdl_tuning)) |> mutate(select = "mse", preds = "tuning_pass"),
  bind_rows(goal_scores(pois_rps_preds_outer),  wdl_scores(pois_rps_wdl_outer))  |> mutate(select = "rps", preds = "nested_outer"),
  bind_rows(goal_scores(pois_rps_preds_tuning), wdl_scores(pois_rps_wdl_tuning)) |> mutate(select = "rps", preds = "tuning_pass"),
  bind_rows(goal_scores(pois_skel_preds_outer),  wdl_scores(pois_skel_wdl_outer))  |> mutate(select = "skel", preds = "nested_outer"),
  bind_rows(goal_scores(pois_skel_preds_tuning), wdl_scores(pois_skel_wdl_tuning)) |> mutate(select = "skel", preds = "tuning_pass")
) |>
  mutate(model_class = "pois")

# nothing is tuned in the baseline, so its leave-one-year-out predictions are honest like nested_outer
baseline_results = bind_rows(goal_scores(baseline_preds_outer), wdl_scores(baseline_wdl_outer)) |>
  mutate(select = "none", preds = "nested_outer", model_class = "baseline")

bind_rows(baseline_results, pois_results) |>
  pivot_wider(names_from = c(model_class, select, preds), values_from = value)

# Section 13.2: Confusion matrices----
# tuning pass = the final tuning run (optimistic, comparable to the teammates' matrices)
# nested_outer beside it is the honest version; the gap between them is the selection optimism
# _raw = plain argmax, _t = range-adjusted (leave-one-tournament-out draw threshold)
# conf_mat layout: rows = Prediction, columns = Truth
wdl_conf_mat = function(wdl, pred_col) {
  yardstick::conf_mat(wdl, truth = actual_wdl, estimate = {{ pred_col }})
}

pois_mse_cm_tuning_raw = wdl_conf_mat(pois_mse_wdl_tuning, pred_wdl_raw)
pois_mse_cm_tuning_t   = wdl_conf_mat(pois_mse_wdl_tuning, pred_wdl_t)
pois_rps_cm_tuning_raw = wdl_conf_mat(pois_rps_wdl_tuning, pred_wdl_raw)
pois_rps_cm_tuning_t   = wdl_conf_mat(pois_rps_wdl_tuning, pred_wdl_t)
pois_skel_cm_tuning_raw = wdl_conf_mat(pois_skel_wdl_tuning, pred_wdl_raw)
pois_skel_cm_tuning_t   = wdl_conf_mat(pois_skel_wdl_tuning, pred_wdl_t)

pois_mse_cm_outer_raw  = wdl_conf_mat(pois_mse_wdl_outer, pred_wdl_raw)
pois_mse_cm_outer_t    = wdl_conf_mat(pois_mse_wdl_outer, pred_wdl_t)
pois_rps_cm_outer_raw  = wdl_conf_mat(pois_rps_wdl_outer, pred_wdl_raw)
pois_rps_cm_outer_t    = wdl_conf_mat(pois_rps_wdl_outer, pred_wdl_t)
pois_skel_cm_outer_raw  = wdl_conf_mat(pois_skel_wdl_outer, pred_wdl_raw)
pois_skel_cm_outer_t    = wdl_conf_mat(pois_skel_wdl_outer, pred_wdl_t)

baseline_cm_outer_raw  = wdl_conf_mat(baseline_wdl_outer, pred_wdl_raw)

pois_mse_cm_tuning_raw
pois_mse_cm_tuning_t
pois_rps_cm_tuning_raw
pois_rps_cm_tuning_t
pois_skel_cm_tuning_raw
pois_skel_cm_tuning_t

pois_mse_cm_outer_t
pois_rps_cm_outer_t
pois_skel_cm_outer_t
baseline_cm_outer_raw

save_plot(autoplot(pois_mse_cm_tuning_raw, type = "heatmap") + labs(title = "pois, selected on mse: tuning pass, argmax"),
          "cm_pois_mse_tuning_raw", width = 5, height = 4)
save_plot(autoplot(pois_mse_cm_tuning_t,   type = "heatmap") + labs(title = "pois, selected on mse: tuning pass, range-adjusted"),
          "cm_pois_mse_tuning_t", width = 5, height = 4)
save_plot(autoplot(pois_rps_cm_tuning_raw, type = "heatmap") + labs(title = "pois, selected on rps: tuning pass, argmax"),
          "cm_pois_rps_tuning_raw", width = 5, height = 4)
save_plot(autoplot(pois_rps_cm_tuning_t,   type = "heatmap") + labs(title = "pois, selected on rps: tuning pass, range-adjusted"),
          "cm_pois_rps_tuning_t", width = 5, height = 4)
save_plot(autoplot(pois_skel_cm_tuning_raw, type = "heatmap") + labs(title = "pois, selected on skellam nll: tuning pass, argmax"),
          "cm_pois_skel_tuning_raw", width = 5, height = 4)
save_plot(autoplot(pois_skel_cm_tuning_t,   type = "heatmap") + labs(title = "pois, selected on skellam nll: tuning pass, range-adjusted"),
          "cm_pois_skel_tuning_t", width = 5, height = 4)

# Section 13.3: Nested evaluation stats for model comparison----
# Use these (nested_outer) to compare models / selection rules: every year was predicted by a
# procedure that never saw it. Tuning-pass numbers are NOT for comparison (optimistic)
#   pooled   = metric computed once over all held-out matches (same as pois_results)
#   se       = sd of the 6 per-year values / sqrt(6), how much the estimate moves between tournaments
#   min/max  = worst and best single tournament
# Accuracy-type and F1 metrics aren't averages over matches, so their per-year values can sit
# outside the pooled value (e.g. the baseline predicts one class per year but both classes pooled);
# read their se loosely

# Scores for each held-out year separately
per_year_scores = function(preds, wdl) {
  map(sort(unique(wdl$Year)), function(y) {
    bind_rows(
      goal_scores(preds[euro$Year[preds$.row] == y, ], check_rows = FALSE),
      wdl_scores(filter(wdl, Year == y))
    ) |>
      mutate(Year = y)
  }) |>
    bind_rows()
}

nested_per_year = bind_rows(
  per_year_scores(baseline_preds_outer, baseline_wdl_outer) |> mutate(experiment = "baseline"),
  per_year_scores(pois_mse_preds_outer, pois_mse_wdl_outer) |> mutate(experiment = "pois_mse"),
  per_year_scores(pois_rps_preds_outer, pois_rps_wdl_outer) |> mutate(experiment = "pois_rps"),
  per_year_scores(pois_skel_preds_outer, pois_skel_wdl_outer) |> mutate(experiment = "pois_skel")
)

nested_pooled = bind_rows(baseline_results, pois_results) |>
  filter(preds == "nested_outer") |>
  mutate(experiment = if_else(model_class == "baseline", "baseline", paste0(model_class, "_", select))) |>
  select(experiment, metric, pooled = value)

nested_eval = nested_per_year |>
  group_by(experiment, metric) |>
  summarise(se       = sd(value) / sqrt(n()),
            min_year = min(value),
            max_year = max(value),
            .groups  = "drop") |>
  left_join(nested_pooled, by = c("experiment", "metric")) |>
  select(experiment, metric, pooled, se, min_year, max_year)

nested_eval

# compact version for the report: "pooled (se)", one column per experiment
nested_eval |>
  mutate(value = sprintf("%.3f (%.3f)", pooled, se)) |>
  select(metric, experiment, value) |>
  pivot_wider(names_from = experiment, values_from = value)

# Section 13.4: Baseline vs argmax Skellam vs range-adjusted Skellam----
# Same layout as Random_Forest_LOTOCV.Rmd section 16: one row per model, metrics as columns
# Log_Loss, Brier_Score and RPS only use the probabilities, so the argmax and range-adjusted rows
# of the same model share them; only the class-based columns differ

# Class-based scores, with the draw class broken out
#   Draw_Precision: of the matches called a draw, how many were draws (NA if none were called)
#   Draw_Recall:    of the actual draws, how many were called a draw
draw_scores = function(truth, pred) {
  tp     = sum(pred == "draw" & truth == "draw")
  n_pred = sum(pred == "draw")
  n_true = sum(truth == "draw")
  tibble(
    Accuracy        = mean(pred == truth),
    Macro_F1        = macro_f1(truth, pred),
    Draws_Predicted = n_pred,
    Draw_Precision  = if (n_pred == 0) NA_real_ else tp / n_pred,
    Draw_Recall     = tp / n_true,
    Draw_F1         = if (n_pred + n_true == 0) 0 else 2 * tp / (n_pred + n_true)
  )
}

comparison_row = function(wdl, pred_col, model) {
  prob = wdl_scores(wdl)
  tibble(
    Model       = model,
    Log_Loss    = prob$value[prob$metric == "log_loss"],
    Brier_Score = prob$value[prob$metric == "brier"],
    RPS         = prob$value[prob$metric == "rps"]
  ) |>
    bind_cols(draw_scores(wdl$actual_wdl, wdl[[pred_col]]))
}

# Rmd 17.3-style dodged bars of the class-based scores
plot_model_comparison = function(comparison, title) {
  comparison |>
    select(Model, Accuracy, Macro_F1, Draw_Recall, Draw_F1) |>
    pivot_longer(-Model, names_to = "Metric", values_to = "Value") |>
    mutate(Metric = factor(Metric, levels = c("Accuracy", "Macro_F1", "Draw_Recall", "Draw_F1")),
           Model  = factor(Model, levels = comparison$Model)) |>
    ggplot(aes(x = Metric, y = Value, fill = Model)) +
    geom_col(position = "dodge", width = 0.7) +
    labs(title = title, x = NULL, y = NULL, fill = "Model") +
    theme_minimal()
}

# honest: nested outer predictions, leave-one-year-out baseline
model_comparison = bind_rows(
  comparison_row(baseline_wdl_outer, "pred_wdl_raw", "Historical Frequency Baseline"),
  comparison_row(pois_mse_wdl_outer, "pred_wdl_raw", "Pois + Skellam (argmax), mse"),
  comparison_row(pois_mse_wdl_outer, "pred_wdl_t",   "Pois + Skellam (range-adjusted), mse"),
  comparison_row(pois_rps_wdl_outer, "pred_wdl_raw", "Pois + Skellam (argmax), rps"),
  comparison_row(pois_rps_wdl_outer, "pred_wdl_t",   "Pois + Skellam (range-adjusted), rps"),
  comparison_row(pois_skel_wdl_outer, "pred_wdl_raw", "Pois + Skellam (argmax), skel"),
  comparison_row(pois_skel_wdl_outer, "pred_wdl_t",   "Pois + Skellam (range-adjusted), skel")
)
print(model_comparison, width = Inf)

# optimistic: tuning pass and in-sample baseline, to line up with the teammates' numbers
model_comparison_tuning = bind_rows(
  comparison_row(baseline_wdl_insample, "pred_wdl_raw", "Historical Frequency Baseline"),
  comparison_row(pois_mse_wdl_tuning,   "pred_wdl_raw", "Pois + Skellam (argmax), mse"),
  comparison_row(pois_mse_wdl_tuning,   "pred_wdl_t",   "Pois + Skellam (range-adjusted), mse"),
  comparison_row(pois_rps_wdl_tuning,   "pred_wdl_raw", "Pois + Skellam (argmax), rps"),
  comparison_row(pois_rps_wdl_tuning,   "pred_wdl_t",   "Pois + Skellam (range-adjusted), rps"),
  comparison_row(pois_skel_wdl_tuning,  "pred_wdl_raw", "Pois + Skellam (argmax), skel"),
  comparison_row(pois_skel_wdl_tuning,  "pred_wdl_t",   "Pois + Skellam (range-adjusted), skel")
)
print(model_comparison_tuning, width = Inf)

save_plot(plot_model_comparison(model_comparison, "Baseline vs Skellam (nested outer)"),
          "model_comparison_outer")
save_plot(plot_model_comparison(model_comparison_tuning, "Baseline vs Skellam (tuning pass)"),
          "model_comparison_tuning")

# Section 14: Final model coefficients----
# glmnet coefficients of the final (2028) model, on the original scale of each predictor
# Predictors are team-minus-opponent differences (except GroupStage), so:
#   estimate       = change in log expected goals per 1-unit difference
#   rate_ratio     = exp(estimate): expected goals are multiplied by this per 1-unit difference
#   std_estimate   = estimate * sd(x): effect of a 1-SD difference, comparable across predictors
#                    (UefaPoints is in the 10,000s and GDP around 1, so raw estimates aren't comparable)
#   std_rate_ratio = exp(std_estimate)
# glmnet shrinks coefficients toward 0 (attenuated), and an exact 0 means the lasso dropped the variable
coef_table = function(final) {
  predictor_sd = euro |>
    summarise(across(all_of(predictors), sd)) |>
    pivot_longer(everything(), names_to = "term", values_to = "sd_x")

  broom::tidy(extract_fit_parsnip(final)) |>
    select(term, estimate) |>
    left_join(predictor_sd, by = "term") |>
    mutate(rate_ratio     = exp(estimate),
           std_estimate   = estimate * sd_x,
           std_rate_ratio = exp(std_estimate))
}

pois_mse_final$selected
pois_mse_coefs = coef_table(pois_mse_final)
pois_mse_coefs

pois_rps_final$selected
pois_rps_coefs = coef_table(pois_rps_final)
pois_rps_coefs

pois_skel_final$selected
pois_skel_coefs = coef_table(pois_skel_final)
pois_skel_coefs

# sanity check: the coefficients reproduce the final model's prediction for the first row
exp(sum(pois_rps_coefs$estimate * c(1, unlist(euro[1, predictors]))))
predict(extract_workflow(pois_rps_final), new_data = euro[1, ])

# Section 15: Hyperparameter selection variability----
# Each outer fold runs its own inner tuning. These heatmaps show every candidate's inner CV score
# per outer fold; the black outline is the candidate that fold selected. Lower is better for mse
# and wdl_rps. fill = "std_err" shows how much that candidate's score varies across inner folds
plot_inner_heatmap = function(nested, metric_name, fill_stat = c("mean", "std_err"), title) {
  fill_stat = match.arg(fill_stat)
  inner  = collect_inner_metrics(nested) |> filter(.metric == metric_name)
  chosen = collect_selections(nested)

  ggplot(inner, aes(x = factor(signif(penalty, 2)), y = factor(mixture))) +
    geom_tile(aes(fill = .data[[fill_stat]])) +
    geom_tile(data = chosen, fill = NA, colour = "black", linewidth = 1) +
    facet_wrap(~ id) +
    scale_fill_viridis_c() +
    labs(title = title, x = "penalty", y = "mixture", fill = paste(metric_name, fill_stat)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Across outer folds: each candidate's mean inner score (stat = "mean") or how much it moves
# between outer folds (stat = "sd")
plot_inner_summary = function(nested, metric_name, stat = c("mean", "sd"), title) {
  stat = match.arg(stat)
  collect_inner_metrics(nested) |>
    filter(.metric == metric_name) |>
    group_by(penalty, mixture) |>
    summarise(value = if (stat == "mean") mean(mean) else sd(mean), .groups = "drop") |>
    ggplot(aes(x = factor(signif(penalty, 2)), y = factor(mixture), fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(title = title, x = "penalty", y = "mixture",
         fill = paste(stat, "of", metric_name, "\nacross outer folds")) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

save_plot(plot_inner_heatmap(pois_mse_nested, "mse", "mean",    "pois, selected on mse: inner mean mse by outer fold"),
          "heatmap_pois_mse_mean", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_mse_nested, "mse", "std_err", "pois, selected on mse: inner std_err of mse by outer fold"),
          "heatmap_pois_mse_std_err", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_rps_nested, "wdl_rps", "mean",    "pois, selected on rps: inner mean wdl_rps by outer fold"),
          "heatmap_pois_rps_mean", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_rps_nested, "wdl_rps", "std_err", "pois, selected on rps: inner std_err of wdl_rps by outer fold"),
          "heatmap_pois_rps_std_err", width = 10, height = 6)

save_plot(plot_inner_heatmap(pois_skel_nested, "skellam_nll_by_match", "mean",    "pois, selected on skellam nll: inner mean by outer fold"),
          "heatmap_pois_skel_mean", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_skel_nested, "skellam_nll_by_match", "std_err", "pois, selected on skellam nll: inner std_err by outer fold"),
          "heatmap_pois_skel_std_err", width = 10, height = 6)

save_plot(plot_inner_summary(pois_mse_nested, "mse", "mean", "pois, mse: mean across outer folds"),     "summary_pois_mse_mean")
save_plot(plot_inner_summary(pois_mse_nested, "mse", "sd",   "pois, mse: sd across outer folds"),       "summary_pois_mse_sd")
save_plot(plot_inner_summary(pois_rps_nested, "wdl_rps", "mean", "pois, rps: mean across outer folds"), "summary_pois_rps_mean")
save_plot(plot_inner_summary(pois_rps_nested, "wdl_rps", "sd",   "pois, rps: sd across outer folds"),   "summary_pois_rps_sd")
save_plot(plot_inner_summary(pois_skel_nested, "skellam_nll_by_match", "mean", "pois, skellam nll: mean across outer folds"), "summary_pois_skel_mean")
save_plot(plot_inner_summary(pois_skel_nested, "skellam_nll_by_match", "sd",   "pois, skellam nll: sd across outer folds"),   "summary_pois_skel_sd")

# Section 15.1: Did the selection metric (mse / rps / skellam nll) change the hyperparameters?----
# per outer fold, side by side
selection_comparison = full_join(
  collect_selections(pois_mse_nested) |> select(id, mse_penalty = penalty, mse_mixture = mixture),
  collect_selections(pois_rps_nested) |> select(id, rps_penalty = penalty, rps_mixture = mixture),
  by = "id"
) |>
  full_join(collect_selections(pois_skel_nested) |> select(id, skel_penalty = penalty, skel_mixture = mixture),
            by = "id") |>
  mutate(mse_vs_rps  = mse_penalty == rps_penalty  & mse_mixture == rps_mixture,
         skel_vs_mse = skel_penalty == mse_penalty & skel_mixture == mse_mixture,
         skel_vs_rps = skel_penalty == rps_penalty & skel_mixture == rps_mixture)
selection_comparison

# how often each config was picked across the 6 outer folds
agreement(pois_mse_nested)
agreement(pois_rps_nested)
agreement(pois_skel_nested)

# final (2028) models
bind_rows(pois_mse_final$selected |> mutate(select = "mse"),
          pois_rps_final$selected |> mutate(select = "rps"),
          pois_skel_final$selected |> mutate(select = "skel"))

# Same check within ONE run: the rps run also computed mse on the inner folds, so we can read off
# what mse would have picked from exactly the same fits (no run-to-run randomness)
best_inner = function(nested, metric_name) {
  collect_inner_metrics(nested) |>
    filter(.metric == metric_name) |>
    group_by(id) |>
    slice_min(mean, n = 1, with_ties = FALSE) |>
    ungroup() |>
    select(id, penalty, mixture)
}

within_rps_run = full_join(
  best_inner(pois_rps_nested, "wdl_rps") |> rename(rps_penalty = penalty, rps_mixture = mixture),
  best_inner(pois_rps_nested, "mse")     |> rename(mse_penalty = penalty, mse_mixture = mixture),
  by = "id"
) |>
  mutate(same = mse_penalty == rps_penalty & mse_mixture == rps_mixture)
within_rps_run

# The skellam run computed all three criteria on the same inner fits, so this is the cleanest
# three-way comparison of what each metric would pick
within_skel_run = best_inner(pois_skel_nested, "skellam_nll_by_match") |>
  rename(skel_penalty = penalty, skel_mixture = mixture) |>
  full_join(best_inner(pois_skel_nested, "wdl_rps") |> rename(rps_penalty = penalty, rps_mixture = mixture), by = "id") |>
  full_join(best_inner(pois_skel_nested, "mse")     |> rename(mse_penalty = penalty, mse_mixture = mixture), by = "id") |>
  mutate(skel_vs_mse = skel_penalty == mse_penalty & skel_mixture == mse_mixture,
         skel_vs_rps = skel_penalty == rps_penalty & skel_mixture == rps_mixture)
within_skel_run

# Section 16: Win/draw/loss probability plots (as in Random_Forest_LOTOCV.Rmd section 17)----
# Built from the nested outer predictions
plot_wdl_probs = function(wdl, label) {
  lvls = c("win", "draw", "loss")
  long = wdl |>
    select(actual_wdl, p_win, p_draw, p_loss) |>
    pivot_longer(c(p_win, p_draw, p_loss), names_to = "outcome", values_to = "probability") |>
    mutate(outcome = factor(recode(outcome, p_win = "win", p_draw = "draw", p_loss = "loss"), levels = lvls))

  # 17.1 distribution of predicted probabilities
  hist = ggplot(long, aes(x = probability, fill = outcome)) +
    geom_histogram(bins = 25, alpha = 0.75, colour = "white") +
    facet_wrap(~ outcome, ncol = 1) +
    labs(title = paste("Distribution of predicted outcome probabilities:", label),
         x = "Predicted probability", y = "Number of matches") +
    theme_minimal() +
    theme(legend.position = "none")

  # 17.2 predicted probabilities by actual outcome
  box = ggplot(long, aes(x = actual_wdl, y = probability, fill = actual_wdl)) +
    geom_boxplot(alpha = 0.75) +
    facet_wrap(~ outcome, ncol = 3, labeller = as_labeller(\(x) paste0("P(", x, ")"))) +
    labs(title = paste("Predicted probabilities by actual outcome:", label),
         x = "Actual outcome", y = "Predicted probability") +
    theme_minimal() +
    theme(legend.position = "none")

  # 17.3 mean predicted probability vs actual frequency
  calib = tibble(
    outcome = factor(lvls, levels = lvls),
    `Mean predicted probability` = c(mean(wdl$p_win), mean(wdl$p_draw), mean(wdl$p_loss)),
    `Actual frequency`           = c(mean(wdl$actual_wdl == "win"),
                                     mean(wdl$actual_wdl == "draw"),
                                     mean(wdl$actual_wdl == "loss"))
  ) |>
    pivot_longer(-outcome, names_to = "measure", values_to = "probability") |>
    ggplot(aes(x = outcome, y = probability, fill = measure)) +
    geom_col(position = "dodge", width = 0.7) +
    labs(title = paste("Mean predicted probabilities vs actual frequencies:", label),
         x = "Match outcome", y = "Probability", fill = NULL) +
    theme_minimal()

  list(hist = hist, box = box, calib = calib)
}

pois_mse_wdl_plots = plot_wdl_probs(pois_mse_wdl_outer, "pois, selected on mse")
pois_rps_wdl_plots = plot_wdl_probs(pois_rps_wdl_outer, "pois, selected on rps")
pois_skel_wdl_plots = plot_wdl_probs(pois_skel_wdl_outer, "pois, selected on skellam nll")

iwalk(pois_mse_wdl_plots, \(plot, name) save_plot(plot, paste0("wdl_pois_mse_", name)))
iwalk(pois_rps_wdl_plots, \(plot, name) save_plot(plot, paste0("wdl_pois_rps_", name)))
iwalk(pois_skel_wdl_plots, \(plot, name) save_plot(plot, paste0("wdl_pois_skel_", name)))
