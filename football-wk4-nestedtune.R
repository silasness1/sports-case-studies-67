# README: now that we've each tried to get some results about nnet, random forest, xgboost, and poisson 
# separately, this script tries to replicate it using a more streamlined/organized high-level R packages
# namely parsnip models, workflows, and nestedtune


# HIGHLEVEL 

# Important objects: 
#   - resample_struct
#   - metrics_sel_mae, metrics_sel_rps, metrics_sel_skel (Section 4.3)
#   - RUN_* switches and the naming scheme (Section 9)
#   - <model_class>_<select>_nested / _final (Section 10); pois_cov_* = pois + AgeDifference
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
library(corrplot)

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

euro_2004_cov = read_csv("raw-data/euro_2004_with_age.csv")
euro_2008_cov = read_csv("raw-data/euro_2008_with_age.csv")
euro_2012_cov = read_csv("raw-data/euro_2012_with_age.csv")
euro_2016_cov = read_csv("raw-data/euro_2016_with_age.csv")
euro_2020_cov = read_csv("raw-data/euro_2020_with_age.csv")
euro_2024_cov = read_csv("raw-data/euro_2024_with_age.csv")

euro_cov = rbind(euro_2004_cov,
             euro_2008_cov,
             euro_2012_cov,
             euro_2016_cov,
             euro_2020_cov,
             euro_2024_cov) 

# euro_cov = the same rows in the same order as euro, plus age covariates (TeamAge, OpponentAge,
# AgeDifference, RelativeAgeDifference), but without match_id. Check the shared columns really are
# identical, then copy match_id over
shared_cols = setdiff(names(euro), "match_id")
stopifnot(nrow(euro_cov) == nrow(euro),
          isTRUE(all.equal(as.data.frame(euro_cov[shared_cols]), as.data.frame(euro[shared_cols]),
                           check.attributes = FALSE)))
euro_cov$match_id = euro$match_id

# From here on `euro` carries the age covariates, so the base and +age models use exactly the same
# rows and splits. The base formulas don't mention the new columns, so they ignore them.
# Only AgeDifference (team age - opponent age) is added to the +age formulas: it's a difference like
# the other covariates, and RelativeAgeDifference is the same information (correlation 1.000)
euro = euro_cov


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

# Section 4.1.1: why we tune on yardstick::mae / mse instead of rmse-----
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
# mae so the selection matches the teammates' model class comparisons (caveat: mae is minimized by
# the median, mse by the mean, and Skellam needs each team's MEAN goals (lambda)). Both are means
# over rows, so fold-weighted averaging is exactly pooled (Section 4.1.1)
metrics_sel_mae = yardstick::metric_set(
  mae,
  mse,
  poisson_log_loss
)

metrics_sel_rps = yardstick::metric_set(
  wdl_rps,
  mae,
  mse,
  poisson_log_loss
)

# skellam_nll_by_match first (selection), plus wdl_rps and mae so this one run has all three
# selection criteria computed on exactly the same fits (Section 15.1)
metrics_sel_skel = yardstick::metric_set(
  skellam_nll_by_match,
  wdl_rps,
  mse,
  mae,
  poisson_log_loss
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

# + age: same model and grid, AgeDifference added to the formula
pois_cov_rec = recipe(
  data = euro,
  Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers + AgeDifference + match_id) |>
  update_role(match_id, new_role = "id")

pois_cov_wf = workflow() |>
  add_model(pois_tune) |>
  add_recipe(pois_cov_rec)

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

nnet_cov_wf = workflow() |>
  add_model(nnet_tune) |>
  add_formula(Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers + AgeDifference)

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

# + age
predictors_cov <- c(predictors, "AgeDifference")
rf_formula_cov <- as.formula(
  paste(target, "~", paste(predictors_cov, collapse = " + "))
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

# + age: mtry can now go up to 7 predictors; min_n / sample.fraction as in rf_grid (incl. SIMPLE)
rf_cov_grid <- expand.grid(
  mtry            = if (SIMPLE == T) 3:length(predictors_cov) else 1:length(predictors_cov),
  min_n           = unique(rf_grid$min_n),
  sample.fraction = unique(rf_grid$sample.fraction)
)

rf_cov_wf = workflow() |>
  add_model(rf_tune) |>
  add_formula(rf_formula_cov)

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

xg_cov_wf = workflow() |>
  add_model(xg_tune) |>
  add_formula(Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers + AgeDifference)

# Section 9: Experiment settings----

# Naming scheme: <model_class>_<selection metric>_<stage>
#   pois_mae_nested        nested_tune_grid result, hyperparameters selected on mae (old pois_tuned-style object)
#   pois_skel_nested       same, selected on the skellam negative log likelihood of each match
#   pois_cov_mae_nested    pois_mae_nested with AgeDifference added (pois_cov_wf); same for every stage
#   pois_rps_final         nested_final_fit: tuning pass on all 6 years + fit on all data (for 2028)
#   pois_mae_preds_outer   goal predictions from the nested outer folds (honest)
#   pois_mae_preds_tuning  goal predictions from the tuning pass at the chosen config (optimistic,
#                          comparable to Random_Forest_LOTOCV.Rmd and xgboost_with_skellam.R)
#   pois_mae_wdl_outer_raw match-level win/draw/loss built from preds_outer: probabilities + argmax, no threshold
#   pois_mae_wdl_outer     the same with the leave-one-tournament-out draw threshold added (draw_t, pred_wdl_t)
# To add rf / xg / nnet: copy a Section 10 block and rename pois -> rf etc.

# TRUE = refit and overwrite results/<name>.rds, FALSE = load the last saved fit (no refit)
RUN_POIS_MAE = FALSE
RUN_POIS_RPS = FALSE
RUN_POIS_SKEL = FALSE
RUN_POIS_COV_MAE  = FALSE
RUN_POIS_COV_RPS  = FALSE
RUN_POIS_COV_SKEL = FALSE

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
    # nested_tune_grid produced 6 model fits and produced the score (e.g. collect_metrics(pois_mae_nested))
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

# nested_final_fit names its tuning-pass resamples Resample1..6 in no particular year order
# (Resample1 is 2012). Rename them "test YYYY" like the nested folds, reusing Section 3.1's helpers.
# Applied after Section 10 so it covers both freshly fitted and readRDS-loaded fits
name_final_ids = function(final) {
  final$tuning = rename_rset_ids(final$tuning, get_resample_ids(final$tuning, all_years))
  final
}

# Section 10: Running the experiments----

# Section 10.1: pois, hyperparameters selected on mae----
if (RUN_POIS_MAE) {
  pois_mae_nested = nested_tune_grid_mirai(          # full run: ~121s
    object     = pois_wf,
    resamples  = resample_struct,
    grid       = pois_grid,
    metrics    = metrics_sel_mae,
    select     = selection_rule(rule = "best"),     # best on mae (first in metrics_sel_mae)
    control    = nested_control
  )
  pois_mae_final = nested_final_fit_mirai(object = pois_wf, results = pois_mae_nested)

  saveRDS(list(nested = pois_mae_nested, final = pois_mae_final),
          file.path(RESULTS_DIR, "pois_mae.rds"))
} else {
  pois_mae_saved  = readRDS(file.path(RESULTS_DIR, "pois_mae.rds"))
  pois_mae_nested = pois_mae_saved$nested
  pois_mae_final  = pois_mae_saved$final
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

# Section 10.4: pois + age, hyperparameters selected on mae----
if (RUN_POIS_COV_MAE) {
  pois_cov_mae_nested = nested_tune_grid_mirai(          # full run: ~?s
    object     = pois_cov_wf,
    resamples  = resample_struct,
    grid       = pois_grid,
    metrics    = metrics_sel_mae,
    select     = selection_rule(rule = "best"),     # best on mae (first in metrics_sel_mae)
    control    = nested_control
  )
  pois_cov_mae_final = nested_final_fit_mirai(object = pois_cov_wf, results = pois_cov_mae_nested)

  saveRDS(list(nested = pois_cov_mae_nested, final = pois_cov_mae_final),
          file.path(RESULTS_DIR, "pois_cov_mae.rds"))
} else {
  pois_cov_mae_saved  = readRDS(file.path(RESULTS_DIR, "pois_cov_mae.rds"))
  pois_cov_mae_nested = pois_cov_mae_saved$nested
  pois_cov_mae_final  = pois_cov_mae_saved$final
}

# Section 10.5: pois + age, hyperparameters selected on wdl_rps----
if (RUN_POIS_COV_RPS) {
  pois_cov_rps_nested = nested_tune_grid_mirai(          # full run: ~?s
    object     = pois_cov_wf,
    resamples  = resample_struct,
    grid       = pois_grid,
    metrics    = metrics_sel_rps,
    select     = selection_rule(rule = "best"),     # best on wdl_rps (first in metrics_sel_rps)
    control    = nested_control
  )
  pois_cov_rps_final = nested_final_fit_mirai(object = pois_cov_wf, results = pois_cov_rps_nested)

  saveRDS(list(nested = pois_cov_rps_nested, final = pois_cov_rps_final),
          file.path(RESULTS_DIR, "pois_cov_rps.rds"))
} else {
  pois_cov_rps_saved  = readRDS(file.path(RESULTS_DIR, "pois_cov_rps.rds"))
  pois_cov_rps_nested = pois_cov_rps_saved$nested
  pois_cov_rps_final  = pois_cov_rps_saved$final
}

# Section 10.6: pois + age, hyperparameters selected on skellam negative log likelihood----
if (RUN_POIS_COV_SKEL) {
  pois_cov_skel_nested = nested_tune_grid_mirai(         # full run: ~?s
    object     = pois_cov_wf,
    resamples  = resample_struct,
    grid       = pois_grid,
    metrics    = metrics_sel_skel,
    select     = selection_rule(rule = "best"),     # best on skellam_nll_by_match (first in metrics_sel_skel)
    control    = nested_control
  )
  pois_cov_skel_final = nested_final_fit_mirai(object = pois_cov_wf, results = pois_cov_skel_nested)

  saveRDS(list(nested = pois_cov_skel_nested, final = pois_cov_skel_final),
          file.path(RESULTS_DIR, "pois_cov_skel.rds"))
} else {
  pois_cov_skel_saved  = readRDS(file.path(RESULTS_DIR, "pois_cov_skel.rds"))
  pois_cov_skel_nested = pois_cov_skel_saved$nested
  pois_cov_skel_final  = pois_cov_skel_saved$final
}

# Section 10.7: Name the final fits' tuning-pass resamples "test YYYY"----
pois_mae_final  = name_final_ids(pois_mae_final)
pois_rps_final  = name_final_ids(pois_rps_final)
pois_skel_final = name_final_ids(pois_skel_final)
pois_cov_mae_final  = name_final_ids(pois_cov_mae_final)
pois_cov_rps_final  = name_final_ids(pois_cov_rps_final)
pois_cov_skel_final = name_final_ids(pois_cov_skel_final)

# Section 10.8: Run times----
# (selections, final hyperparameters and nested metrics are reported in Sections 13-15)
pois_timing = tibble(
  experiment  = c("pois_mae_nested", "pois_mae_final", "pois_rps_nested", "pois_rps_final",
                  "pois_skel_nested", "pois_skel_final",
                  "pois_cov_mae_nested", "pois_cov_mae_final", "pois_cov_rps_nested", "pois_cov_rps_final",
                  "pois_cov_skel_nested", "pois_cov_skel_final"),
  started_at  = do.call(c, map(list(pois_mae_nested, pois_mae_final, pois_rps_nested, pois_rps_final,
                                    pois_skel_nested, pois_skel_final,
                                    pois_cov_mae_nested, pois_cov_mae_final, pois_cov_rps_nested, pois_cov_rps_final,
                                    pois_cov_skel_nested, pois_cov_skel_final),
                               \(x) attr(x, "started_at"))),
  elapsed_sec = map_dbl(list(pois_mae_nested, pois_mae_final, pois_rps_nested, pois_rps_final,
                             pois_skel_nested, pois_skel_final,
                             pois_cov_mae_nested, pois_cov_mae_final, pois_cov_rps_nested, pois_cov_rps_final,
                             pois_cov_skel_nested, pois_cov_skel_final),
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
  # get the best draw range hyperparam for all but the hold-out year
  t_by_year = sapply(years, function(y) best_draw_t(filter(wdl, Year != y), grid))
  names(t_by_year) = years
  
  wdl |>
    mutate(
      draw_t     = unname(t_by_year[as.character(Year)]), # draw_t contains one per match
      pred_wdl_t = pred_at_threshold(wdl, draw_t) # use the winner
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
pois_mae_preds_outer  = collect_predictions(pois_mae_nested)
pois_mae_preds_tuning = get_tuning_preds(pois_mae_final)
pois_rps_preds_outer  = collect_predictions(pois_rps_nested)
pois_rps_preds_tuning = get_tuning_preds(pois_rps_final)
pois_skel_preds_outer  = collect_predictions(pois_skel_nested)
pois_skel_preds_tuning = get_tuning_preds(pois_skel_final)
pois_cov_mae_preds_outer  = collect_predictions(pois_cov_mae_nested)
pois_cov_mae_preds_tuning = get_tuning_preds(pois_cov_mae_final)
pois_cov_rps_preds_outer  = collect_predictions(pois_cov_rps_nested)
pois_cov_rps_preds_tuning = get_tuning_preds(pois_cov_rps_final)
pois_cov_skel_preds_outer  = collect_predictions(pois_cov_skel_nested)
pois_cov_skel_preds_tuning = get_tuning_preds(pois_cov_skel_final)

# Section 12.2: Win/draw/loss with leave-one-tournament-out draw threshold----
# _raw: one row per match with Skellam probabilities, p_range and the argmax prediction (no threshold)
pois_mae_wdl_outer_raw  = convert_preds_to_wdl(pois_mae_preds_outer)
pois_rps_wdl_outer_raw  = convert_preds_to_wdl(pois_rps_preds_outer)
pois_skel_wdl_outer_raw = convert_preds_to_wdl(pois_skel_preds_outer)
pois_cov_mae_wdl_outer_raw  = convert_preds_to_wdl(pois_cov_mae_preds_outer)
pois_cov_rps_wdl_outer_raw  = convert_preds_to_wdl(pois_cov_rps_preds_outer)
pois_cov_skel_wdl_outer_raw = convert_preds_to_wdl(pois_cov_skel_preds_outer)

# adds draw_t (each year's threshold, chosen on the other years) and pred_wdl_t
pois_mae_wdl_outer  = tune_draw_range(pois_mae_wdl_outer_raw)
pois_rps_wdl_outer  = tune_draw_range(pois_rps_wdl_outer_raw)
pois_skel_wdl_outer = tune_draw_range(pois_skel_wdl_outer_raw)
pois_cov_mae_wdl_outer  = tune_draw_range(pois_cov_mae_wdl_outer_raw)
pois_cov_rps_wdl_outer  = tune_draw_range(pois_cov_rps_wdl_outer_raw)
pois_cov_skel_wdl_outer = tune_draw_range(pois_cov_skel_wdl_outer_raw)

# tuning pass: no final threshold is taken from these, so convert and threshold in one go
pois_mae_wdl_tuning  = pois_mae_preds_tuning  |> convert_preds_to_wdl() |> tune_draw_range()
pois_rps_wdl_tuning  = pois_rps_preds_tuning  |> convert_preds_to_wdl() |> tune_draw_range()
pois_skel_wdl_tuning = pois_skel_preds_tuning |> convert_preds_to_wdl() |> tune_draw_range()
pois_cov_mae_wdl_tuning  = pois_cov_mae_preds_tuning  |> convert_preds_to_wdl() |> tune_draw_range()
pois_cov_rps_wdl_tuning  = pois_cov_rps_preds_tuning  |> convert_preds_to_wdl() |> tune_draw_range()
pois_cov_skel_wdl_tuning = pois_cov_skel_preds_tuning |> convert_preds_to_wdl() |> tune_draw_range()

# thresholds chosen per held-out year
pois_mae_wdl_outer |> distinct(Year, draw_t)
pois_rps_wdl_outer |> distinct(Year, draw_t)
pois_skel_wdl_outer |> distinct(Year, draw_t)
pois_cov_mae_wdl_outer |> distinct(Year, draw_t)
pois_cov_rps_wdl_outer |> distinct(Year, draw_t)
pois_cov_skel_wdl_outer |> distinct(Year, draw_t)

# Section 12.3: Final draw threshold for the 2028 rule----
# best t on all pooled outer matches (the _raw tables: best_draw_t doesn't use the per-year
# draw_t / pred_wdl_t columns). This is a decision rule, not a performance estimate
pois_mae_t_final  = best_draw_t(pois_mae_wdl_outer_raw)
pois_rps_t_final  = best_draw_t(pois_rps_wdl_outer_raw)
pois_skel_t_final = best_draw_t(pois_skel_wdl_outer_raw)
pois_cov_mae_t_final  = best_draw_t(pois_cov_mae_wdl_outer_raw)
pois_cov_rps_t_final  = best_draw_t(pois_cov_rps_wdl_outer_raw)
pois_cov_skel_t_final = best_draw_t(pois_cov_skel_wdl_outer_raw)

# Section 12.4: Baseline (no model)----
baseline_preds_outer   = baseline_preds()
baseline_wdl_outer_raw = baseline_wdl()
baseline_wdl_outer     = tune_draw_range(baseline_wdl_outer_raw)
baseline_t_final       = best_draw_t(baseline_wdl_outer_raw)
baseline_wdl_insample  = baseline_wdl(leave_out = FALSE) |> tune_draw_range()

baseline_wdl_outer |> distinct(Year, p_win, p_draw, p_loss, draw_t)

# Section 13: pois results----

# Section 13.1: Score table----
# preds = nested_outer is the honest estimate, tuning_pass is optimistic (comparable to teammates)
pois_results = bind_rows(
  bind_rows(goal_scores(pois_mae_preds_outer),  wdl_scores(pois_mae_wdl_outer))  |> mutate(select = "mae", preds = "nested_outer"),
  bind_rows(goal_scores(pois_mae_preds_tuning), wdl_scores(pois_mae_wdl_tuning)) |> mutate(select = "mae", preds = "tuning_pass"),
  bind_rows(goal_scores(pois_rps_preds_outer),  wdl_scores(pois_rps_wdl_outer))  |> mutate(select = "rps", preds = "nested_outer"),
  bind_rows(goal_scores(pois_rps_preds_tuning), wdl_scores(pois_rps_wdl_tuning)) |> mutate(select = "rps", preds = "tuning_pass"),
  bind_rows(goal_scores(pois_skel_preds_outer),  wdl_scores(pois_skel_wdl_outer))  |> mutate(select = "skel", preds = "nested_outer"),
  bind_rows(goal_scores(pois_skel_preds_tuning), wdl_scores(pois_skel_wdl_tuning)) |> mutate(select = "skel", preds = "tuning_pass")
) |>
  mutate(model_class = "pois")

pois_cov_results = bind_rows(
  bind_rows(goal_scores(pois_cov_mae_preds_outer),  wdl_scores(pois_cov_mae_wdl_outer))  |> mutate(select = "mae", preds = "nested_outer"),
  bind_rows(goal_scores(pois_cov_mae_preds_tuning), wdl_scores(pois_cov_mae_wdl_tuning)) |> mutate(select = "mae", preds = "tuning_pass"),
  bind_rows(goal_scores(pois_cov_rps_preds_outer),  wdl_scores(pois_cov_rps_wdl_outer))  |> mutate(select = "rps", preds = "nested_outer"),
  bind_rows(goal_scores(pois_cov_rps_preds_tuning), wdl_scores(pois_cov_rps_wdl_tuning)) |> mutate(select = "rps", preds = "tuning_pass"),
  bind_rows(goal_scores(pois_cov_skel_preds_outer),  wdl_scores(pois_cov_skel_wdl_outer))  |> mutate(select = "skel", preds = "nested_outer"),
  bind_rows(goal_scores(pois_cov_skel_preds_tuning), wdl_scores(pois_cov_skel_wdl_tuning)) |> mutate(select = "skel", preds = "tuning_pass")
) |>
  mutate(model_class = "pois_cov")

# nothing is tuned in the baseline, so its leave-one-year-out predictions are honest like nested_outer
baseline_results = bind_rows(goal_scores(baseline_preds_outer), wdl_scores(baseline_wdl_outer)) |>
  mutate(select = "none", preds = "nested_outer", model_class = "baseline")

pois_results_baseline_wide = bind_rows(baseline_results, pois_results, pois_cov_results) |>
  pivot_wider(names_from = c(model_class, select, preds), values_from = value)

# Section 13.2: Confusion matrices----
# tuning pass = the final tuning run (optimistic, comparable to the teammates' matrices)
# nested_outer beside it is the honest version; the gap between them is the selection optimism
# _raw = plain argmax, _t = range-adjusted (leave-one-tournament-out draw threshold)
# conf_mat layout: rows = Prediction, columns = Truth
wdl_conf_mat = function(wdl, pred_col) {
  yardstick::conf_mat(wdl, truth = actual_wdl, estimate = {{ pred_col }})
}

pois_mae_cm_tuning_raw = wdl_conf_mat(pois_mae_wdl_tuning, pred_wdl_raw)
pois_mae_cm_tuning_t   = wdl_conf_mat(pois_mae_wdl_tuning, pred_wdl_t)
pois_rps_cm_tuning_raw = wdl_conf_mat(pois_rps_wdl_tuning, pred_wdl_raw)
pois_rps_cm_tuning_t   = wdl_conf_mat(pois_rps_wdl_tuning, pred_wdl_t)
pois_skel_cm_tuning_raw = wdl_conf_mat(pois_skel_wdl_tuning, pred_wdl_raw)
pois_skel_cm_tuning_t   = wdl_conf_mat(pois_skel_wdl_tuning, pred_wdl_t)
pois_cov_mae_cm_tuning_raw = wdl_conf_mat(pois_cov_mae_wdl_tuning, pred_wdl_raw)
pois_cov_mae_cm_tuning_t   = wdl_conf_mat(pois_cov_mae_wdl_tuning, pred_wdl_t)
pois_cov_rps_cm_tuning_raw = wdl_conf_mat(pois_cov_rps_wdl_tuning, pred_wdl_raw)
pois_cov_rps_cm_tuning_t   = wdl_conf_mat(pois_cov_rps_wdl_tuning, pred_wdl_t)
pois_cov_skel_cm_tuning_raw = wdl_conf_mat(pois_cov_skel_wdl_tuning, pred_wdl_raw)
pois_cov_skel_cm_tuning_t   = wdl_conf_mat(pois_cov_skel_wdl_tuning, pred_wdl_t)

pois_mae_cm_outer_raw  = wdl_conf_mat(pois_mae_wdl_outer, pred_wdl_raw)
pois_mae_cm_outer_t    = wdl_conf_mat(pois_mae_wdl_outer, pred_wdl_t)
pois_rps_cm_outer_raw  = wdl_conf_mat(pois_rps_wdl_outer, pred_wdl_raw)
pois_rps_cm_outer_t    = wdl_conf_mat(pois_rps_wdl_outer, pred_wdl_t)
pois_skel_cm_outer_raw  = wdl_conf_mat(pois_skel_wdl_outer, pred_wdl_raw)
pois_skel_cm_outer_t    = wdl_conf_mat(pois_skel_wdl_outer, pred_wdl_t)
pois_cov_mae_cm_outer_raw  = wdl_conf_mat(pois_cov_mae_wdl_outer, pred_wdl_raw)
pois_cov_mae_cm_outer_t    = wdl_conf_mat(pois_cov_mae_wdl_outer, pred_wdl_t)
pois_cov_rps_cm_outer_raw  = wdl_conf_mat(pois_cov_rps_wdl_outer, pred_wdl_raw)
pois_cov_rps_cm_outer_t    = wdl_conf_mat(pois_cov_rps_wdl_outer, pred_wdl_t)
pois_cov_skel_cm_outer_raw  = wdl_conf_mat(pois_cov_skel_wdl_outer, pred_wdl_raw)
pois_cov_skel_cm_outer_t    = wdl_conf_mat(pois_cov_skel_wdl_outer, pred_wdl_t)

baseline_cm_outer_raw  = wdl_conf_mat(baseline_wdl_outer, pred_wdl_raw)

pois_mae_cm_tuning_raw
pois_mae_cm_tuning_t
pois_rps_cm_tuning_raw
pois_rps_cm_tuning_t
pois_skel_cm_tuning_raw
pois_skel_cm_tuning_t
pois_cov_mae_cm_tuning_raw
pois_cov_mae_cm_tuning_t
pois_cov_rps_cm_tuning_raw
pois_cov_rps_cm_tuning_t
pois_cov_skel_cm_tuning_raw
pois_cov_skel_cm_tuning_t

pois_mae_cm_outer_t
pois_rps_cm_outer_t
pois_skel_cm_outer_t
pois_cov_mae_cm_outer_t
pois_cov_rps_cm_outer_t
pois_cov_skel_cm_outer_t
baseline_cm_outer_raw

save_plot(autoplot(pois_mae_cm_tuning_raw, type = "heatmap") + labs(title = "pois, selected on mae: tuning pass, argmax"),
          "cm_pois_mae_tuning_raw", width = 5, height = 4)
save_plot(autoplot(pois_mae_cm_tuning_t,   type = "heatmap") + labs(title = "pois, selected on mae: tuning pass, range-adjusted"),
          "cm_pois_mae_tuning_t", width = 5, height = 4)
save_plot(autoplot(pois_rps_cm_tuning_raw, type = "heatmap") + labs(title = "pois, selected on rps: tuning pass, argmax"),
          "cm_pois_rps_tuning_raw", width = 5, height = 4)
save_plot(autoplot(pois_rps_cm_tuning_t,   type = "heatmap") + labs(title = "pois, selected on rps: tuning pass, range-adjusted"),
          "cm_pois_rps_tuning_t", width = 5, height = 4)
save_plot(autoplot(pois_skel_cm_tuning_raw, type = "heatmap") + labs(title = "pois, selected on skellam nll: tuning pass, argmax"),
          "cm_pois_skel_tuning_raw", width = 5, height = 4)
save_plot(autoplot(pois_skel_cm_tuning_t,   type = "heatmap") + labs(title = "pois, selected on skellam nll: tuning pass, range-adjusted"),
          "cm_pois_skel_tuning_t", width = 5, height = 4)
save_plot(autoplot(pois_cov_mae_cm_tuning_raw, type = "heatmap") + labs(title = "pois + age, selected on mae: tuning pass, argmax"),
          "cm_pois_cov_mae_tuning_raw", width = 5, height = 4)
save_plot(autoplot(pois_cov_mae_cm_tuning_t,   type = "heatmap") + labs(title = "pois + age, selected on mae: tuning pass, range-adjusted"),
          "cm_pois_cov_mae_tuning_t", width = 5, height = 4)
save_plot(autoplot(pois_cov_rps_cm_tuning_raw, type = "heatmap") + labs(title = "pois + age, selected on rps: tuning pass, argmax"),
          "cm_pois_cov_rps_tuning_raw", width = 5, height = 4)
save_plot(autoplot(pois_cov_rps_cm_tuning_t,   type = "heatmap") + labs(title = "pois + age, selected on rps: tuning pass, range-adjusted"),
          "cm_pois_cov_rps_tuning_t", width = 5, height = 4)
save_plot(autoplot(pois_cov_skel_cm_tuning_raw, type = "heatmap") + labs(title = "pois + age, selected on skellam nll: tuning pass, argmax"),
          "cm_pois_cov_skel_tuning_raw", width = 5, height = 4)
save_plot(autoplot(pois_cov_skel_cm_tuning_t,   type = "heatmap") + labs(title = "pois + age, selected on skellam nll: tuning pass, range-adjusted"),
          "cm_pois_cov_skel_tuning_t", width = 5, height = 4)

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
  per_year_scores(pois_mae_preds_outer, pois_mae_wdl_outer) |> mutate(experiment = "pois_mae"),
  per_year_scores(pois_rps_preds_outer, pois_rps_wdl_outer) |> mutate(experiment = "pois_rps"),
  per_year_scores(pois_skel_preds_outer, pois_skel_wdl_outer) |> mutate(experiment = "pois_skel"),
  per_year_scores(pois_cov_mae_preds_outer,  pois_cov_mae_wdl_outer)  |> mutate(experiment = "pois_cov_mae"),
  per_year_scores(pois_cov_rps_preds_outer,  pois_cov_rps_wdl_outer)  |> mutate(experiment = "pois_cov_rps"),
  per_year_scores(pois_cov_skel_preds_outer, pois_cov_skel_wdl_outer) |> mutate(experiment = "pois_cov_skel")
)

nested_pooled = bind_rows(baseline_results, pois_results, pois_cov_results) |>
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
  pivot_wider(names_from = experiment, values_from = value) -> nested_eval_report

nested_eval_report

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
  comparison_row(pois_mae_wdl_outer, "pred_wdl_raw", "Pois + Skellam (argmax), mae"),
  comparison_row(pois_mae_wdl_outer, "pred_wdl_t",   "Pois + Skellam (range-adjusted), mae"),
  comparison_row(pois_rps_wdl_outer, "pred_wdl_raw", "Pois + Skellam (argmax), rps"),
  comparison_row(pois_rps_wdl_outer, "pred_wdl_t",   "Pois + Skellam (range-adjusted), rps"),
  comparison_row(pois_skel_wdl_outer, "pred_wdl_raw", "Pois + Skellam (argmax), skel"),
  comparison_row(pois_skel_wdl_outer, "pred_wdl_t",   "Pois + Skellam (range-adjusted), skel"),
  comparison_row(pois_cov_mae_wdl_outer,  "pred_wdl_raw", "Pois + Skellam (argmax), mae + age"),
  comparison_row(pois_cov_mae_wdl_outer,  "pred_wdl_t",   "Pois + Skellam (range-adjusted), mae + age"),
  comparison_row(pois_cov_rps_wdl_outer,  "pred_wdl_raw", "Pois + Skellam (argmax), rps + age"),
  comparison_row(pois_cov_rps_wdl_outer,  "pred_wdl_t",   "Pois + Skellam (range-adjusted), rps + age"),
  comparison_row(pois_cov_skel_wdl_outer, "pred_wdl_raw", "Pois + Skellam (argmax), skel + age"),
  comparison_row(pois_cov_skel_wdl_outer, "pred_wdl_t",   "Pois + Skellam (range-adjusted), skel + age")
)
print(model_comparison, width = Inf)

# optimistic: tuning pass and in-sample baseline, to line up with the teammates' numbers
model_comparison_tuning = bind_rows(
  comparison_row(baseline_wdl_insample, "pred_wdl_raw", "Historical Frequency Baseline"),
  comparison_row(pois_mae_wdl_tuning,   "pred_wdl_raw", "Pois + Skellam (argmax), mae"),
  comparison_row(pois_mae_wdl_tuning,   "pred_wdl_t",   "Pois + Skellam (range-adjusted), mae"),
  comparison_row(pois_rps_wdl_tuning,   "pred_wdl_raw", "Pois + Skellam (argmax), rps"),
  comparison_row(pois_rps_wdl_tuning,   "pred_wdl_t",   "Pois + Skellam (range-adjusted), rps"),
  comparison_row(pois_skel_wdl_tuning,  "pred_wdl_raw", "Pois + Skellam (argmax), skel"),
  comparison_row(pois_skel_wdl_tuning,  "pred_wdl_t",   "Pois + Skellam (range-adjusted), skel"),
  comparison_row(pois_cov_mae_wdl_tuning,  "pred_wdl_raw", "Pois + Skellam (argmax), mae + age"),
  comparison_row(pois_cov_mae_wdl_tuning,  "pred_wdl_t",   "Pois + Skellam (range-adjusted), mae + age"),
  comparison_row(pois_cov_rps_wdl_tuning,  "pred_wdl_raw", "Pois + Skellam (argmax), rps + age"),
  comparison_row(pois_cov_rps_wdl_tuning,  "pred_wdl_t",   "Pois + Skellam (range-adjusted), rps + age"),
  comparison_row(pois_cov_skel_wdl_tuning, "pred_wdl_raw", "Pois + Skellam (argmax), skel + age"),
  comparison_row(pois_cov_skel_wdl_tuning, "pred_wdl_t",   "Pois + Skellam (range-adjusted), skel + age")
)
print(model_comparison_tuning, width = Inf)

# base models only under the original names; base vs + age (range-adjusted only, so the bars
# stay readable) under _before_after
is_age = \(model) grepl("+ age", model, fixed = TRUE)
keep_before_after = \(model) grepl("range-adjusted", model) | model == "Historical Frequency Baseline"

save_plot(plot_model_comparison(model_comparison |> filter(!is_age(Model)), "Baseline vs Skellam (nested outer)"),
          "model_comparison_outer")
save_plot(plot_model_comparison(model_comparison_tuning |> filter(!is_age(Model)), "Baseline vs Skellam (tuning pass)"),
          "model_comparison_tuning")
save_plot(plot_model_comparison(model_comparison |> filter(keep_before_after(Model)),
                                "Baseline vs range-adjusted Skellam, base vs + age (nested outer)"),
          "model_comparison_outer_before_after", width = 10)
save_plot(plot_model_comparison(model_comparison_tuning |> filter(keep_before_after(Model)),
                                "Baseline vs range-adjusted Skellam, base vs + age (tuning pass)"),
          "model_comparison_tuning_before_after", width = 10)

# Section 14: Final model coefficients----
# glmnet coefficients of the final (2028) model, on the original scale of each predictor
# Predictors are team-minus-opponent differences (except GroupStage), so:
#   beta_k        = change in log expected goals (log lambda) per 1-unit difference in x_k
#   exp_beta_k    = exp(beta_k): expected goals are multiplied by this per 1-unit difference
#   sd_x          = standard deviation of x_k in euro
#   beta_k_sd     = beta_k * sd_x: effect of a 1-SD difference, comparable across predictors
#                   (UefaPoints is in the 10,000s and GDP around 1, so raw beta_k aren't comparable)
#   exp_beta_k_sd = exp(beta_k_sd): expected goals are multiplied by this per 1-SD difference
# These are point estimates only (no standard errors: glmnet doesn't provide them)
# glmnet shrinks coefficients toward 0 (attenuated), and an exact 0 means the lasso dropped the variable
# the predictors a final model was fit with (base: predictors, + age: predictors_cov)
final_predictors = function(final) {
  setdiff(broom::tidy(extract_fit_parsnip(final))$term, "(Intercept)")
}

coef_table = function(final) {
  predictor_sd = euro |>
    summarise(across(all_of(final_predictors(final)), sd)) |>
    pivot_longer(everything(), names_to = "term", values_to = "sd_x")

  broom::tidy(extract_fit_parsnip(final)) |>
    select(term, beta_k = estimate) |>
    mutate(exp_beta_k = exp(beta_k)) |>
    left_join(predictor_sd, by = "term") |>
    mutate(beta_k_sd     = beta_k * sd_x,
           exp_beta_k_sd = exp(beta_k_sd))
}

pois_mae_final$selected
pois_mae_coefs = coef_table(pois_mae_final)
pois_mae_coefs

pois_rps_final$selected
pois_rps_coefs = coef_table(pois_rps_final)
pois_rps_coefs

pois_skel_final$selected
pois_skel_coefs = coef_table(pois_skel_final)
pois_skel_coefs

pois_cov_mae_final$selected
pois_cov_mae_coefs = coef_table(pois_cov_mae_final)
pois_cov_mae_coefs

pois_cov_rps_final$selected
pois_cov_rps_coefs = coef_table(pois_cov_rps_final)
pois_cov_rps_coefs

pois_cov_skel_final$selected
pois_cov_skel_coefs = coef_table(pois_cov_skel_final)
pois_cov_skel_coefs

# sanity check: the coefficients reproduce the final model's prediction for the first row
exp(sum(pois_rps_coefs$beta_k * c(1, unlist(euro[1, predictors]))))
predict(extract_workflow(pois_rps_final), new_data = euro[1, ])
exp(sum(pois_cov_rps_coefs$beta_k * c(1, unlist(euro[1, predictors_cov]))))
predict(extract_workflow(pois_cov_rps_final), new_data = euro[1, ])

# before / after: per-SD effects side by side (AgeDifference is new in the + age models)
coef_before_after = full_join(
  pois_rps_coefs     |> select(term, base_exp_beta_k_sd = exp_beta_k_sd),
  pois_cov_rps_coefs |> select(term, age_exp_beta_k_sd  = exp_beta_k_sd),
  by = "term"
)
coef_before_after

# Rate ratios e^beta_k as horizontal bars starting at 1: right of the line = more expected goals,
# left = fewer, per 1-unit difference in the predictor (so bar lengths depend on each predictor's units).
# Bars are drawn as ratio - 1 so they grow out of 1; the axis labels add the 1 back.
# Terms with beta_k = 0 were dropped by the lasso: no bar, just a label
plot_coef_effects = function(coefs, title) {
  effects = coefs |>
    filter(term != "(Intercept)") |>
    mutate(term      = reorder(term, exp_beta_k),
           direction = if_else(exp_beta_k >= 1, "above", "below"),
           # 4 decimals near 1, so e.g. FifaRank's 0.9996 per rank doesn't print as "1.000"
           label     = case_when(beta_k == 0 ~ "dropped by lasso",
                                 abs(exp_beta_k - 1) < 0.005 ~ sprintf("%.4f", exp_beta_k),
                                 TRUE ~ sprintf("%.3f", exp_beta_k)))

  ggplot(effects, aes(x = exp_beta_k - 1, y = term)) +
    geom_col(aes(fill = direction), width = 0.6, show.legend = FALSE) +
    geom_vline(xintercept = 0, colour = "grey30") +
    geom_text(aes(label = label, hjust = if_else(exp_beta_k >= 1, -0.15, 1.15),
                  colour = beta_k == 0), size = 3.5, show.legend = FALSE) +
    scale_x_continuous(labels = \(x) format(x + 1), expand = expansion(mult = 0.3)) +
    scale_fill_manual(values = c(above = "#2c7fb8", below = "#f0a830")) +
    scale_colour_manual(values = c(`FALSE` = "grey15", `TRUE` = "grey50")) +
    labs(title = title,
         subtitle = "Multiplicative change in a team's expected goals per 1-unit difference",
         x = expression("Rate ratio " * e^{beta[k]}), y = NULL,
         caption = "Penalty = .0215 Mixture = 1 selected on MAE") +
    theme_minimal(base_size = 13) +
    theme(plot.title         = element_text(face = "bold"),
          plot.subtitle      = element_text(colour = "grey30"),
          panel.grid.major.y = element_blank(),
          panel.grid.minor   = element_blank())
}

save_plot(plot_coef_effects(pois_cov_mae_coefs, "Poisson incl. Age Final Model: Coefficient Effects"),
          "coef_effects_pois_cov_mae", width = 11, height = 5)

# (football graph4) correlogram 
#        Obs: Lots of correlation between MarketValue, FifaRank, UefaPoints, CLPlayers
euro_cov %>%
  select(MarketValue, GDP, FifaRank, CLPlayers, GroupStage, UefaPoints, AgeDifference) %>%
  mutate(across(everything(), as.numeric)) %>%   #https://dplyr.tidyverse.org/reference/across.html
  cor() %>%
  corrplot(diag = F, tl.srt = 45, tl.col = "black")

# The same correlogram rebuilt in ggplot, so it can be inset into the coefficient plot
# (corrplot draws base graphics, which patchwork can't place without the gridGraphics package).
# Circles sized by |r| and coloured red (-1) to blue (+1) like corrplot; lower triangle only, to fit the inset
plot_cor_inset = function(data, vars) {
  r = data |>
    select(all_of(vars)) |>
    mutate(across(everything(), as.numeric)) |>
    cor()
  cors = as.data.frame(as.table(r)) |>
    rename(x = Var1, y = Var2, r = Freq) |>
    mutate(x = factor(x, levels = vars), y = factor(y, levels = rev(vars))) |>
    filter(as.integer(x) < match(y, vars))

  ggplot(cors, aes(x = x, y = y)) +
    geom_tile(fill = "white", colour = "grey85") +
    geom_point(aes(size = abs(r), colour = r)) +
    scale_size_area(max_size = 7, limits = c(0, 1), guide = "none") +
    scale_colour_gradient2(low = "#b2182b", mid = "white", high = "#2166ac", limits = c(-1, 1),
                           breaks = c(-1, 0, 1)) +
    scale_x_discrete(limits = vars[-length(vars)]) +
    scale_y_discrete(limits = rev(vars[-1])) +
    coord_equal() +
    labs(title = "Predictor Correlations", x = NULL, y = NULL, colour = "r") +
    theme_minimal(base_size = 9) +
    theme(plot.title      = element_text(face = "bold", size = 9),
          axis.text.x     = element_text(angle = 45, hjust = 1, colour = "black"),
          axis.text.y     = element_text(colour = "black"),
          panel.grid      = element_blank(),
          legend.key.height = unit(0.4, "cm"),
          legend.key.width  = unit(0.25, "cm"),
          plot.background = element_rect(fill = "white", colour = "grey70"))
}

coef_cor_vars = c("MarketValue", "GDP", "CLPlayers", "GroupStage", "UefaPoints", "AgeDifference")
coef_effects_pois_cov_mae_cor =
  plot_coef_effects(pois_cov_mae_coefs, "Poisson incl. Age Final Model: Coefficient Effects") +
  patchwork::inset_element(plot_cor_inset(euro_cov, coef_cor_vars),
                           left = 0.66, bottom = 0.02, right = 1, top = 0.62)
save_plot(coef_effects_pois_cov_mae_cor, "coef_effects_pois_cov_mae_cor", width = 11, height = 6.5)

# Section 15: Hyperparameter selection variability----
# Each outer fold runs its own inner tuning. These heatmaps show every candidate's inner CV score
# per outer fold; the black outline is the candidate that fold selected. Lower is better for mae
# and wdl_rps. fill = "std_err" shows how much that candidate's score varies across inner folds
# Shared look for the hyperparameter heatmaps: readable tick labels (penalty in 2 significant digits,
# mixture in 2 decimals), Greek axis titles, no grid lines behind the tiles. Labels are plain strings
# and the axis order is fixed from the grid, so every layer (selections, final pick) lands on the same cells
heat_x = function(penalty) formatC(signif(penalty, 2), format = "g", digits = 2)
heat_y = function(mixture) sprintf("%.2f", mixture)
heat_scales = function(inner) {
  list(scale_x_discrete(limits = heat_x(sort(unique(inner$penalty))), expand = c(0, 0)),
       scale_y_discrete(limits = heat_y(sort(unique(inner$mixture))), expand = c(0, 0)))
}
metric_label = c(mae = "MAE", mse = "MSE", wdl_rps = "RPS", skellam_nll_by_match = "Skellam NLL")
theme_heatmap = function() {
  theme_minimal(base_size = 13) +
    theme(plot.title       = element_text(face = "bold"),
          plot.subtitle    = element_text(colour = "grey30"),
          panel.grid       = element_blank(),
          strip.text       = element_text(face = "bold"),
          axis.text.x      = element_text(angle = 45, hjust = 1),
          legend.key.height = unit(1.2, "cm"))
}

plot_inner_heatmap = function(nested, metric_name, fill_stat = c("mean", "std_err"), title) {
  fill_stat = match.arg(fill_stat)
  inner  = collect_inner_metrics(nested) |> filter(.metric == metric_name)
  chosen = collect_selections(nested)

  ggplot(inner, aes(x = heat_x(penalty), y = heat_y(mixture))) +
    geom_tile(aes(fill = .data[[fill_stat]])) +
    geom_tile(data = chosen, fill = NA, colour = "red", linewidth = 1) +
    facet_wrap(~ id, labeller = as_labeller(\(id) sub("test", "Held Out", id))) +
    scale_fill_viridis_c() +
    heat_scales(inner) +
    labs(title = title, subtitle = "Red outline: candidate selected by that outer fold",
         x = expression("Penalty " * lambda), y = expression("Mixture " * alpha),
         fill = paste("Inner", metric_label[[metric_name]], if (fill_stat == "mean") "Mean" else "Std. Err.")) +
    theme_heatmap()
}

# Across outer folds: each candidate's mean inner score (stat = "mean") or how much it moves
# between outer folds (stat = "sd"). Cells are labelled with the value, outlined in red with the
# number of outer folds that selected them, and the final (2028) model's pick is outlined in white
plot_inner_summary = function(nested, final, metric_name, stat = c("mean", "sd"), title) {
  stat   = match.arg(stat)
  picked = collect_selections(nested) |> count(penalty, mixture, name = "n_folds")
  summary = collect_inner_metrics(nested) |>
    filter(.metric == metric_name) |>
    group_by(penalty, mixture) |>
    summarise(value = if (stat == "mean") mean(mean) else sd(mean), .groups = "drop")

  ggplot(summary, aes(x = heat_x(penalty), y = heat_y(mixture))) +
    geom_tile(aes(fill = value)) +
    # dark text on the bright (high) end of viridis, light text on the dark end
    geom_text(aes(label = sprintf(if (stat == "mean") "%.3f" else "%.4f", value),
                  colour = value > mean(range(value))), size = 4, show.legend = FALSE) +
    scale_colour_manual(values = c(`TRUE` = "grey10", `FALSE` = "grey90")) +
    geom_tile(data = picked, fill = NA, colour = "red", linewidth = 1) +
    geom_label(data = picked, aes(label = paste0(n_folds, "x")), size = 3, vjust = -0.6,
               linewidth = 0, fill = "red", colour = "white") +
    # drawn slightly inside the cell so it stays visible when the final pick is also a red cell
    geom_tile(data = final$selected, width = 0.85, height = 0.85,
              fill = NA, colour = "white", linewidth = 0.7, linetype = "dashed") +
    scale_fill_viridis_c() +
    heat_scales(summary) +
    labs(title = title,
         subtitle = "Red: selected by n outer folds; dashed white: final (2028) model",
         x = expression("Penalty " * lambda), y = expression("Mixture " * alpha),
         fill = paste(if (stat == "mean") "Mean" else "SD", "of Inner", metric_label[[metric_name]], "Across Outer Folds")) +
    guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5)) +
    theme_heatmap() +
    # legend below with a wide horizontal colour bar, so the tiles get the full plot width
    theme(legend.position   = "bottom",
          legend.key.width  = unit(3, "cm"),
          legend.key.height = unit(0.4, "cm"))
}

save_plot(plot_inner_heatmap(pois_mae_nested, "mae", "mean",    "Poisson, Selected on MAE: Inner Mean MAE by Outer Fold"),
          "heatmap_pois_mae_mean", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_mae_nested, "mae", "std_err", "Poisson, Selected on MAE: Inner Std. Err. of MAE by Outer Fold"),
          "heatmap_pois_mae_std_err", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_rps_nested, "wdl_rps", "mean",    "Poisson, Selected on RPS: Inner Mean RPS by Outer Fold"),
          "heatmap_pois_rps_mean", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_rps_nested, "wdl_rps", "std_err", "Poisson, Selected on RPS: Inner Std. Err. of RPS by Outer Fold"),
          "heatmap_pois_rps_std_err", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_skel_nested, "skellam_nll_by_match", "mean",    "Poisson, Selected on Skellam NLL: Inner Mean Skellam NLL by Outer Fold"),
          "heatmap_pois_skel_mean", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_skel_nested, "skellam_nll_by_match", "std_err", "Poisson, Selected on Skellam NLL: Inner Std. Err. of Skellam NLL by Outer Fold"),
          "heatmap_pois_skel_std_err", width = 10, height = 6)

save_plot(plot_inner_summary(pois_mae_nested, pois_mae_final, "mae", "mean", "Poisson, Selected on MAE: Mean Across Outer Folds"),
          "summary_pois_mae_mean", width = 9, height = 6)
save_plot(plot_inner_summary(pois_mae_nested, pois_mae_final, "mae", "sd",   "Poisson, Selected on MAE: SD Across Outer Folds"),
          "summary_pois_mae_sd", width = 9, height = 6)
save_plot(plot_inner_summary(pois_rps_nested, pois_rps_final, "wdl_rps", "mean", "Poisson, Selected on RPS: Mean Across Outer Folds"),
          "summary_pois_rps_mean", width = 9, height = 6)
save_plot(plot_inner_summary(pois_rps_nested, pois_rps_final, "wdl_rps", "sd",   "Poisson, Selected on RPS: SD Across Outer Folds"),
          "summary_pois_rps_sd", width = 9, height = 6)
save_plot(plot_inner_summary(pois_skel_nested, pois_skel_final, "skellam_nll_by_match", "mean", "Poisson, Selected on Skellam NLL: Mean Across Outer Folds"),
          "summary_pois_skel_mean", width = 9, height = 6)
save_plot(plot_inner_summary(pois_skel_nested, pois_skel_final, "skellam_nll_by_match", "sd",   "Poisson, Selected on Skellam NLL: SD Across Outer Folds"),
          "summary_pois_skel_sd", width = 9, height = 6)

save_plot(plot_inner_heatmap(pois_cov_mae_nested, "mae", "mean",    "Poisson + Age, Selected on MAE: Inner Mean MAE by Outer Fold"),
          "heatmap_pois_cov_mae_mean", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_cov_mae_nested, "mae", "std_err", "Poisson + Age, Selected on MAE: Inner Std. Err. of MAE by Outer Fold"),
          "heatmap_pois_cov_mae_std_err", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_cov_rps_nested, "wdl_rps", "mean",    "Poisson + Age, Selected on RPS: Inner Mean RPS by Outer Fold"),
          "heatmap_pois_cov_rps_mean", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_cov_rps_nested, "wdl_rps", "std_err", "Poisson + Age, Selected on RPS: Inner Std. Err. of RPS by Outer Fold"),
          "heatmap_pois_cov_rps_std_err", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_cov_skel_nested, "skellam_nll_by_match", "mean",    "Poisson + Age, Selected on Skellam NLL: Inner Mean Skellam NLL by Outer Fold"),
          "heatmap_pois_cov_skel_mean", width = 10, height = 6)
save_plot(plot_inner_heatmap(pois_cov_skel_nested, "skellam_nll_by_match", "std_err", "Poisson + Age, Selected on Skellam NLL: Inner Std. Err. of Skellam NLL by Outer Fold"),
          "heatmap_pois_cov_skel_std_err", width = 10, height = 6)

save_plot(plot_inner_summary(pois_cov_mae_nested, pois_cov_mae_final, "mae", "mean", "Poisson + Age, Selected on MAE: Mean Across Outer Folds"),
          "summary_pois_cov_mae_mean", width = 9, height = 6)
save_plot(plot_inner_summary(pois_cov_mae_nested, pois_cov_mae_final, "mae", "sd",   "Poisson + Age, Selected on MAE: SD Across Outer Folds"),
          "summary_pois_cov_mae_sd", width = 9, height = 6)
save_plot(plot_inner_summary(pois_cov_rps_nested, pois_cov_rps_final, "wdl_rps", "mean", "Poisson + Age, Selected on RPS: Mean Across Outer Folds"),
          "summary_pois_cov_rps_mean", width = 9, height = 6)
save_plot(plot_inner_summary(pois_cov_rps_nested, pois_cov_rps_final, "wdl_rps", "sd",   "Poisson + Age, Selected on RPS: SD Across Outer Folds"),
          "summary_pois_cov_rps_sd", width = 9, height = 6)
save_plot(plot_inner_summary(pois_cov_skel_nested, pois_cov_skel_final, "skellam_nll_by_match", "mean", "Poisson + Age, Selected on Skellam NLL: Mean Across Outer Folds"),
          "summary_pois_cov_skel_mean", width = 9, height = 6)
save_plot(plot_inner_summary(pois_cov_skel_nested, pois_cov_skel_final, "skellam_nll_by_match", "sd",   "Poisson + Age, Selected on Skellam NLL: SD Across Outer Folds"),
          "summary_pois_cov_skel_sd", width = 9, height = 6)

# Section 15.1: Did the selection metric (mae / rps / skellam nll) change the hyperparameters?----
# per outer fold, side by side
selection_comparison = full_join(
  collect_selections(pois_mae_nested) |> select(id, mae_penalty = penalty, mae_mixture = mixture),
  collect_selections(pois_rps_nested) |> select(id, rps_penalty = penalty, rps_mixture = mixture),
  by = "id"
) |>
  full_join(collect_selections(pois_skel_nested) |> select(id, skel_penalty = penalty, skel_mixture = mixture),
            by = "id") |>
  mutate(mae_vs_rps  = mae_penalty == rps_penalty  & mae_mixture == rps_mixture,
         skel_vs_mae = skel_penalty == mae_penalty & skel_mixture == mae_mixture,
         skel_vs_rps = skel_penalty == rps_penalty & skel_mixture == rps_mixture)
selection_comparison

# same for the + age runs
selection_comparison_cov = full_join(
  collect_selections(pois_cov_mae_nested) |> select(id, mae_penalty = penalty, mae_mixture = mixture),
  collect_selections(pois_cov_rps_nested) |> select(id, rps_penalty = penalty, rps_mixture = mixture),
  by = "id"
) |>
  full_join(collect_selections(pois_cov_skel_nested) |> select(id, skel_penalty = penalty, skel_mixture = mixture),
            by = "id") |>
  mutate(mae_vs_rps  = mae_penalty == rps_penalty  & mae_mixture == rps_mixture,
         skel_vs_mae = skel_penalty == mae_penalty & skel_mixture == mae_mixture,
         skel_vs_rps = skel_penalty == rps_penalty & skel_mixture == rps_mixture)
selection_comparison_cov

# how often each config was picked across the 6 outer folds
agreement(pois_mae_nested)
agreement(pois_rps_nested)
agreement(pois_skel_nested)
agreement(pois_cov_mae_nested)
agreement(pois_cov_rps_nested)
agreement(pois_cov_skel_nested)

# final (2028) models
bind_rows(pois_mae_final$selected |> mutate(select = "mae"),
          pois_rps_final$selected |> mutate(select = "rps"),
          pois_skel_final$selected |> mutate(select = "skel")) -> pois_select_rule_hyperparams_final

bind_rows(pois_cov_mae_final$selected  |> mutate(select = "mae"),
          pois_cov_rps_final$selected  |> mutate(select = "rps"),
          pois_cov_skel_final$selected |> mutate(select = "skel")) -> pois_cov_select_rule_hyperparams_final

# before / after: final (2028) hyperparameters per selection metric
pois_select_rule_hyperparams_final_before_after = bind_rows(
  pois_select_rule_hyperparams_final     |> mutate(covariates = "base"),
  pois_cov_select_rule_hyperparams_final |> mutate(covariates = "+ age")
) |>
  select(select, covariates, penalty, mixture) |>
  arrange(select, covariates)
pois_select_rule_hyperparams_final_before_after

pois_select_rule_hyperparams_final

# Same check within ONE run: the rps run also computed mae on the inner folds, so we can read off
# what mae would have picked from exactly the same fits (no run-to-run randomness)
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
  best_inner(pois_rps_nested, "mae")     |> rename(mae_penalty = penalty, mae_mixture = mixture),
  by = "id"
) |>
  mutate(same = mae_penalty == rps_penalty & mae_mixture == rps_mixture)
within_rps_run

# The skellam run computed all three criteria on the same inner fits, so this is the cleanest
# three-way comparison of what each metric would pick
within_skel_run = best_inner(pois_skel_nested, "skellam_nll_by_match") |>
  rename(skel_penalty = penalty, skel_mixture = mixture) |>
  full_join(best_inner(pois_skel_nested, "wdl_rps") |> rename(rps_penalty = penalty, rps_mixture = mixture), by = "id") |>
  full_join(best_inner(pois_skel_nested, "mae")     |> rename(mae_penalty = penalty, mae_mixture = mixture), by = "id") |>
  mutate(skel_vs_mae = skel_penalty == mae_penalty & skel_mixture == mae_mixture,
         skel_vs_rps = skel_penalty == rps_penalty & skel_mixture == rps_mixture)
within_skel_run

# Section 15.2: Why + age classifies better although AgeDifference is dropped----
# Per held-out Euro: the lambda / alpha that fold's inner tuning picked and the draw threshold t it
# was given (tuned on the other years' outer predictions), without and with AgeDifference.
# The inner mae surface is flat, so the extra column flips near-ties in some folds; the changed
# probabilities then move t in every fold. The final (2028) row uses t_final (Section 12.3)
fold_picks = function(nested, final, wdl_outer, t_final) {
  per_year = collect_selections(nested) |>
    mutate(held_out = sub("test ", "", id)) |>
    select(held_out, penalty, mixture) |>
    left_join(wdl_outer |> distinct(Year, draw_t) |> mutate(held_out = as.character(Year)),
              by = "held_out") |>
    select(held_out, penalty, mixture, draw_t) |>
    arrange(held_out)
  final_row = final$selected |>
    transmute(held_out = "Final (2028)", penalty, mixture, draw_t = t_final)
  bind_rows(per_year, final_row)
}

age_fold_picks = full_join(
  fold_picks(pois_mae_nested, pois_mae_final, pois_mae_wdl_outer, pois_mae_t_final) |>
    rename(base_penalty = penalty, base_mixture = mixture, base_t = draw_t),
  fold_picks(pois_cov_mae_nested, pois_cov_mae_final, pois_cov_mae_wdl_outer, pois_cov_mae_t_final) |>
    rename(age_penalty = penalty, age_mixture = mixture, age_t = draw_t),
  by = "held_out"
) |>
  mutate(same_hyperparams = base_penalty == age_penalty & base_mixture == age_mixture)
age_fold_picks

# pooled nested outer classification scores (range-adjusted) for the bottom rows of the slide table
age_fold_scores = bind_rows(pois_results |> mutate(covariates = "base"),
                            pois_cov_results |> mutate(covariates = "age")) |>
  filter(select == "mae", preds == "nested_outer", metric %in% c("macro_f1_t", "accuracy_t")) |>
  select(metric, covariates, value) |>
  pivot_wider(names_from = covariates, values_from = value)
age_fold_scores

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

pois_mae_wdl_plots = plot_wdl_probs(pois_mae_wdl_outer, "pois, selected on mae")
pois_rps_wdl_plots = plot_wdl_probs(pois_rps_wdl_outer, "pois, selected on rps")
pois_skel_wdl_plots = plot_wdl_probs(pois_skel_wdl_outer, "pois, selected on skellam nll")
pois_cov_mae_wdl_plots = plot_wdl_probs(pois_cov_mae_wdl_outer, "pois + age, selected on mae")
pois_cov_rps_wdl_plots = plot_wdl_probs(pois_cov_rps_wdl_outer, "pois + age, selected on rps")
pois_cov_skel_wdl_plots = plot_wdl_probs(pois_cov_skel_wdl_outer, "pois + age, selected on skellam nll")

iwalk(pois_mae_wdl_plots, \(plot, name) save_plot(plot, paste0("wdl_pois_mae_", name)))
iwalk(pois_rps_wdl_plots, \(plot, name) save_plot(plot, paste0("wdl_pois_rps_", name)))
iwalk(pois_skel_wdl_plots, \(plot, name) save_plot(plot, paste0("wdl_pois_skel_", name)))
iwalk(pois_cov_mae_wdl_plots, \(plot, name) save_plot(plot, paste0("wdl_pois_cov_mae_", name)))
iwalk(pois_cov_rps_wdl_plots, \(plot, name) save_plot(plot, paste0("wdl_pois_cov_rps_", name)))
iwalk(pois_cov_skel_wdl_plots, \(plot, name) save_plot(plot, paste0("wdl_pois_cov_skel_", name)))

# Section 17: Poisson assumption diagnostics (final models)----
# Checked on the final models (fit on all data): they're the models we interpret and use for 2028.
# The hundreds of nested fits evaluate the tuning procedure; they share these assumptions
# pois_mae_final and pois_rps_final picked the same hyperparameters, and pois_skel_final is
# nearly the same model, so the summary table covers all three and the plots can use any one

# Fitted expected goals (lambda) and Pearson residuals for every row of euro
#   Pearson residual = (observed - expected) / sqrt(expected): for a Poisson, Var = mean = lambda,
#   so these should have mean 0 and variance 1 if the model is right
final_fitted = function(final) {
  euro |>
    mutate(.row = row_number(),
           lambda = predict(extract_workflow(final), new_data = euro)$.pred,
           resid_pearson = (Goals - lambda) / sqrt(lambda))
}

# Dispersion: sum of squared Pearson residuals / (n - p)
#   ~1: variance matches the mean, as Poisson assumes
#   >1: overdispersion, goals vary more than Poisson allows (probabilities too confident)
#   <1: underdispersion
# p = number of non-zero coefficients incl. the intercept (the usual degrees of freedom for a lasso)
dispersion_stat = function(fitted, final) {
  p = sum(broom::tidy(extract_fit_parsnip(final))$estimate != 0)
  sum(fitted$resid_pearson^2) / (nrow(fitted) - p)
}

# Within-match independence: correlation of team 1's and team 2's Pearson residuals in the same match
# Skellam assumes the two teams' goals are independent given lambda1 and lambda2 (correlation 0)
#   > 0: open games / cagey games, both teams over- or under-score together -> more draws than Skellam predicts
#   < 0: one team's goals come at the other's expense
within_match_cor = function(fitted) {
  pairs = fitted |>
    arrange(.row) |>
    group_by(Year, match_id) |>
    summarise(resid1 = first(resid_pearson), resid2 = last(resid_pearson), .groups = "drop")
  cor.test(pairs$resid1, pairs$resid2)
}

diagnostics_row = function(final, select) {
  fitted = final_fitted(final)
  wm = within_match_cor(fitted)
  tibble(select           = select,
         dispersion       = dispersion_stat(fitted, final),
         within_match_cor = unname(wm$estimate),
         cor_95_lo        = wm$conf.int[1],
         cor_95_hi        = wm$conf.int[2],
         cor_p_value      = wm$p.value)
}

final_diagnostics = bind_rows(
  diagnostics_row(pois_mae_final,  "mae"),
  diagnostics_row(pois_rps_final,  "rps"),
  diagnostics_row(pois_skel_final, "skel"),
  diagnostics_row(pois_cov_mae_final,  "mae + age"),
  diagnostics_row(pois_cov_rps_final,  "rps + age"),
  diagnostics_row(pois_cov_skel_final, "skel + age")
)
print(final_diagnostics, width = Inf)

# Log-linear form: the model assumes log(expected goals) is a straight line in each predictor.
# Binned mean Pearson residual (+/- 2 SE) against each predictor and against the fitted lambda:
# points scattered around 0 = fine; a curve or trend = the straight-line assumption misses something.
# Continuous variables are split into 10 equal-count bins; GroupStage keeps its 2 values
plot_loglinear_check = function(final, label) {
  fitted = final_fitted(final)
  fitted |>
    select(resid_pearson, lambda, all_of(final_predictors(final))) |>
    pivot_longer(-resid_pearson, names_to = "variable", values_to = "x") |>
    group_by(variable) |>
    mutate(bin = if (n_distinct(x) <= 10) as.integer(factor(x)) else ntile(x, 10)) |>
    group_by(variable, bin) |>
    summarise(x_mean     = mean(x),
              resid_mean = mean(resid_pearson),
              resid_se   = sd(resid_pearson) / sqrt(n()),
              .groups    = "drop") |>
    mutate(variable = factor(variable, levels = c("lambda", final_predictors(final)))) |>
    ggplot(aes(x = x_mean, y = resid_mean)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_pointrange(aes(ymin = resid_mean - 2 * resid_se, ymax = resid_mean + 2 * resid_se)) +
    facet_wrap(~ variable, scales = "free_x") +
    labs(title = paste("Log-linear form check:", label),
         subtitle = "Binned mean Pearson residual (+/- 2 SE); lambda = fitted expected goals",
         x = "bin mean of x", y = "mean Pearson residual") +
    theme_minimal()
}

save_plot(plot_loglinear_check(pois_mae_final,  "pois final, selected on mae"),         "loglinear_pois_mae",  width = 10, height = 6)
save_plot(plot_loglinear_check(pois_rps_final,  "pois final, selected on rps"),         "loglinear_pois_rps",  width = 10, height = 6)
save_plot(plot_loglinear_check(pois_skel_final, "pois final, selected on skellam nll"), "loglinear_pois_skel", width = 10, height = 6)
save_plot(plot_loglinear_check(pois_cov_mae_final,  "pois + age final, selected on mae"),         "loglinear_pois_cov_mae",  width = 10, height = 6)
save_plot(plot_loglinear_check(pois_cov_rps_final,  "pois + age final, selected on rps"),         "loglinear_pois_cov_rps",  width = 10, height = 6)
save_plot(plot_loglinear_check(pois_cov_skel_final, "pois + age final, selected on skellam nll"), "loglinear_pois_cov_skel", width = 10, height = 6)

# Poisson QQ plot: observed goals against Poisson theoretical quantiles. Every row has its own
# lambda_i, so the theoretical distribution is the mixture of the fitted Pois(lambda_i):
#   F(k) = mean_i ppois(k, lambda_i), quantile(p) = smallest k with F(k) >= p, at p = ppoints(n)
# Goals are counts, so points stack on whole numbers; point size = how many rows sit there.
# On the 45-degree line = the fitted Poisson model reproduces the goal distribution
plot_poisson_qq = function(final, title, label_size = 3.5) {
  fitted = final_fitted(final)
  n      = nrow(fitted)
  k      = 0:(max(fitted$Goals) + 5)
  mix_cdf = sapply(k, \(kk) mean(ppois(kk, fitted$lambda)))
  theoretical = sapply(ppoints(n), \(p) k[which(mix_cdf >= p)[1]])

  tibble(theoretical = theoretical, observed = sort(fitted$Goals)) |>
    count(theoretical, observed) |>
    ggplot(aes(x = theoretical, y = observed)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey50", linetype = "dashed") +
    geom_point(aes(size = n), colour = "#2c7fb8", alpha = 0.8) +
    geom_text(aes(label = n), nudge_x = 0.4, hjust = 0, size = label_size, colour = "grey30") +
    scale_x_continuous(breaks = k) +
    scale_y_continuous(breaks = k) +
    scale_size_area(max_size = 12, guide = "none") +
    coord_equal(xlim = c(-0.3, max(theoretical, fitted$Goals) + 0.8), ylim = c(-0.3, max(fitted$Goals) + 0.3)) +
    labs(title = title,
         subtitle = "Fitted Poisson mixture quantiles; labels = number of rows",
         x = "Theoretical quantile (Poisson)", y = "Observed goals") +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold"), panel.grid.minor = element_blank())
}

save_plot(plot_poisson_qq(pois_mae_final, "Poisson QQ Plot: Final Model Selected on MAE"),
          "qq_pois_mae", width = 7, height = 7)

# Section 18: LaTeX tables for the presentation----
# Each table is written with kable (booktabs) to presentations/football-week-4/tables/<name>.tex
# and pulled into the slides with \input{tables/<name>.tex}, so the slides update with the results.
# Debug runs (SIMPLE = T) write to results_debug/tables instead, so they can't overwrite these
TABLE_DIR = if (SIMPLE) file.path(RESULTS_DIR, "tables") else "presentations/football-week-4/tables"
dir.create(TABLE_DIR, showWarnings = FALSE, recursive = TRUE)

options(knitr.kable.NA = "--")

save_table = function(df, name, digits = 3, ...) {
  tab = knitr::kable(df, format = "latex", booktabs = TRUE, digits = digits, linesep = "", ...)
  writeLines(tab, file.path(TABLE_DIR, paste0(name, ".tex")))
  invisible(tab)
}

# numbers to 3 significant digits without scientific notation (coefficients span 1e-4 to 1e4)
fmt_sig = function(x) ifelse(is.na(x), NA, sub("\\.$", "", formatC(signif(x, 3), format = "fg", digits = 3, flag = "#")))
# penalties span 1e-10 to 1, so these keep scientific notation
fmt_penalty = function(x) formatC(x, format = "g", digits = 3)

# short labels keep the tables narrow, so they don't have to be shrunk to fit a slide
# "0.317 (0.011)" -> "0.317 (.011)" to keep the table narrow
save_table(nested_eval_report |>
             select(metric, baseline, pois_mae, pois_rps, pois_skel) |>
             mutate(across(-metric, \(x) if_else(x == "NA (NA)", NA_character_, gsub("\\(0\\.", "(.", x)))),
           "nested_eval_report",
           col.names = c("metric", "baseline", "mae", "rps", "skellam"))

# "Pois + Skellam (range-adjusted), mae" -> "mae, range"
short_model = function(model) {
  model |>
    sub(pattern = "Historical Frequency Baseline", replacement = "baseline") |>
    sub(pattern = "Pois \\+ Skellam \\((.*)\\), (.*)", replacement = "\\2, \\1") |>
    sub(pattern = "range-adjusted", replacement = "range")
}

# probability scores are the same for argmax and range-adjusted, so one row per model
save_table(model_comparison |>
             filter(!grepl("range-adjusted", Model), !grepl("+ age", Model, fixed = TRUE)) |>
             mutate(Model = sub(", argmax", "", short_model(Model))) |>
             select(Model, Log_Loss, Brier_Score, RPS),
           "model_comparison_prob",
           col.names = c("model", "log loss", "Brier", "RPS"))
save_table(model_comparison |>
             filter(!grepl("+ age", Model, fixed = TRUE)) |>
             mutate(Model = short_model(Model)) |>
             select(Model, Accuracy, Macro_F1, Draws_Predicted, Draw_Precision, Draw_Recall, Draw_F1),
           "model_comparison_class",
           col.names = c("model", "accuracy", "macro F1", "draws called", "draw prec.", "draw recall", "draw F1"))

coef_for_table = function(coefs) {
  coefs |>
    mutate(across(c(beta_k, exp_beta_k, sd_x, beta_k_sd, exp_beta_k_sd), fmt_sig))
}
coef_col_names = c("term", "$\\beta_k$", "$e^{\\beta_k}$", "$sd_x$", "$\\beta_k \\, sd_x$", "$e^{\\beta_k sd_x}$")
save_table(coef_for_table(pois_mae_coefs),  "pois_mae_coefs",  col.names = coef_col_names, escape = FALSE)
save_table(coef_for_table(pois_rps_coefs),  "pois_rps_coefs",  col.names = coef_col_names, escape = FALSE)
save_table(coef_for_table(pois_skel_coefs), "pois_skel_coefs", col.names = coef_col_names, escape = FALSE)

# one "penalty / mixture" cell per selection metric keeps it to 4 columns
lambda_alpha = function(penalty, mixture) paste(fmt_penalty(penalty), "/", sprintf("%.2f", mixture))
save_table(selection_comparison |>
             mutate(id = sub("test ", "", id),
                    mae  = lambda_alpha(mae_penalty,  mae_mixture),
                    rps  = lambda_alpha(rps_penalty,  rps_mixture),
                    skel = lambda_alpha(skel_penalty, skel_mixture)) |>
             select(id, mae, rps, skel) |>
             arrange(id),
           "selection_comparison",
           col.names = c("held out", "mae $\\lambda$ / $\\alpha$", "rps $\\lambda$ / $\\alpha$",
                         "skel $\\lambda$ / $\\alpha$"),
           escape = FALSE)

save_table(pois_select_rule_hyperparams_final |>
             mutate(penalty = fmt_penalty(penalty)) |>
             select(select, penalty, mixture),
           "pois_select_rule_hyperparams_final", digits = 2,
           col.names = c("selected on", "penalty $\\lambda$", "mixture $\\alpha$"), escape = FALSE)

save_table(final_diagnostics |>
             filter(!grepl("+ age", select, fixed = TRUE)) |>
             mutate(cor_95 = sprintf("[%.2f, %.2f]", cor_95_lo, cor_95_hi)) |>
             select(select, dispersion, within_match_cor, cor_95, cor_p_value),
           "final_diagnostics",
           col.names = c("selected on", "dispersion", "match $r$", "95\\% CI", "$p$"),
           escape = FALSE)

# Slide-sized copies of the figures used in the presentation: a smaller canvas makes the
# text relatively larger once the image is scaled into a slide column
SLIDE_FIG_DIR = if (SIMPLE) file.path(RESULTS_DIR, "tables") else "presentations/football-week-4"
ggsave(file.path(SLIDE_FIG_DIR, "wdl_pois_rps_box.png"), pois_rps_wdl_plots$box + labs(title = NULL),
       width = 6, height = 4, bg = "white")
ggsave(file.path(SLIDE_FIG_DIR, "loglinear_pois_rps.png"),
       plot_loglinear_check(pois_rps_final, "final model") + labs(title = NULL, subtitle = NULL),
       width = 6.5, height = 4.5, bg = "white")
ggsave(file.path(SLIDE_FIG_DIR, "qq_pois_mae.png"),
       plot_poisson_qq(pois_mae_final, "Poisson QQ Plot: Final Model (MAE)", label_size = 5) +
         labs(subtitle = NULL) + theme(text = element_text(size = 17), axis.title = element_text(size = 14),
               plot.title = element_text(size = 15, face = "bold"), plot.title.position = "plot"),
       width = 5, height = 5.4, bg = "white")

# Before / after tables: base covariates vs + AgeDifference
save_table(nested_eval_report |>
             select(metric, baseline, pois_mae, pois_cov_mae, pois_rps, pois_cov_rps, pois_skel, pois_cov_skel) |>
             mutate(across(-metric, \(x) if_else(x == "NA (NA)", NA_character_, gsub("\\(0\\.", "(.", x)))),
           "nested_eval_report_before_after",
           col.names = c("metric", "baseline", "mae", "mae + age", "rps", "rps + age", "skellam", "skellam + age"))

save_table(model_comparison |>
             filter(grepl("range-adjusted", Model) | Model == "Historical Frequency Baseline") |>
             mutate(Model = short_model(Model)) |>
             select(Model, Accuracy, Macro_F1, Draws_Predicted, Draw_Precision, Draw_Recall, Draw_F1),
           "model_comparison_class_before_after",
           col.names = c("model", "accuracy", "macro F1", "draws called", "draw prec.", "draw recall", "draw F1"))

save_table(coef_for_table(pois_cov_mae_coefs),  "pois_cov_mae_coefs",  col.names = coef_col_names, escape = FALSE)
save_table(coef_for_table(pois_cov_rps_coefs),  "pois_cov_rps_coefs",  col.names = coef_col_names, escape = FALSE)
save_table(coef_for_table(pois_cov_skel_coefs), "pois_cov_skel_coefs", col.names = coef_col_names, escape = FALSE)

save_table(coef_before_after |> mutate(across(-term, fmt_sig)),
           "coef_before_after",
           col.names = c("term", "base $e^{\\beta_k sd_x}$", "+ age $e^{\\beta_k sd_x}$"), escape = FALSE)

save_table(selection_comparison_cov |>
             mutate(id = sub("test ", "", id),
                    mae  = lambda_alpha(mae_penalty,  mae_mixture),
                    rps  = lambda_alpha(rps_penalty,  rps_mixture),
                    skel = lambda_alpha(skel_penalty, skel_mixture)) |>
             select(id, mae, rps, skel) |>
             arrange(id),
           "selection_comparison_cov",
           col.names = c("held out", "mae $\\lambda$ / $\\alpha$", "rps $\\lambda$ / $\\alpha$",
                         "skel $\\lambda$ / $\\alpha$"),
           escape = FALSE)

save_table(pois_select_rule_hyperparams_final_before_after |>
             mutate(penalty = fmt_penalty(penalty)),
           "pois_select_rule_hyperparams_final_before_after", digits = 2,
           col.names = c("selected on", "covariates", "penalty $\\lambda$", "mixture $\\alpha$"), escape = FALSE)

# Per-fold picks with and without age (Section 15.2), for the "Effect of AgeDifference" slide.
# Written by hand rather than kable: grouped headers and the spanning score rows need kableExtra,
# which isn't installed. Cells where the two runs picked different hyperparameters are bold
fmt_pick = function(x, differs) {
  x = trimws(x)
  ifelse(differs, paste0("\\textbf{", x, "}"), x)
}
age_fold_rows = age_fold_picks |>
  mutate(differs = !same_hyperparams,
         row = paste(held_out,
                     fmt_pick(fmt_penalty(base_penalty), differs), fmt_pick(as.character(signif(base_mixture, 2)), differs),
                     sprintf("%.3g", base_t),
                     fmt_pick(fmt_penalty(age_penalty), differs), fmt_pick(as.character(signif(age_mixture, 2)), differs),
                     sprintf("%.3g", age_t),
                     sep = " & "))
score_row = function(name, metric_name) {
  s = filter(age_fold_scores, metric == metric_name)
  sprintf("%s & \\multicolumn{3}{c}{%.3f} & \\multicolumn{3}{c}{%.3f} \\\\", name, s$base, s$age)
}
writeLines(c(
  "\\begin{tabular}{lrrr@{\\hspace{10pt}}rrr}",
  "\\toprule",
  "& \\multicolumn{3}{c}{\\textbf{Without Age}} & \\multicolumn{3}{c}{\\textbf{$+$Age}} \\\\",
  "\\cmidrule(r{10pt}){2-4} \\cmidrule(l){5-7}",
  "Held out & $\\lambda$ & $\\alpha$ & $t$ & $\\lambda$ & $\\alpha$ & $t$ \\\\",
  "\\midrule",
  paste(head(age_fold_rows$row, -1), "\\\\"),
  "\\midrule",
  paste(tail(age_fold_rows$row, 1), "\\\\"),
  "\\midrule",
  score_row("Macro F1", "macro_f1_t"),
  score_row("Accuracy", "accuracy_t"),
  "\\bottomrule",
  "\\end{tabular}"
), file.path(TABLE_DIR, "age_fold_picks.tex"))

# Draw-range tuning results without / with age (mae selection): the leave-one-Euro-out t's,
# the 2028 t_final, and what the range-adjusted rule then does on the pooled outer matches
t_summary = function(t) {
  counts = sort(table(t), decreasing = TRUE)
  paste(sprintf("%s (%dx)", names(counts), as.integer(counts)), collapse = ", ")
}
draw_row = function(model) filter(model_comparison, Model == model)
age_draw_base = draw_row("Pois + Skellam (range-adjusted), mae")
age_draw_age  = draw_row("Pois + Skellam (range-adjusted), mae + age")
age_draw_tuning = tibble(
  quantity = c("$t$ per held-out Euro", "Final $t$ (2028)", "Draws called (of 246)",
               "Draw recall", "Macro F1", "Accuracy"),
  base = c(t_summary(head(age_fold_picks$base_t, -1)), tail(age_fold_picks$base_t, 1),
           age_draw_base$Draws_Predicted,
           sprintf("%.3f", c(age_draw_base$Draw_Recall, age_draw_base$Macro_F1, age_draw_base$Accuracy))),
  age  = c(t_summary(head(age_fold_picks$age_t, -1)), tail(age_fold_picks$age_t, 1),
           age_draw_age$Draws_Predicted,
           sprintf("%.3f", c(age_draw_age$Draw_Recall, age_draw_age$Macro_F1, age_draw_age$Accuracy)))
)
age_draw_tuning
save_table(age_draw_tuning, "age_draw_tuning", align = "lrr",
           col.names = c("", "Without Age", "$+$Age"), escape = FALSE)

save_table(final_diagnostics |>
             mutate(cor_95 = sprintf("[%.2f, %.2f]", cor_95_lo, cor_95_hi)) |>
             select(select, dispersion, within_match_cor, cor_95, cor_p_value),
           "final_diagnostics_before_after",
           col.names = c("selected on", "dispersion", "match $r$", "95\\% CI", "$p$"),
           escape = FALSE)
