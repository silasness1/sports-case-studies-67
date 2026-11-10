# README: now that we've each tried to get some results about nnet, random forest, xgboost, and poisson 
# separately, this script tries to replicate it using a more streamlined/organized high-level R packages
# namely parsnip models, workflows, and nestedtune


# HIGHLEVEL 

# Important objects: 
#   - resample_struct
#   - metrics_goals
#   - model_class_wfs
#   - model_class_grids
#   - model_class_metrics
#   - model_class_tuned

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
library(knitr)
# install.packages("pak")
# pak::pak("tidymodels/nestedtune")
library(nestedtune)
library(lubridate)
library(skellam)
library(mirai)
library(tailor)

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

# adding weights to (at least) outer layer (in approximation of pooled estimates)
nested_leave_one_year = add_resample_weights(
  rset = nested_leave_one_year, 
  weights = calculate_resample_weights(nested_leave_one_year)
) 

for (i in 1:6) { # I don't know whether the inner weights are actually supported
  nested_leave_one_year$inner_resamples[[i]] = add_resample_weights(
    rset = nested_leave_one_year$inner_resamples[[i]], 
    weights = calculate_resample_weights(nested_leave_one_year$inner_resamples[[i]])
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
  name = "skellam_nll",
  aggregate = function(x) mean(x$.estimate, na.rm = TRUE),
  direction = "minimize"
)

# tells new_groupwise_metric which variable to group_by
skellam_nll_by_match <- skellam_nll_by_match(match_id)

# WDL metrics: each sees one match at a time (2 rows, grouped by match_id)
#   truth    = c(goals team 1, goals team 2)
#   estimate = c(lambda team 1, lambda team 2)
# WDL is from the first row's perspective

# Section 4.2: Implementing yardstick compatible classification metrics all at once----
  # Basically, "tune" package didn't like I was using poisson "regression" based parsnip model 
  # and mixing with classification based metrics (so here had to make my own)

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

# Section 4.3: Declare metrics for nested_grid_tune in  Win Draw Loss & Goals Prediction Cases----
# Define evaluation metrics for WDL (first metric is what selection_rule("best") uses)
metrics_wdl = yardstick::metric_set(
  wdl_rps,
  wdl_brier,
  wdl_accuracy,
  wdl_accuracy_range
)

# Define evaluation metrics to be used across all models for goals (later define for WDL)
metrics_goals = yardstick::metric_set(
  rmse,
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


# option to reduce runtime when debugging
if (SIMPLE == T) {
  N_TREES = 100
  
  rf_grid <- expand.grid(,
    mtry            = 3:length(predictors),
    min_n           = c(3, 5), 
    sample.fraction = c(0.7, 0.85)
  )
}

# FULL
rf_grid <- expand.grid(
  mtry            = 1:length(predictors),
  min_n           = c(3, 5, 10, 15),
  sample.fraction = c(0.7, 0.85, 1.0)
)

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

# Section 9: Setting up Model_class iterators----
model_class_wfs = list(nnet = nnet_wf, 
                       pois = pois_wf,
                       rf   = rf_wf, 
                       xg   = xg_wf)

model_class_grids = list(nnet = nnet_grid, 
                         pois = pois_grid,
                         rf   = rf_grid, 
                         xg   = xg_grid)

model_class_resamples = list(nnet = resample_struct,
                             pois = resample_struct,
                             rf   = resample_struct, 
                             xg   = resample_struct)

model_class_metrics = list(nnet = metrics_goals, 
                           pois = metrics_goals,
                           rf   = metrics_goals,
                           xg   = metrics_goals)

# Section 10: Running nested_tune_grid for each model_class----

model_class_vec = c(
  "rf", 
  "nnet",
  "xg", 
  "pois"
)

nested_tune_grid_mirai = function(...) {
  
  # Control parallelization
  mirai::daemons(2) # 0 vs 2 makes a difference 3,4, no real difference.
  
  # Check time to run
  print(system.time({tuned = nested_tune_grid(...)}))
  
  mirai::daemons(0)
  
  return(tuned)
}

run_nested_tune_for_model = function(model_class) {
  
    
  tuned = nested_tune_grid_mirai(
    object     = model_class_wfs[[model_class]], 
    resamples  = model_class_resamples[[model_class]], 
    grid       = model_class_grids[[model_class]], 
    metrics    = model_class_metrics[[model_class]], 
    select     = selection_rule(                      #TODO: reconsider
      rule = "desirability", 
      minimize(rmse),
      minimize(poisson_log_loss)
    ),
    control = control_grid( # reaches inner tune grid in every fold
      verbose = FALSE, 
      extract = NULL,   # can save other intermediates from the nested cv & call collect_extracts() 
      save_pred = TRUE, # The inner run's predictions and extracts are still discarded
                        # Use extract = function(x) x to keep each fold's fitted workflow
    )
  )
  return(tuned)
}

# Switch to vectorized after debugging a bit
# model_class_fits = map(model_class_vec, run_nested_tune_for_model)

pois_tuned = run_nested_tune_for_model("pois") # simplified run:  NA, full run:  23s
# rf_tuned   = run_nested_tune_for_model("rf")   # simplified run: 40s, full run: 540s (9 min)
# xg_tuned   = run_nested_tune_for_model("xg")   # simplified run: 63s, full run: 125s (2 min)
# nnet_tuned = run_nested_tune_for_model("nnet") # simplified run:  NA, full run:  46s

model_class_tuned = list(pois = pois_tuned, 
                         rf   = rf_tuned, 
                         xg   = xg_tuned, 
                         nnet = nnet_tuned)

# Section 11: Inspecting results----
summary(pois_tuned)
summary(rf_tuned)
summary(xg_tuned)
summary(nnet_tuned)

model_class_comparison = Reduce(
  rbind, 
  map2(model_class_tuned, 
       names(model_class_tuned), 
       function(x,y){
         model_compare_metrics = collect_metrics(x)
         model_compare_metrics$model_class = y                      
         return(model_compare_metrics)
       }
  )
)

model_class_selections = map(model_class_tuned, collect_selections)


# Section 12: Rerunning but with tailor to convert to WDL----

collect_metrics(pois_tuned)
collect_predictions(pois_tuned)
collect_selections(pois_tuned)



# NOTE: the tailor approach below doesn't work in tailor 0.1.0: tailor::fit()
# drops every column but Goals/.pred (so match_id is gone), and .by inside
# adjust_predictions_custom() becomes a new column instead of grouping.
# WDL tuning is done with metrics_wdl instead (see pois_wdl below).
raw_skellam_wdl_tailor = tailor() |> adjust_predictions_custom(
    raw_pwin = pskellam(
      q = 0, 
      lambda1 = first(.pred), 
      lambda2 = last(.pred), 
      lower.tail = F # strictly > 0 
    ),
    raw_ploss = pskellam(
      q = -1, 
      lambda1 = first(.pred), 
      lambda2 = last(.pred), 
      lower.tail = T # less than or eq -1
    ),
    raw_pdraw = dskellam(
      x = 0,         # eq 0 case
      lambda1 = first(.pred), 
      lambda2 = last(.pred)
    ),
    raw_maxp = pmax(raw_pwin, raw_ploss, raw_pdraw), 
    pred_wdl = case_when(
      raw_maxp == raw_pwin ~ "win",
      raw_maxp == raw_ploss ~ "loss",
      raw_maxp == raw_pdraw ~ "draw"
    ),
    actual_wdl = case_when(
      first(Goals) > last(Goals) ~ "win",
      first(Goals) == last(Goals) ~ "draw",
      first(Goals) < last(Goals) ~ "loss"
    ),
    .by = match_id,
    .pkgs = c("skellam", "dplyr")
)

# TODO: Try to do the threshold approach
# threshold_skellam_wdl_tailor = raw_skellam_wdl_tailor |> 
#   adjust_probability_threshold()
  
range_skellam_wdl_tailor = raw_skellam_wdl_tailor |> 
  adjust_predictions_custom(
    raw_wdl = pred_wdl,
    pred_wdl = replace_when(
      pred_wdl,
      pmax(raw_pwin, raw_ploss, raw_pdraw) - pmin(raw_pwin, raw_ploss, raw_pdraw) < .15 ~ "draw"
    )
  )

# brier_wdl <- function(data, truth, estimate, ...) {
#   
#   truth <- rlang::enquo(truth)
#   estimate <- rlang::enquos(estimate)
#   
#   truth_values <- dplyr::pull(data, !!truth)
#   
#   pred_values <- dplyr::select(data, !!!estimate) |>
#     as.matrix()
#   
#   actual_values <- model.matrix(~ factor(truth_values) - 1)
#   
#   score <- rowSums((pred_values - actual_values)^2)
#   
#   tibble::tibble(.estimate = mean(score))
# }
# brier_wdl <- yardstick::new_numeric_metric(
#   brier_wdl,
#   direction = "minimize"
# )
# 
# metrics_wdl = metric_set(
#   rmse               #TODO patch for brier later
# )


pois_wf |>
  add_tailor(range_skellam_wdl_tailor) -> pois_wf_wdl_range

pois_wf |>
  add_tailor(raw_skellam_wdl_tailor) -> pois_wf_wdl_raw

tune_grid(object = pois_wf, 
          resamples = nested_leave_one_year$inner_resamples[[1]], 
          grid = pois_grid, 
          metrics = metrics_goals, 
          control = control_grid(
            save_pred = T
          )) -> check


# WDL-based tuning: same workflow, no tailor, tuned on metrics_wdl
pois_wdl = nested_tune_grid_mirai(            # full run: ~240s
  object     = pois_wf,
  resamples  = nested_leave_one_year,
  grid       = pois_grid,
  metrics    = metrics_wdl,
  select     = selection_rule(rule = "best"), # picks on wdl_rps (first in metrics_wdl)
  control = control_grid(
    verbose = FALSE,
    extract = NULL,
    save_pred = TRUE
  )
)

collect_metrics(pois_wdl)
collect_selections(pois_wdl)

# pois_wdl_range = nested_tune_grid_mirai(
#   object     = pois_wf_wdl_range, 
#   resamples  = nested_leave_one_year, 
#   grid       = pois_grid, 
#   metrics    = metrics_wdl, 
#   select     = selection_rule(                      #TODO: reconsider
#     rule = "desirability", 
#     minimize(rmse)
#   ),
#   control = control_grid( # reaches inner tune grid in every fold
#     verbose = FALSE, 
#     extract = NULL,   # can save other intermediates from the nested cv & call collect_extracts() 
#     save_pred = TRUE  # The inner run's predictions and extracts are still discarded
#     # Use extract = function(x) x to keep each fold's fitted workflow
#   )
# )

# nnet_tailor =                 #TODO we can use tailors to get to the WDL conversion
#   tailor() |>
#   adjust_predictions_custom(
# )


# https://yardstick.tidymodels.org/articles/grouping.html
# accuracy_diff <-
#   new_groupwise_metric(
#     fn = accuracy,
#     name = "accuracy_diff",
#     aggregate = function(acc_by_group) {
#       diff(c(acc_by_group$.estimate[2], acc_by_group$.estimate[1]))
#     }
#   )



# ex_outer = group_vfold_cv(data = euro, 
#                v = NULL, 
#                group = Year,
#                repeats = 1,
#                strata = NULL)
# 
# 
# results <- tune::tune_grid(
#   pois_wf,
#   resamples = ex_outer,
#   grid = pois_grid,
#   metrics = metrics
# )

# nested_fit_resamples

# TODO find way to implement the skellam WDL comparison as well https://www.tidymodels.org/learn/develop/metrics/
# TODO: Implement a similar function for nn, rf, xgboost
# TODO: rework so that it combines various model_classes as well
# TODO: Visualize tuning performances (thinking means with error bars, faceted by metric & outer)
# TODO have to find a resolution between rmse & mae for each fold (Currently only based on RMSE
# TODO: reconsider alternative to just taking first best parameter using selection_rule()
# TODO: e.g. but later call programmatically using lists for each model_class_wfs, model_class_grids, model_class_metrics
# aka forced pois win
# TODO: build winner workflow using winner_best_hyper_params & class_winner_wf but somehow 
# updating the workflow to use specified params
