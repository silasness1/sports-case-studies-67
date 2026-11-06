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
#   - model_class_fits


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
library(mirai)

# Section 2: Read in data ----
euro_2004 = read_csv("raw-data/euro_2004.csv")
euro_2008 = read_csv("raw-data/euro_2008.csv")
euro_2012 = read_csv("raw-data/euro_2012.csv")
euro_2016 = read_csv("raw-data/euro_2016.csv")
euro_2020 = read_csv("raw-data/euro_2020.csv")
euro_2024 = read_csv("raw-data/euro_2024.csv")

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

# Rename outer resamples
get_outer_names <- function(resamples, all_years) {
  purrr::map_chr(resamples$splits, function(split) {
    test_year <- setdiff(all_years, unique(analysis(split)$Year))
    
    stopifnot(length(test_year) == 1)
    
    paste("test", test_year)
  })
}

nested_leave_one_year <- rename_rset_ids(
  nested_leave_one_year,
  get_outer_names(nested_leave_one_year, all_years)
)

# Rename inner resamples
for (i in seq_along(nested_leave_one_year$inner_resamples)) {
  
  outer_test_year <- setdiff(
    all_years,
    unique(analysis(nested_leave_one_year$splits[[i]])$Year)
  )
  
  inner <- nested_leave_one_year$inner_resamples[[i]]
  
  inner_names <- get_split_names(
    inner,
    all_years,
    outer_test_year
  )
  
  nested_leave_one_year$inner_resamples[[i]] <- rename_rset_ids(
    inner,
    inner_names
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

# Section 4: Poisson Workflow & Grid----
pois_tune <- poisson_reg(engine = "glmnet", 
                             penalty = tune(), 
                             mixture = tune())

pois_grid = grid_regular(penalty(), 
                         mixture(), 
                         levels = 7)

pois_wf = workflow() |>
  add_model(pois_tune) |> 
  add_formula(Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers)

# Section 5: Defining Neural Net Workflow & Grid -----


nnet_tune = mlp(engine = "nnet", 
                mode = "regression",
                penalty = tune(), 
                hidden_units = tune()
)

nnet_grid = expand.grid(
  penalty = c(0.7, 0.5, 1,0.9,1.1,2), # TODO think about whether to include much smaller values!
  hidden_units = 1:5
)

# nnet_tailor =                 #TODO we can use tailors to get to the WDL conversion
#   tailor() |>
#   adjust_predictions_custom(
# )
  
nnet_wf = workflow() |>
  add_model(nnet_tune) |> 
  add_formula(Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers)

# Section 6: Defining Random Forest Workflow & Grid -----

target <- "Goals"
predictors <- c(
  "GroupStage",
  "GDP",
  "MarketValue",
  "FifaRank",
  "UefaPoints",
  "CLPlayers"
)
N_TREES = 100

rf_formula <- as.formula(
  paste(target, "~", paste(predictors, collapse = " + "))
)

rf_tune = rand_forest(
  min_n = tune(), 
  mtry = tune(), 
  # trees = 1000
  trees= !!N_TREES
) %>%
  set_engine(
    "ranger", 
    sample.fraction = tune(), 
    num.threads = 1) %>%
  set_mode("regression")

rf_grid <- expand.grid(
  # mtry = 1:length(predictors),
  mtry = 3:length(predictors),
  # min_n = c(3, 5, 10, 15), #simplifying
  min_n = c(3, 5), 
  sample.fraction = c(0.7, 0.85)
  # sample.fraction = c(0.7, 0.85, 1.0)
)

rf_wf = workflow() |>
  add_model(rf_tune) |> 
  add_formula(rf_formula)

# Section 7: Defining XGBoost Workflow & Grid-----

# min_child_weight <- function(range = c(1, 10)) {
#   dials::new_quant_param(
#     type = "double",
#     range = range,
#     inclusive = c(TRUE, TRUE),
#     label = c(min_child_weight = "Minimum Child Weight")
#   )
# }
# xg_params <- dials::parameters(
#   dials::tree_depth(),
#   dials::learn_rate(),
#   dials::sample_size(),
#   min_child_weight()
# )

xg_tune <- boost_tree(
  tree_depth = tune(), 
  learn_rate = tune(), 
  sample_size = tune(), 
  min_n = tune() # not really the same as min_child_weight but... close enough for now
  ) |> 
  set_engine("xgboost",
             counts = FALSE, # changes interpretation of colsample_bytree
             # min_child_weight = tune(), # not possible..
             eval_metric = "poisson-nloglik", 
             objective =  "count:poisson",
             colsample_bytree = 0.8) |> 
  set_mode("regression")

xg_grid <- expand.grid(
  # tree_depth       = c(1, 2, 3),
  # learn_rate       = c(0.01, 0.05, 0.1), # assuming equivalent to eta
  # sample_size      = c(0.5, 0.8), #assuming this was subsample
  # min_child_weight = c(1, 5, 10)
  #SIMPLE
  tree_depth       = c(1, 2),
  learn_rate       = c(0.01, 0.05), # assuming equivalent to eta
  sample_size      = c(0.5, 0.8), #assuming this was subsample
  min_n            = c(1,5,10)
)

xg_wf = workflow() |>
  add_model(xg_tune) |> 
  add_formula(Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers)

# Section 8: Setting up Model_class iterators----

# Define evaluation metrics to be used across all models for goals (later define for WDL)
metrics_goals = yardstick::metric_set(
  rmse,
  mae,
  poisson_log_loss
)

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

# model_class_param_infos = list(nnet = NULL, 
#                                pois = NULL,
#                                rf   = NULL,
#                                xg   = xg_params)

# Section 9: Running nested_tune_grid for each model_class----

model_class_vec = c(
  "rf", 
  "nnet",
  "xg", 
  "pois"
)

run_nested_tune_for_model = function(model_class) {
  mirai::daemons(2) # 0 vs 2 makes a difference 3,4, no real difference.
  
  print(system.time({
  tuned = nested_tune_grid(
    object     = model_class_wfs[[model_class]], 
    resamples  = model_class_resamples[[model_class]], 
    grid       = model_class_grids[[model_class]], 
    metrics    = model_class_metrics[[model_class]],
    # param_info = model_class_param_infos[[model_class]]
  )}))
  
  mirai::daemons(0)
  
  return(tuned)
}

# Switch to vectorized after debugging a bit
# model_class_fits = map(model_class_vec, run_nested_tune_for_model)

pois_tuned = run_nested_tune_for_model("pois") #full run took 23 seconds
rf_tuned = run_nested_tune_for_model("rf") #simplified run took 40 seconds
xg_tuned = run_nested_tune_for_model("xg") #simplfied run took 63 seconds
nnet_tuned = run_nested_tune_for_model("nnet") # full run took 46 seconds




summary(poisson_competition)

collect_metrics(poisson_competition)
collect_selections(poisson_competition)

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
