# README: Silas doing nested CV (see docs/getting-on-the-same-page-about-splits.txt) on UEFA
# football fitting penalized poisson regressions

# Section 0: lots of sources which I probably glanced at for 3 seconds----

# https://www.r-bloggers.com/2018/11/a-tutorial-on-tidy-cross-validation-with-r/
# https://sjspielman.github.io/bio5312_fall2017/files/kfold_supplement.pdf
# https://www.tidymodels.org/
# https://www.tidymodels.org/learn/models/calibration/
# https://www.tidymodels.org/start/recipes/
# https://www.tidymodels.org/start/models/
# https://www.tidymodels.org/start/resampling/
# https://tune.tidymodels.org/articles/tune.html
# https://www.tidymodels.org/learn/models/time-series/
# https://www.tidymodels.org/start/tuning/
# https://cran.r-project.org/web/packages/caret/vignettes/caret.html
# https://en.wikipedia.org/wiki/Elastic_net_regularization
# https://rsample.tidymodels.org/articles/Working_with_rsets.html
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://stackoverflow.com/questions/43240525/should-i-split-my-data-into-training-testing-validation-sets-with-k-fold-cross-v
# https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k*l-fold_cross-validation
# https://academic.oup.com/jrsssb/article/39/1/44/7027527?login=true
# https://users.cs.duke.edu/~cynthia/CourseNotes/CrossValidationNotes.pdf
# https://github.com/myles-lewis/nestedcv

# Other modeling considerations: 
  # variable selection
  # transforming towards poisson regression? 
  # link function options? 
  # multicolinearity thing
  # including interactions & poly terms


  # "The potential problem is that once we pick the tuning parameter associated with
  # the best performance, this performance value is usually quoted as the performance of the model.
  # There is serious potential for optimization bias since we use the same data to tune the model and 
  # to assess performance. This would result in an optimistic estimate of performance."
  
  # This passage above doesn't apply to our nested situation but it gives us an analogy: 
  # We still have to do model & hypertuning performance combined based on the inner folds
  # The outer folds are for test/training



# Section 1: load libraries---------
# library(broom) 
# library(caret)
library(tune)
library(readr)
library(rsample)
# library(parsnip)
# library(glmnet)
library(poissonreg)
library(yardstick)
library(purrr)
library(dials)
library(recipes)
library(workflows)
# library(furrr)


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
# TODO find a way to programatically label the folds or custom build the nest
set.seed(67676769)

outer_folds = group_vfold_cv(data = euro,  # done outside of nested_cv so we can reuse later on in the process
                             v = NULL,     # to find model_class_winner's best hyperparameter 
                             group = Year,
                             repeats = 1,
                             strata = NULL)

for (layer_1 in 1:6) {   # for seed 67676769: order = 2020, 2004, 2016, 2012, 2024, 2008
  print(paste("layer_1:", layer_1))
  thesplit = outer_folds$splits[[layer_1]]
  print(unique(analysis(thesplit)$Year))
}

nested_cv(data = euro,
          outside = outer_folds,
          inside = group_vfold_cv(v = NULL, 
                                  group = Year,
                                  # group = match_id,
                                  repeats = 1,
                                  strata = NULL)) -> nest_folds

# inner layer training year appearance
outer_names = seq(2004,2024,4)# missing 
for (i in 1:6) {
  print("")
  for (layer_2 in 1:5) {
    print(paste("layer_1:", i, "layer_2:", layer_2))
    thesplit = nest_folds$inner_resamples[[i]]$splits[[layer_2]]
    print(unique(analysis(thesplit)$Year))
  }
}

# for seed 67676769: 
        # Test = 2020 (Outer 1)
            # eval = 2024
            # eval = 2008
            # eval = 2012
            # eval = 2016
            # eval = 2004
        # Test = 2004  (Outer 2)
            # eval = 2020
            # eval = 2008
            # eval = 2012
            # eval = 2024
            # eval = 2016
        # Test = 2016  (Outer 3)
            # eval = 2024
            # eval = 2012
            # eval = 20
            # eval = 20
            # eval = 20
        # Test = 2012  (Outer 4)
            # eval = 20
            # eval = 20
            # eval = 20
            # eval = 20
            # eval = 20
        # Test = 2024  (Outer 5)
            # eval = 20
            # eval = 20
            # eval = 20
            # eval = 20
            # eval = 20
        # Test = 2008  (Outer 6)
            # eval = 20
            # eval = 20
            # eval = 20
            # eval = 20
            # eval = 20

# Section 4: Fitting Poisson Elastic Net----

# specify model using parsnip 
#    Parsnip hides implementation details & standardize modeling approach between model classes
#    Observe how tune() is passed in order to delay specifying the parameters
pois_net_tune <- poisson_reg(engine = "glmnet", 
                        penalty = tune(), 
                        mixture = tune())

# Dials package generates a grid of hyperparameters over which to compare
pois_grid = grid_regular(penalty(), 
                         mixture(), 
                         levels = 7)


# workflow: allows integration of preprocessing, modeling, and post-processing
#    TODO add recipe later to attempt transforming to a better fitting poisson
#    Note: must be unfitted for use with tune()
pois_wf = workflow() |>
  add_model(pois_net_tune) |> 
  add_formula(Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers)
  
# Define evaluation metrics to be used
pois_metrics = yardstick::metric_set(
  rmse,
  mae
)

# tune_grid: uses our rset (k-folds) object and the specified workflow and hyperparameter grid to fit & observe 
# performance metrics for each model
  # Note about warning: Since you are fitting a lasso model, there are some penalty values that are high enough that, for some folds, all of the predictors are removed.
  # https://github.com/tidymodels/parsnip/issues/282
  # TODO find way to implement the skellam WDL comparison as well https://www.tidymodels.org/learn/develop/metrics/

pois_tune_on_inner <- function(wf, inner_num) {
  wf |> 
    tune_grid(
      resamples = nest_folds$inner_resamples[[inner_num]],
      grid = pois_grid,
      metrics = pois_metrics
    )
}

# TODO: Implement a similar function for nn, rf, xgboost

# hold inner tuning competition on each of the outer folds
# TODO look into using future_map to reduce fit time
# plan(multisession, workers = 3)
# future_map(1:6, function(x){pois_tune_on_inner(pois_wf, x)}) -> all_splits_hyperparms_metrics
# TODO: rework so that it combines various model_classes as well
map(1:6, function(x){pois_tune_on_inner(pois_wf, x)}) -> all_splits_hyperparms_metrics


# Section 5: Getting best hyperparameters across inner folds for each outer fold----

summarize_hyperparams_competition <- function(outer_fold_result) {
  # TODO: rewrite doc string & stuff to make this work for both this step and for the winner competition
  # outer_fold_results: should be the a single output of `tune_grid()` (E.g. all_splits_hyperparms_metrics[[1]])
    # resamples with extra list columns for .metrics and .notes 
    #(optional columns are .predictions and .extracts). .notes contains warnings
    # and errors that occur during execution.
  
  metrics_for_each_inner = outer_fold_result$.metrics
  metrics_across_inner = Reduce(rbind, metrics_for_each_inner) # join the metric results from the 5 inner folds
  metrics_across_inner |> 
    group_by(penalty, mixture, .metric) %>% #TODO: generalize for multi model_class!!!
    summarize(avg_metric_est = mean(.estimate), 
              var_metric_est = var(.estimate),
              n_obs_avg = n(),
              .groups = "drop") -> tuning_competition_result
  return(tuning_competition_result)
}

# For each outer fold, average the results of the inner folds to identify best performing hyperparam
map(1:6, function(x) {summarize_hyperparams_competition(all_splits_hyperparms_metrics[[x]])}) -> tuning_performances

# TODO: Visualize tuning performances (thinking means with error bars, faceted by metric & outer)


find_best_hyperparam <- function(tuning_performance_i) {
  tuning_performance_i %>%
    filter(.metric == "rmse") %>%   # TODO have to find a resolution between rmse & mae for each fold (Currently only based on RMSE
    group_by(.metric) %>% 
    slice_min(avg_metric_est,  # slice_min: selects rows with smallest value
              with_ties = FALSE # TODO: reconsider alternative to just taking first best parameter
              ) %>%
    ungroup()
}

# finding the best hyperparameter for each of the outer folds
map(1:6, function(x) {find_best_hyperparam(tuning_performances[[x]])}) -> best_hyper_params

# One summary table with a row for each outer fold
Reduce(rbind, best_hyper_params) -> best_hyper_params
best_hyper_params$outer_fold = 1:6
best_hyper_params$model_class = "poisson net" # TODO: fix later, temporary as example

# Section 6: summarizing tuned-model_class performance across outer folds----
best_hyper_params %>%
  group_by(model_class) %>%
  summarize(class_avg_metric_est = mean(avg_metric_est)) -> class_performances

best_class <- class_performances |> slice_min(class_avg_metric_est)


# Section 7: finding model_class_winner's best hyperparameter using outer folds----

class_winner_tune_on_outer <- function(class_winner_wf, 
                                       class_winner_grid,
                                       class_winner_metrics) {
  class_winner_wf |> 
    tune_grid(
      resamples = outer_folds,
      grid = class_winner_grid,
      metrics = class_winner_metrics
    )
}

# TODO: e.g. but later call programmatically using lists for each model_class_wfs, model_class_grids, model_class_metrics
      # aka forced pois win
class_winner_tune_on_outer(pois_wf, 
                           pois_grid, 
                           pois_metrics) -> winner_hyperparam_competition

# find metric averages for each param-combo across the folds for winner_class
summarize_hyperparams_competition(winner_hyperparam_competition) -> winner_tuning_performances

# simplify output to the best hyperparameter for the winner
find_best_hyperparam(winner_tuning_performances) -> winner_best_hyper_params


# Section 8: train winner using best hyperparameter on full data-----

# TODO: build winner workflow using winner_best_hyper_params & class_winner_wf but somehow 
# updating the workflow to use specified params
