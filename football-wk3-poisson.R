library(broom) # lm summaries & easier comparison
library(caret) # cv useful function
library(tune)
library(rsample)
library(parsnip)
library(glmnet)
library(poissonreg)
library(yardstick)
library(purrr)
library(dials)
library(recipes)
library(workflows)

# https://www.r-bloggers.com/2018/11/a-tutorial-on-tidy-cross-validation-with-r/
  # broom::augment
# https://sjspielman.github.io/bio5312_fall2017/files/kfold_supplement.pdf
  #crossv_kfold


#https://www.tidymodels.org/
# https://www.tidymodels.org/learn/models/calibration/
# https://www.tidymodels.org/start/models/
# https://www.tidymodels.org/start/resampling/
# https://www.tidymodels.org/start/tuning/
# https://cran.r-project.org/web/packages/caret/vignettes/caret.html
# https://en.wikipedia.org/wiki/Elastic_net_regularization
# https://rsample.tidymodels.org/articles/Working_with_rsets.html
# https://www.tidymodels.org/learn/models/time-series/
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://stackoverflow.com/questions/43240525/should-i-split-my-data-into-training-testing-validation-sets-with-k-fold-cross-v
 
# variable selection
# transforming towards poisson regression? 
# link function options? 
# multicolinearity thing
# including interactions & poly terms



# idea: maybe I can directly create an rset object? 
# better idea rsample::make_split() and make_rset
# Also group resampling? 
       #if we split a pair of matches between test and train for a given fold..
# Also rsample has time sensitive splitting

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

split_2004 <- make_splits(
  x = rbind(euro_2008,
            euro_2012,
            euro_2016,
            euro_2020,
            euro_2024),
  assessment = euro_2004
)

split_2008 <- make_splits(
  x = rbind(euro_2004,
            euro_2012,
            euro_2016,
            euro_2020,
            euro_2024),
  assessment = euro_2008
)

split_2012 <- make_splits(
  x = rbind(euro_2004,
            euro_2008,
            euro_2016,
            euro_2020,
            euro_2024),
  assessment = euro_2012
)

split_2016 <- make_splits(
  x = rbind(euro_2004,
            euro_2008,
            euro_2012,
            euro_2020,
            euro_2024),
  assessment = euro_2016
)

split_2020 <- make_splits(
  x = rbind(euro_2004,
            euro_2008,
            euro_2012,
            euro_2016,
            euro_2024),
  assessment = euro_2020
)

split_2024 <- make_splits(
  x = rbind(euro_2004,
            euro_2008,
            euro_2012,
            euro_2016,
            euro_2020),
  assessment = euro_2024
)

manual_rset(list(split_2004,
                 split_2008,
                 split_2012,
                 split_2016,
                 split_2020,
                 split_2024), 
            c("split_2004",
              "split_2008",
              "split_2012",
              "split_2016",
              "split_2020",
              "split_2024")) -> tourn_k_fold

# 
nested_cv(data = euro,
          outside = tourn_k_fold,
          inside = group_vfold_cv(v = 5, 
                                  group = Year,
                                  # group = match_id,
                                  repeats = 1,
                                  strata = NULL)) -> nest_folds

# Seeing whether I can shorten code just by using group_vfold_cv twice...
nested_cv(data = euro,
          outside = group_vfold_cv(v = 6, 
                                   group = Year,
                                   repeats = 1,
                                   strata = NULL),
          inside = group_vfold_cv(v = 5, 
                                  group = Year,
                                  # group = match_id,
                                  repeats = 1,
                                  strata = NULL)) -> nest_folds2

# nest_folds2 (quicker)
for (i in 1:6) {
  print("")
  for (layer_2 in 1:5) {
    print(paste("layer_1:", i, "layer_2:", layer_2))
    thesplit = nest_folds2$inner_resamples[[i]]$splits[[layer_2]]
    print(unique(analysis(thesplit)$Year))
  }
}

# THIS IS WHAT WE'RE DOING? https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k*l-fold_cross-validation


map(nest_folds$splits, function(x){unique(analysis(x)$Year)}) # outer layer

# inner layer training year appearance
outer_names = seq(2004,2024,4)# missing 
for (i in 1:6) {
  print("")
  for (layer_2 in 1:5) {
    print(paste("layer_1:", outer_names[i], "layer_2:", layer_2))
    thesplit = nest_folds$inner_resamples[[i]]$splits[[layer_2]]
    print(unique(analysis(thesplit)$Year))
  }
}

# MAKING LIFE SIMPLER ONLY 1 TUNING PARAM
pois_net_tune <- poisson_reg(engine = "glmnet", 
                        penalty = tune(), 
                        mixture = 1)


  # fit(Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers,
  #     data = football_train)
  


pois_grid = grid_regular(penalty(), 
                         # mixture(), 
                         levels = 5)
pois_wf = workflow() |>
  add_model(pois_net_tune) |> 
  add_formula(Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers)

# https://www.tidymodels.org/learn/work/nested-resampling/
# Code below doesn't run because nested not supported in tune package. See link above
# pois_res <- pois_wf |>
#   tune_grid(
#     resamples = nest_folds, 
#     grid = pois_grid) -> check
#   )



# `object` will be an `rsplit` object from our `results` tibble
# `cost` is the tuning parameter
pois_rmse <- function(object, penalty = 1) {
  set.seed(1234)
  y_col <- ncol(object$data)
  mod <- 
    poisson_reg(penalty = penalty, 
                mixture = 1) |> 
    set_engine("glmnet") |>
    fit(data = analysis(object),
        Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers)
  
  
  holdout_pred <- 
    predict(mod, assessment(object) |> dplyr::select(-Goals)) |> 
    bind_cols(assessment(object) |> dplyr::select(Goals))
  rmse(holdout_pred, truth = Goals, estimate = .pred)$.estimate
}

# In some case, we want to parameterize the function over the tuning parameter:
rmse_wrapper <- function(penalty, object) pois_rmse(object, penalty)

# `object` will be an `rsplit` object for the inner folds
tune_over_penalty <- function(object) {
  pois_grid |> 
    mutate(RMSE = map_dbl(penalty, rmse_wrapper, object = object))
}


# `object` is an `rsplit` object in `nest_folds$inner_resamples` 
summarize_tune_results <- function(object) {
  # Return row-bound tibble that has the 5 inner fold results
  map_df(object$splits, tune_over_penalty) |>
    # For each value of the tuning parameter, compute the 
    # average RMSE which is the inner bootstrap estimate. 
    group_by(penalty) |>
    summarize(mean_RMSE = mean(RMSE, na.rm = TRUE),
              n = length(RMSE),
              .groups = "drop")
}

tuning_results <- map(nest_folds$inner_resamples, summarize_tune_results) 

library(scales)
library(ggplot2)

pooled_inner <- tuning_results |> bind_rows()

best_penalty <- function(dat) dat[which.min(dat$mean_RMSE),]

p <- 
  ggplot(pooled_inner, aes(x = penalty, y = mean_RMSE)) + 
  scale_x_continuous(trans = 'log2') +
  xlab("Pois penalty") + ylab("Inner RMSE")

for (i in 1:length(tuning_results))
  p <- p  +
  geom_line(data = tuning_results[[i]], alpha = .2) +
  geom_point(data = best_penalty(tuning_results[[i]]), pch = 16, alpha = 3/4)

p <- p + geom_smooth(data = pooled_inner, se = FALSE)
p

penalty_vals <- 
  tuning_results |> 
  map_df(best_penalty) |> 
  select(penalty)

nest_folds <- 
  bind_cols(nest_folds, penalty_vals) |> 
  mutate(penalty = factor(penalty, levels = pois_grid$penalty))

ggplot(nest_folds, aes(x = penalty)) + 
  geom_bar() + 
  xlab("Pois penalty") + 
  scale_x_discrete(drop = FALSE)

# "The potential problem is that once we pick the tuning parameter associated with
# the best performance, this performance value is usually quoted as the performance of the model.
# There is serious potential for optimization bias since we use the same data to tune the model and 
# to assess performance. This would result in an optimistic estimate of performance."

# This passage above doesn't apply to our nested situation but it gives us an analogy: 
    # We still have to do model & hypertuning performance combined based on the inner folds
    # The outer folds are for test/training







# !!!!!!!!!!!!!! https://academic.oup.com/jrsssb/article/39/1/44/7027527?login=true

# https://github.com/myles-lewis/nestedcv



# !!!!!!!!!! https://users.cs.duke.edu/~cynthia/CourseNotes/CrossValidationNotes.pdf