
# README: 
# This script is part of Case Studies at TU Dortmund -- week 2 of the sports project
# We're provided 3 sports datasets with different response types {ordinal, binary, metric}
# Now we'll be fitting Classification & Regression Trees (CART)
# And learning about overfitting, Regression Trees, and Tuning


# Section 0: Load Libraries-----
library(readr)
library(tidyr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(purrr)

# Section 1: Read in the csv----
football_test = read_csv("raw-data/EuroTest.csv")
football_train = read_csv("raw-data/EuroTraining.csv")
handball_long = read_csv("raw-data/handball_long.csv")
handball_wide = read_csv("raw-data/handball_wide.csv")
tennis = read_csv("raw-data/SlamsData.csv")

# quickly cleaning data if necessary
tennis$win <- factor(x = tennis$win, levels = c(0,1), labels = c("loss", "win"), ordered = T)
str(tennis)
clean_football_types <- function(df) {
  df  %>%
    mutate(across(c(Team, Opponent, GroupStage), as.factor)) %>%
    mutate(across(c(Goals, FifaRank, UefaPoints), as.integer))
}

football_test <- clean_football_types(football_test)
football_train <- clean_football_types(football_train)

# Some online resources for getting started with `rpart`
# (didn't use) https://bookdown.org/tpinto_home/Beyond-Additivity/regression-and-classification-trees.html

# General rpart notes from help section----

# rpart.object$frame: surrogate best reproduces a split; competititor is next best
# rpart.object$split: appears to have more than final splits
#                     index = splitpoint 
#                     Pay attention to row labels & count variable: equal counts for competitors, 0 for surrogat
# rpart.object$cptable: cp = R(T) + alpha*|T| where |T| is number of leaves for tree T
#                       records how the training (relative to root single avg) and internal CV (xerror) change with number of splits
# rpart.object$control: the hyperparameters used
#           rpart.control: minsplit: # obs in a node for a split to be attempted
#                          minbucket: # of obs needed to create a leaf node
#                          cp: minimum improvement in risk minimzation by making children
#                          maxcompete/maxsurrogate: how many options to record in output
#                          xval: # of CVs
#                          maxdepth: like it sounds for any leaf node
# rpart(method = "anova"): for nummerical outcomes -> handball, football (unless we convert to Win/Loss/Draw first)
# rpart(method = "class"): for categorical outcomes -> probably tennis
# rpart(method = "poisson"): would need associated "exposure" like time between goals

# Section 2: Fit a regression tree to handball data----
# Use formula scorediff ~ bet.probs.diff + season
handball_mod1 = rpart(data = handball_wide, scorediff ~ bet.probs.diff + season)
rpart.plot(handball_mod1)
summary(handball_mod1)

# handball evaluating predictions
handball_mod1_pred = predict(handball_mod1)
handball_mod1_mae = mean(abs(handball_mod1_pred - handball_wide$scorediff))
football_mod1_rmse = sqrt(mean((handball_mod1_pred - handball_wide$scorediff)^2))

# identifying poorly predicted outcomes handball
handball_wide$handball_mod1_pred = handball_mod1_pred
handball_wide$residual = handball_wide$handball_mod1_pred - handball_wide$scorediff

# Section 3: Fit Regresion tree to football_train----
# Use Formula Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers
football_mod1 = rpart(data = football_train, 
                      Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers)
rpart.plot(football_mod1)

# football evaluating predictions
football_mod1_pred = predict(football_mod1)
football_mod1_mae = mean(abs(football_mod1_pred - football_train$Goals))
football_mod1_rmse = sqrt(mean((football_mod1_pred - football_train$Goals)^2))

# identifying poorly predicted outcomes football
football_train$football_mod1_pred = football_mod1_pred
football_train$residual = football_train$football_mod1_pred - football_train$Goals

# Section 4: Fit Classification Tree to tennis data----
# Use formula win ~ rank + points + elo + age.30
#         Observation: only splits based on elo...
tennis_mod1 = rpart(data = tennis, 
                    method = "class",
                    win ~ rank + points + elo + age.30)
rpart.plot(tennis_mod1)

# tennis evaluating predictions tennis
tennis_mod1_pred = predict(tennis_mod1)
tennis_mod1_pred_ratio = sum(((tennis_mod1_pred[,"win"] >= .5) & (tennis$win == "win")) | 
                             ((tennis_mod1_pred[,"win"] < .5)  & (tennis$win == "loss")) )/nrow(tennis_mod1_pred)

# identifying poorly predicted outcomes football
tennis$tennis_mod1_pred_win = tennis_mod1_pred[,"win"]
tennis$tennis_mod1_pred_loss = tennis_mod1_pred[,"loss"]
tennis$pred_correct = ((tennis$tennis_mod1_pred_win >= .5) & (tennis$win == "win")) | 
  ((tennis$tennis_mod1_pred_win < .5)  & (tennis$win == "loss")) 


# Section 5: handball Scatterplots with fitted lines----
ggplot(data = handball_wide)+
  geom_point(aes(x = bet.probs.diff, y = scorediff), color = "blue")+
  geom_vline(xintercept =.0095)+
  geom_vline(xintercept = -.46)+
  geom_vline(xintercept =.55 )+
  geom_vline(xintercept = .78)+
  geom_segment(y = -4.6, x = -1, xend = -.46)+
  geom_segment(y = -.94, x = -.46, xend = .0095)+
  geom_segment(y = 1.5, x = .0095, xend = .55)+
  geom_segment(y = 4.7, x = .55, xend = .78)+
  geom_segment(y = 7.4, x = .78, xend = 1)+
  labs(title = "Handball Mod1 Regression Tree")+
  theme_bw()

# Section 6: football scatter plots----
# I'm too lazy to do the final 3 plots for football: 
#        (GDP < 1.1 ) & (FifaRank >= -12) and (MarketValue >=1.1) 

ggplot(data = football_train)+
  geom_point(aes(x = MarketValue, y = Goals, color = residual))+
  scale_color_continuous(palette = c("red", "white", "blue"))+
  geom_vline(xintercept =-.26, color = "green")+
  geom_vline(xintercept =1.6)+
  geom_segment(y = 1.1, x = -3, xend = 1.6)+
  geom_segment(y = 2.3, x = 1.6, xend = 3)+
  geom_segment(y = .8, x = -3, xend = -.26, color = "green")+
  geom_segment(y = 1.3, x = -.26, xend = 1.6, color = "green")+
  labs(title = "Football Mod1 Regression Tree: Market Value")+
  theme_bw()

ggplot(data = football_train)+
  geom_point(aes(x = MarketValue, y = Goals, color = residual))+
  scale_color_continuous(palette = c("red", "white", "blue"))+
  geom_vline(xintercept =1.6)+
  geom_segment(y = 1.1, x = -3, xend = 1.6)+
  geom_segment(y = 2.3, x = 1.6, xend = 3)+
  labs(title = "Football Mod1 Regression Tree: Market Value")+
  theme_bw()

football_train %>%
  filter(MarketValue >= 1.6) %>%
ggplot()+
  geom_point(aes(x = .data[["FifaRank"]], y = Goals, color = residual))+
  scale_color_continuous(palette = c("red", "white", "blue"))+
  geom_vline(xintercept =-20)+
  geom_segment(y = 1.9, x = -100, xend = -20)+
  geom_segment(y = 3.3, x = -20, xend = 100)+
  labs(title = "Football Mod1 Regression Tree: FifaRank", 
       caption = "Data only from parent node MarketValue >=1.6")+
  theme_bw()


# Section 7: tennis boxplots (chose histograms instead) ----
tennis %>%
  ggplot()+ 
  geom_histogram(aes(x = tennis_mod1_pred_win))+
  labs(title = "Not Much of a Distribution of Win Probabilites for Tennis Mod1", 
       subtitle = "cp was set too high so tree growth prevented")

tennis %>%
  ggplot()+ 
  geom_histogram(aes(x = tennis_mod1_pred_loss))+
  labs(title = "Not Much of a Distribution of Losing Probabilites for Tennis Mod1", 
       subtitle = "cp was set too high so tree growth prevented")

# Section 8: Reexamine the controls for Tennis regression----
#         - My fit only ever had 2 predicted probabilities (1 split based on elo) 
#         - Think about what error measure is used for classification (if anova -> R^2)
#         - (Need help!) Understanding what CP actually represents in classification...
#                    - In dcoumentation found R_cp(T) = R(T) + cp*|T|*R(T_1) where T is tree T_1 is tree
#                       without splits, R is [empirical] risk  and R_cp is what the splitting criteria is based off of
#                    - Somehow brier/gini probably has something to do with explanation (and R^2)
tennis_mod1$control$cp # the setting was .01
tennis_mod1$cptable # the first split had a CP of .01 so no more were permitted

# Fitting a model with lower cp for tennis
#        Wow thats a lot of splits!
tennis_mod2 = rpart(data = tennis, 
                    method = "class",
                    win ~ rank + points + elo + age.30, 
                    control = list(cp = .001))
rpart.plot(tennis_mod2)
summary(tennis_mod2)

# way more complex, but Veeeeeery little gain (Mod2: 76.5% vs Mod1: 75%)
tennis_mod2_pred = predict(tennis_mod2)
tennis_mod2_pred_ratio = sum(((tennis_mod2_pred[,"win"] >= .5) & (tennis$win == "win")) | 
                               ((tennis_mod2_pred[,"win"] < .5)  & (tennis$win == "loss")) )/nrow(tennis_mod2_pred)


# Section 9: Importance of Predictors----
#     https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#           "Impurity-based feature importances can be misleading for high cardinality features (many unique values)."
#     https://en.wikipedia.org/wiki/Feature_Importance
# Impurity based feature importances: how much feature improves tree splits (reduction in variance of nodes)
# Permutation based feature importance: something about disrupting information & seeing loss in prediction
#      Lets hope impurity based feature importance makes sense (maybe not for handball but at least for football & tennis)

tennis_mod1$variable.importance
football_mod1$variable.importance
handball_mod1$variable.importance # misleadingly large (so skip for now!)


# Section 10: football mod1 train vs test errors----
#     Conclusion: training mae - .778 test mae - .7836
#                 training rmse - .97 test rmse - 1.048
#     All we can really say is that we're not crazily overfitting... otherwise test error would be way higher than training error
#     Don't think we can distinguish between underfit & reasonable (though gut says reasonable considering lower bound of .5)
football_mod1_test_pred <- predict(football_mod1, newdata = football_test)
football_mod1_test_mae = mean(abs(football_test$Goals - football_mod1_test_pred))
football_mod1_test_rmse = sqrt(mean((football_test$Goals - football_mod1_test_pred)^2))

# Section 11: Building a pipeline for Hyper Parameter grid search, test & train----
#    Possibly using  map_dfr, map, pmap
#    Plan: 1) to fit a model given the specific parameters
#    https://www.r-bloggers.com/2024/10/how-to-iterate-over-rows-of-data-frame-in-r-a-complete-guide-for-beginners/
#    Note about pmap:  Arguments will be supply by position if unnamed, and by name if named.

# Create Test and Train datasets
tennis_train = tennis %>%
  filter(year <= 2017)
tennis_test = tennis %>%
  filter(year > 2017)
handball_train = handball_wide %>%
  filter(season != max(season))
handball_test = handball_wide %>%
  filter(season == max(season))


ctrl_params <- expand.grid(cp = c(0, .001, .005, .01, .02, .05), 
            minsplit = c(2, 5, 10, 20, 40), 
            maxdepth = c(1,2, 3, 5, 6, 10, 15))

# FOR TESTING ON SMALLER # OF MODELS
# ctrl_params_small <- expand.grid(cp = c(0, .001), 
#                             minsplit = c(2, 5), 
#                             maxdepth = c(1,2))

# Getting pmap to run multiple rpart models (working!)
# pmap(ctrl_params_small, function(...) {rpart(data = tennis_train, 
#                                             formula = win ~ rank + points + elo + age.30,
#                                             method = "class", 
#                                             ...)})-> check


# Helper Functions to Evaluate Error
calculate_mae <- function(actual, prediction){
  return (mean(abs(actual - prediction)))
}
calculate_rmse <- function(actual, prediction){
  return (sqrt(mean((actual - prediction)^2)))
}

tennis_ratio_wrongly_pred <- function(actual, pred_win, pred_loss) {
  # Example: 
  #       tennis_ratio_wrongly_pred(actual = tennis$win, pred_win = tennis_mod2_pred[,"win"],
  #                                 pred_loss = tennis_mod2_pred[,"loss"])
  # Note: assumes that pred_win & pred_loss are consistent with each other....!!!
  #     E.g. pred_win = .6 -> pred_loss = .7 not allowed
  actual_win <- actual == "win"
  actual_loss <- actual == "loss"
  pred_win_gt_half = pred_win >= .5
  pred_loss_gt_half = pred_loss >= .5
  
  incorrect_pred = (actual_win & pred_loss_gt_half) | (actual_loss & pred_win_gt_half)
  
  return(sum(incorrect_pred)/length(actual_win))
}


# Section 12: Tennis Grid Search Over Hyper Parameters----

tennis_call_pmap_on_me <- function(...) {
  
  # make the tree
  tree = rpart(data = tennis_train, 
               formula = win ~ rank + points + elo + age.30,
               method = "class", 
               maxsurrogate = 0, # wondering if this is consequential...(tried with 2--no diff)
               maxcompete = 0,
                     ...)
  
  #get training error
  train_pred = predict(tree, newdata = tennis_train)
  train_error = tennis_ratio_wrongly_pred(actual = tennis_train$win, 
                                          pred_win = train_pred[,"win"], 
                                          pred_loss = train_pred[,"loss"])
  # get test error
  test_pred = predict(tree, newdata = tennis_test)
  test_error = tennis_ratio_wrongly_pred(actual = tennis_test$win, 
                                         pred_win = test_pred[,"win"], 
                                         pred_loss = test_pred[,"loss"])
  
  # Include splits in the output df
  splits_df = data_frame(var = rownames(tree$splits),
                         count = tree$splits[,"count"],
                         ncat = tree$splits[,"ncat"],
                         index = tree$splits[,"index"])
  splits_df %>%
    mutate(summary_string = paste("var:", var,
                                  "count:", count,
                                  "ncat:", ncat,
                                  "index:", round(index,3))) -> splits_df
  splits_str = paste(splits_df$summary_string, collapse = "\n")
  
  # Create output df 
  df = data_frame(..., 
                  train_error=train_error, 
                  test_error=test_error,
                  nsplits=nrow(splits_df),
                  splits_str=splits_str)
  return(df)
}

# Actually running the grid search
pmap_dfr(ctrl_params,tennis_call_pmap_on_me)-> tennis_grid_search


tennis_mod_winner = rpart(data = tennis_train, 
             formula = win ~ rank + points + elo + age.30,
             method = "class", 
             maxsurrogate = 0, 
             maxcompete = 0,
             cp = 0, 
             minsplit = 2,
             maxdepth = 1)
rpart.plot(tennis_mod_winner)


# Section 13: Handball Grid Search Over Hyper Parameters-----
handball_call_pmap_on_me <- function(...) {
  
  # make the tree
  tree = rpart(data = handball_train, 
               formula = scorediff ~ bet.probs.diff + season,
               method = "anova", 
               maxsurrogate = 0, # wondering if this is consequential...(tried with 2--no diff)
               maxcompete = 0,
               ...)
  
  #get training error
  train_pred = predict(tree, newdata = handball_train)
  train_mae = calculate_mae(actual = handball_train$scorediff, 
                            prediction = train_pred)
  train_rmse = calculate_rmse(actual = handball_train$scorediff,
                              prediction = train_pred)
  
  # get test error
  test_pred = predict(tree, newdata = handball_test)
  test_mae = calculate_mae(actual = handball_test$scorediff, 
                            prediction = test_pred)
  test_rmse = calculate_rmse(actual = handball_test$scorediff,
                              prediction = test_pred)
  
  # Include splits in the output df
  splits_df = data_frame(var = rownames(tree$splits),
                         count = tree$splits[,"count"],
                         ncat = tree$splits[,"ncat"],
             index = tree$splits[,"index"])
  splits_df %>%
    mutate(summary_string = paste("var:", var,
                                  "count:", count,
                                  "ncat:", ncat,
                                  "index:", round(index,3))) -> splits_df
  splits_str = paste(splits_df$summary_string, collapse = "\n")
  
  # Create output df 
  df = data_frame(..., 
                  train_mae=train_mae, 
                  train_rmse=train_rmse,
                  test_mae=test_mae,
                  test_rmse=test_rmse,
                  nsplits=nrow(splits_df),
                  splits_str=splits_str)
  return(df)
}

# Actually running the grid search
pmap_dfr(ctrl_params,handball_call_pmap_on_me)-> handball_grid_search

# winning handball model (min test error)
handball_mod_win = rpart(data = handball_train, 
             formula = scorediff ~ bet.probs.diff + season,
             method = "anova", 
             maxsurrogate = 0, # wondering if this is consequential...(tried with 2--no diff)
             maxcompete = 0,
             cp = 0,
             minsplit = 40,
             maxdepth = 5)
rpart.plot(handball_mod_win)


# Section 14: Football Grid Search Over Hyper Parameters-----
football_call_pmap_on_me <- function(...) {
  
  # make the tree
  tree = rpart(data = football_train, 
               formula = Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers,
               method = "anova", 
               maxsurrogate = 0, # doesn't appear to make a difference.
               maxcompete = 0,
               ...)
  
  #get training error
  train_pred = predict(tree, newdata = football_train)
  train_mae = calculate_mae(actual = football_train$Goals, 
                            prediction = train_pred)
  train_rmse = calculate_rmse(actual = football_train$Goals,
                              prediction = train_pred)
  
  # get test error
  test_pred = predict(tree, newdata = football_test)
  test_mae = calculate_mae(actual = football_test$Goals, 
                           prediction = test_pred)
  test_rmse = calculate_rmse(actual = football_test$Goals,
                             prediction = test_pred)
  
  # Include splits in the output df
  splits_df = data_frame(var = rownames(tree$splits),
                         count = tree$splits[,"count"],
                         ncat = tree$splits[,"ncat"],
                         index = tree$splits[,"index"])
  splits_df %>%
    filter(count != 0)%>% # removes surrogates from nsplits and splits_str!
    mutate(summary_string = paste("var:", var,
                                  "count:", count,
                                  "ncat:", ncat,
                                  "index:", round(index,3))) -> splits_df
  splits_str = paste(splits_df$summary_string, collapse = "\n")
  
  # Create output df 
  df = data_frame(..., 
                  train_mae=train_mae, 
                  train_rmse=train_rmse,
                  test_mae=test_mae,
                  test_rmse=test_rmse,
                  nsplits=nrow(splits_df),
                  splits_str=splits_str)
  return(df)
}

# football actually running the grid search
pmap_dfr(ctrl_params,football_call_pmap_on_me)-> football_grid_search

# football model winner
football_mod_winner = rpart(data = football_train, 
                            formula = Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers,
                            method = "anova",
                            maxsurrogate = 0, # doesn't appear to make a difference.
                            maxcompete = 0,
                            cp = 0, 
                            minsplit = 5,
                            maxdepth = 3)
rpart.plot(football_mod_winner)


# Section 15: Visualizing Overfitting Tennis----

#       Worst Overfit (biggest diff between test & train error):
#              cp = 0, minsplit = 2, maxdepth = 15 resulted in 434 splits!!!
tennis_grid_search %>%
  mutate(diff_test_train = abs(train_error-test_error)) -> tennis_grid_search
tennis_mod_worst_overfit = rpart(data = tennis_train, 
                                        formula = win ~ rank + points + elo + age.30,
                                        method = "class", 
                                        maxsurrogate = 0,
                                        maxcompete = 0,
                                 cp = 0, 
                                 minsplit=2, 
                                 maxdepth=15)
# rpart.plot(tennis_mod_worst_overfit) # barely even runs...

# Plotting tennis overfitting as a function of increasing complexity
tennis_grid_search %>%
  filter(cp ==0 & minsplit ==2) %>%
  pivot_longer(
    c(train_error, test_error),
    names_to = "test_or_train",
    values_to = "percent_false_preds"
  ) %>%
ggplot()+
  geom_line(aes(y = percent_false_preds, x = maxdepth, color = test_or_train))+
  labs(title = "Tennis: Training vs Test Errors with Increasing Complexity", 
       y = "% of matches falsely predicted",
       color = "Train vs Test",
       caption = "Taking control parameters cp ==0 & minsplit ==2 to be fixed")+
  theme_bw()

# FACET WRAP VERSION!
tennis_grid_search %>%
  pivot_longer(
    c(train_error, test_error),
    names_to = "test_or_train",
    values_to = "percent_false_preds"
  ) %>%
  ggplot()+
  geom_line(aes(y = percent_false_preds, x = maxdepth, color = test_or_train))+
  facet_grid(rows = vars(cp), cols = vars(minsplit))+
  labs(title = "Tennis: Training vs Test Errors with Increasing Complexity", 
       y = "% of matches falsely predicted",
       color = "Train vs Test",
       caption = "Facet columns: minsplit \nFacet rows: cp")+
theme_bw()


# tennis selected model comparison
tennis_grid_search %>%
  slice(c(1, # best error
          181, #overfit
          116 # when we first start getting past a one split model
          ))  -> tennis_mod_compare
tennis_mod_compare$comment = c("least test error", "worst overfit", "reasonable performer using more than one split")
tennis_mod_compare %>% 
  arrange(test_error) %>%
  relocate(comment)-> tennis_mod_compare

# Section 16: Visualizing Overfitting Handball----

# worst overfit
handball_grid_search %>%
  mutate(diff_test_train_mae = abs(test_mae - train_mae), 
         diff_test_train_rmse = abs(test_rmse - train_rmse)) -> handball_grid_search

handball_grid_search %>%
  pivot_longer(
    cols = starts_with(c("train_", "test_")),
    names_to = "test_vs_train",
    values_to = "error_measure"
  ) %>%
  separate_wider_delim(test_vs_train, 
                       delim = "_",
                       names = c("test_or_train", "error_measure_type")) %>%
  filter(error_measure_type == "mae") %>%
  ggplot()+
  geom_line(aes(y = error_measure, x = maxdepth, color = test_or_train))+
  facet_grid(rows = vars(cp), cols = vars(minsplit))+
  labs(title = "Handball: Training vs Test Errors with Increasing Complexity", 
       y = "Mean Absolute Error",
       color = "Train vs Test",
       caption = "Facet columns: minsplit \nFacet rows: cp")+
  theme_bw()

# handball selected model comparison
handball_grid_search %>%
  slice(c(
          61,  # compromise
          115, #best test mae
          181 #overfit
          )) %>%
  arrange(test_mae)-> handball_mod_comparison
handball_mod_comparison$comment = c("best test mae", 
                                    "compromise btw number of splits & test mae",
                                    "worst overfit in mae & rmse")
handball_mod_comparison %>% relocate(comment) -> handball_mod_comparison

# Section 17: Visualizing Overfitting Football----

#finding worst
football_grid_search %>%
  mutate(diff_test_train_mae = abs(test_mae - train_mae), 
         diff_test_train_rmse = abs(test_rmse - train_rmse)) -> football_grid_search


football_grid_search %>%
  pivot_longer(
    cols = starts_with(c("train_", "test_")),
    names_to = "test_vs_train",
    values_to = "error_measure"
  ) %>%
  separate_wider_delim(test_vs_train, 
                       delim = "_",
                       names = c("test_or_train", "error_measure_type")) %>%
  filter(error_measure_type == "mae") %>%
  ggplot()+
  geom_line(aes(y = error_measure, x = maxdepth, color = test_or_train))+
  facet_grid(rows = vars(cp), cols = vars(minsplit))+
  labs(title = "Football: Training vs Test Errors with Increasing Complexity", 
       y = "Mean Absolute Error",
       color = "Train vs Test",
       caption = "Facet columns: minsplit \nFacet rows: cp")+
  theme_bw()

# handpicked model comparison
football_grid_search %>%
  slice(c(
    67, # lowest train mae
    1, # a single split model does quite well!!
    181 # worst overfit
  )) %>%
  arrange(test_mae)-> football_mod_comparison
football_mod_comparison$comment = c("best test mae", 
                                    "a single split model does well",
                                    "worst overfit in mae & rmse")
football_mod_comparison %>% relocate(comment) -> football_mod_comparison
