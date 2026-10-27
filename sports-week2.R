
# README: 
# This script is part of Case Studies at TU Dortmund -- week 2 of the sports project
# We're provided 3 sports datasets with different response types {ordinal, binary, metric}
# Now we'll be fitting Classification & Regression Trees (CART)
# And learning about overfitting, Regression Trees, and Tuning


# Load Libraries-----
library(readr)
library(dplyr)
library(rpart)
library(rpart.plot)


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

# handball evaluating predictions
handball_mod1_pred = predict(handball_mod1)
handball_mod1_mae = mean(abs(handball_mod1_pred - handball_wide$scorediff))
football_mod1_rmse = sqrt(mean((handball_mod1_pred - handball_wide$scorediff)^2))


# Section 3: Fit Regresion tree to football_train----
# Use Formula Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers
football_mod1 = rpart(data = football_train, 
                      Goals ~ GroupStage + GDP + MarketValue + FifaRank + UefaPoints + CLPlayers)
rpart.plot(football_mod1)

# football evaluating predictions
football_mod1_pred = predict(football_mod1)
football_mod1_mae = mean(abs(football_mod1_pred - football_train$Goals))
football_mod1_rmse = sqrt(mean((football_mod1_pred - football_train$Goals)^2))


# Section 4: Fit Classification Tree to tennis data----
# Use formula win ~ rank + points + elo + age.30
#         Observation: only splits based on elo...
tennis_mod1 = rpart(data = tennis, 
                    method = "class",
                    win ~ rank + points + elo + age.30)
rpart.plot(tennis_mod1)

# tennis evaluating predictions
tennis_mod1_pred = predict(tennis_mod1)
tennis_mod1_pred_ratio = sum(((tennis_mod1_pred[,"win"] >= .5) & (tennis$win == "win")) | 
                             ((tennis_mod1_pred[,"win"] < .5)  & (tennis$win == "loss")) )/nrow(tennis_mod1_pred)

