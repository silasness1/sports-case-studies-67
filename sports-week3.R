library(readr)
library(dplyr)
library(ggplot2)
library(e1071)
library(neuralnet)
library(ranger)
library(mgcv)
library(xgboost)

football_test = read_csv("raw-data/EuroTest.csv")
football_train = read_csv("raw-data/EuroTraining.csv")

football_train %>%
  group_by(Year, GroupStage) %>%
  summarize(n = n())

# number of teams per year in training
football_train %>%
  group_by(Year, Team, GroupStage) %>%
  summarize(n = n()) %>%
  group_by(Year, GroupStage) %>%
  summarize(n_teams = n())

# number of teams per year in test
football_test %>%
  group_by(Year, Team, GroupStage) %>%
  summarize(n = n()) %>%
  group_by(Year, GroupStage) %>%
  summarize(n_teams = n())

# using 2012 as an example of old system
#       See we have 24 matches (48 observations) in the group stage
#             This comes from choose(4,2) = 6 *4groups = 24 games
#       We have 7 matches in the knockout stage among the final 8 teams
#           (14 observations)
football_train %>%
  filter(Year == 2012) %>%
  group_by(GroupStage) %>%
  summarize(n = n())

# using 2020 as an example of new system
#       See we have 36 matches (72 observations) in the group stage
#             This comes from choose(4,2) = 6 *6groups = 36 games
#       We have 15 matches in the knockout stage among the final 16 teams
#           (30 observations)
football_train %>%
  filter(Year == 2020) %>%
  group_by(GroupStage) %>%
  summarize(n = n())

quantile_cuts =  quantile(football_train$MarketValue, probs = c(0,.25,.5,.75,1))
football_train %>%
  mutate(bin_market_value = cut(MarketValue, quantile_cuts))%>%
ggplot()+
  geom_jitter(aes(x = Year, y= Goals), width = .01)+
  facet_wrap(~bin_market_value)


# possible ways of splitting test/train data ----
#  Note: we only have 492 observations
         # size changes per year

#    - Rolling Window (lots of data wasted! Not enough)
#         Predict 2012 with 2008
#         Predict 2016 with 2012

#         One person vs one person (head to head changes year after year)
#             But in our case team vs team variables less time dependence?

# 1,2,3,4,5
 
# 1: nn (d = ) how many depth of neural network (grid search/CV piece)
#       For each value of d, we do it on each fold?

# even for hyper parameter tuning avoid including 2024

#so using k-fold among 2008-2020 to pick best parameters?
#     
#     test-train still 2024? 

#      Same test/split used for each model type
#      When comparing poisson vs model_x compare on same folds?

# Going between W/D/L vs Raw Goal prediction? 
#      Changing what kind of models to use? 
#      Recombine, will have to 

# One TODO - Reformat into W/L/D? 
#     Switch to classification? 
#    Could predict goals -> transform into W/D/L
#         Intercept? .5 window 
#             If we estimated regression (if goal diff < 1) -> Draw?
          
#         Could open up to classification & regression methods? 
                    # Could we directly classify by first encoding WDL 
                    # Use goal diff as covariate for WDL? 
                        # 
                    # do we lose info by ignoring goal diff

# Part 1 done as a goal

# Tentative comparison

# common split 
# everyone tries a different model
# euro_train1 euro_test1 
# euro_train2 euro_test2
# euro_train3 

#cv function?


# euro_2024  euro_2025 leave one out only

# hyper parameter doing simulatenous

# Rohit - XGBoost (sequentially)
# Sriram - Random Forest (aggregates trees)
# Nick - Neural Network
# Silas - Poisson w/ w/out lasso?
# Claude - The rest (has 2028 results)


# meet..... When? Wednesday? Tuesday night?
# presentation on Wednesday as well? 
# Put R code on github

# MAE/RMSE 

# Pick some metric for hyperparemeter tuning?

# Solid answer for what happens when future leaks into past?

# THINGS TO ASK
#    Multiclass? 
#    Whether to include 2024 in training

# Not combining into one row for WDL? 

#   Include or not include 2024? 

# Part4 of Lab03_FS2: 
#       - could redo conversion of predicted goals to WDL
#       - Measure used depends on whether using 

#     Numerical encoding? 1,2,3     (-1 0 1)? 
#     Order? 


# Act as if going to use predict for 2028? VS Pretend we don't know 2024

# Splitting train test (do we split it once or multiple times?)

# assuming once: 
#      For each fold:
          # working with tree complexity parameter (picking among parameters)

# In general:
#     Each model created for each fold (final is... average? best? )

#  Two questions: 
#       One model - one fold?
#       Or is splitting strategy separate?
#          - how to approximate generalization error? 


# In the veeeery final model? final prediction is 2024?


# 

#    - Expanding window  
#         Predict 2012 with 2008            (waaaay imbalanced?)
#         Predict 2016 with 2008 & 2012
#         Predict 2020 with 2008 & 2012 & 2016

#    - Leave One Tournament Out (future info leaked into past)
#         Predict 2024 with {2008, 2012, 2016, 2020,     }
#         Predict 2020 with {2008, 2012, 2016,      ,2024}
#         Predict 2016 with {2008, 2012,     , 2020 ,2024}
#         E.g. Predicting 2020 given MarketValue from 2024 known from future going backwards?
#            How time-dependent are tournaments?
#            Are tournaments independent?
#            Does it depend it on whether we include Year as a covariate
#            Independence by assumption?
#           2008 team got better or marketvalue only difference in teams
#                  Not using team-specifics in general (differences are less time dependent?)
#

#     - Last Season
#         Use last as test rest as training
#         

# how to split might depend

# Splits for hyper parameter tuning?

# Valid

# Very final model (on all data?)

# Is splitting a tournament in half leakage?

# Use the same training test split for all different model types?

# at the very end use 2024?

# Okay never use 2024 if we want 2024 as the final prediction? 

# training split? 

# trying out multiple split schemes?
#      Rather just choose one method and stick with it



#     - Nested CV
#         Outer CV for test vs train
#         Inner CV for tuning parameters

#     - Stratified CV
#          - Use something combination like c(Year,MarketValue) & split entire data into groups while guarenteeing 
#            representation of market value 

# experimenting with svm----
svm(Goals~GroupStage + MarketValue + FifaRank + UefaPoints + CLPlayers + Year, data = football_train) -> svm_football
neuralnet(Goals~GroupStage + MarketValue + FifaRank + UefaPoints + CLPlayers + Year, data = football_train) -> nn_football
gam(Goals~GroupStage + MarketValue + FifaRank + UefaPoints + CLPlayers + Year, data = football_train)-> gam_football
ranger(Goals~GroupStage + MarketValue + FifaRank + UefaPoints + CLPlayers + Year, data = football_train)-> rf_football
xgboost()
