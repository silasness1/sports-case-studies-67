# load libraries -----
library(readr)
library(dplyr)
library(purrr)
library(ggplot2)
library(forcats)
library(gridExtra)
library(broom)
library(car)
library(MASS)
library(brms)
library(corrplot)
library(stringr)

# README: 
# This script is part of Case Studies at TU Dortmund -- week 1 of the sports project
# We're provided 3 sports datasets and are working on EDA and setting up 
# linear models as baselines. The 3 datasets have different response types {ordinal, binary, }

# Section 1: Read in the csv----
football_test = read_csv("raw-data/EuroTest.csv")
football_train = read_csv("raw-data/EuroTraining.csv")
handball_long = read_csv("raw-data/handball_long.csv")
handball_wide = read_csv("raw-data/handball_wide.csv")
tennis = read_csv("raw-data/SlamsData.csv")

# Section 2: Look at read in types----
str(football_test)
str(football_train)

clean_football_types <- function(df) {
  # convert vars to factors and integers where necessary 
  df  %>%
    mutate(across(c(Team, Opponent, GroupStage, Year), as.factor)) %>%
    mutate(across(c(Goals, FifaRank, UefaPoints), as.integer))
}

football_test <- clean_football_types(football_test)
football_train <- clean_football_types(football_train)

str(handball_long)
handball_long <- handball_long %>%
  mutate(across(c(team, season, home_away), as.factor)) %>%
  mutate(across(c(matchid, goals), as.integer))

str(handball_wide)
handball_wide <- handball_wide %>%
  mutate(across(c(hometeam, awayteam, season), as.factor)) %>%
  mutate(across(c(homegoals, awaygoals, scorediff, matchid), as.integer))

str(tennis)
tennis <- tennis %>%
  mutate(across(c(player.1, player.2, year, tournament, surface, win), as.factor)) %>%
  mutate(across(c(sets.1, sets.2, rank, points), as.integer))


# Section 3: Check for missing values ----
check_for_na <- function(df) {
  df %>% 
    summarise(across(everything(), function(x) sum(is.na(x))))
}

map(list(football_train, football_test, handball_long, handball_wide, tennis), check_for_na)

# Section 4: Visualizing handball----

# How each team's {goals, betting probabilities} changes over time
plot_season <- function(df, variable, var_name) {
  df %>%
    group_by(team, season) %>%
    summarize(
      value = mean({{variable}}),
      num_games = n()
    ) %>%
    ggplot() + 
    geom_col(aes(y=value, x = season, fill = num_games)) +
    facet_wrap(~team) + 
    labs(title = paste0("Seasonal ", var_name), 
         y = paste0("Average Seasonal ", var_name), 
         fill = "Number of Games in Season", 
         x = "Season")
}
plot_season(handball_long, goals, "Goals")
plot_season(handball_long, bet.probs, "Betting Probabilities")

# Relationship between betting probabilities and goals
handball_long %>%
  group_by(team, season) %>%
  summarise(
    mean_goals = mean(goals), 
    mean_prob = mean(bet.probs)
  ) %>%
  ggplot()+
  geom_point(aes(x=mean_goals, y = mean_prob))+
  facet_wrap(~season)+
  labs(title= "Goals vs Betting Probabilities for Teams")

# Home vs Away's impact on {betting, goals}
plot_home_way <- function(df, variable, var_name) {
  df %>%
    group_by(team, home_away) %>%
    summarize(
      value = mean({{variable}}),
      num_games = n(),
      .groups = "drop"
    ) %>%
    mutate(team = fct_reorder(team, value)) %>%
    ggplot() + 
    geom_col(aes(y=team, x = value, fill = home_away), 
             position = "dodge") +
    labs(title = paste0("Home vs Away ", var_name), 
         y = "Team", 
         fill = "Home vs Away", 
         x = paste(var_name))
}

home_away_goals <- plot_home_way(handball_long, goals, "Goals")
home_away_bets <- plot_home_way(handball_long, bet.probs, "Betting Probabilities")

grid.arrange(home_away_bets, home_away_goals, nrow = 1)

# handball long average goals per team
handball_long %>% 
  group_by(team) %>%  
  summarise(team_mean_goals = mean(goals), .groups = "drop") %>%
  mutate(team = fct_reorder(team, team_mean_goals)) %>%
  ggplot()+
  geom_col(aes(y = team, x = team_mean_goals))+
  labs(title = "Handball Team Avg Goals per Match")

# violin plot goals
handball_long %>% 
  group_by(team) %>%
  mutate(team_median_goals = median(goals)) %>%
  ungroup() %>%
  mutate(team = fct_reorder(team, team_median_goals)) %>%
  ggplot()+
  geom_violin(aes(y = team, x = goals))+
  labs(title = "Goal Distribution over Matches Sorted by Median")

# average score diff between unique combinations of teams ???
handball_wide %>%
  mutate(hometeam = as.character(hometeam), 
         awayteam = as.character(awayteam),
         team1 = if_else(hometeam < awayteam, hometeam, awayteam),
         team2 = if_else(hometeam < awayteam, awayteam, hometeam), 
         team_matchup = paste(team1, team2, sep = " - "),
         team1_scorediff = if_else(team1 == hometeam, scorediff, -scorediff)
  ) %>%
  group_by(team_matchup, team1, team2) %>% 
  summarize(mean_scorediff = mean(team1_scorediff), games = n()) -> tmp
tmp %>% mutate(team3 = team1, 
               team1 = team2, 
               team2 = team3, 
               mean_scorediff = -mean_scorediff) %>% 
  rbind(tmp) %>% 
  ggplot()+
  geom_tile(aes(y = team1, x = team2, fill = mean_scorediff))+
  scale_fill_gradient2(low = "red", high ="blue",
                       mid = "white")+
  theme(axis.text.x=element_text(angle = -90))+
  labs(fill = "Team 1 Average Score\n Difference over Team 2",
       title = "Avg. Goal Differences for Handball Matchups", 
       caption = "Note: Score diff of team1 - team2 is consistent among the unique pairs of teams")

  
# Section 5: Visualizing Euro (football data) avg number of goals per team and match----

# TBD:  

# Section 6: Visualizing tennis data share of won matches ----

# TBD: 

# Section 7: Visualizing (handball: scorediff vs regressors) -----

# TBD

# Section 8: Visualizing (football: Goals vs regressors)-----

# correlograms
scatterplotMatrix(football_train) #pre-built correlogram

football_train %>%
  mutate(across(everything(), as.numeric)) %>%   #https://dplyr.tidyverse.org/reference/across.html
  cor() %>%
  corrplot() # https://cran.r-project.org/web/packages/corrplot/vignettes/corrplot-intro.html

to_plot = c("Team", "Opponent", "Year", "GroupStage", "GDP", "MarketValue", "UefaPoints", "CLPlayers")

plot_vs_goals <- function(var,df) {
  df %>%
    ggplot()+
    geom_jitter(aes(x=.data[[var]], y = Goals, color = .data[[var]]),
                height = .1)+
    # for explanation about .data[[var]] syntax see https://dplyr.tidyverse.org/articles/programming.html#indirection
    theme(axis.text.x = element_text(angle = -90), 
          legend.position = "None")+
    labs(title = paste("Handball Train:", var, "vs Goals"), 
         caption = "Jitter applied to make discrete units visible")
}
map(to_plot, function(x) {plot_vs_goals(x, football_train)})

# Section 9: Visualizing (tennis: win vs regressors)-----

#TBD

# Section 10: Linear Regression: Handball----

# fitting model
handball_wide %>%
  mutate(season = as.numeric(season)) -> hb_lm_dat
lm(hb_lm_dat, formula = scorediff ~ bet.probs.diff + season) -> hb.lm
tidy(hb.lm)
glance(hb.lm)

# checking assumptions
plot(hb.lm)
#       QQ- plot: shows not residuals not really normally distributed at extremes... we have more extreme events or 
#                 fatter tailed distribution (when 3rd deviation expected actually closer to fourth)
#                 meaning that more extreme values in empirical distribution 
#       Leverage: observation 1311 is pretty extreme but doesn't have all too much leverage. 
#                 Theres a 22 point difference in the game but it's also like a 20% betting prob.
#                       Seems very reasonable/not worth excluding
#                 https://en.wikipedia.org/wiki/Leverage_(statistics)
#       Scale-location: maybe ever so slight heteroskedasticity as you move away from 0 variance may increase
#                 https://library.virginia.edu/data/articles/diagnostic-plots
#                 In retrospect, this should have set off red flags!!! The pattern is 
#                 being caused by the non-linear relationship with bet.probs.diff

# testing normality based on Q-Q plot
shapiro.test(hb.lm.comparison$pred_error)
#       clearly it deviates from a normal distribution


# maybe using ncvTest in response to scale-location? 
ncvTest(hb.lm) # not rejecting equal variances

# exogeneity: what determines regressors
#     maybe teams stop having seasons if they play really poorly
#     bets are probably influenced from past matches or even prior betting probabilities    

# exogeneity: what determines outcome
#     scorediff is a deterministic function of home - away goals

# exogeneity: could outcome feedback into regressor? 
#     scorediff -?-> season: yeah potentially if they get disbanded as a team
#     scorediff -?-> bet.prob.diff: for future games yes... 
#                    Or maybe even psychologically... like really bad expectations 
#                    might cause worse player performance... 

# excluded variables: 
#     talent, reputation, injuries? 

# rk(X) or multicolinearity using VIF - how well other regressors explain one left out
car::vif(mod = hb.lm)
#     conclusion: multicolinearity not likely to be an issue here
#                 https://www.r-bloggers.com/2023/12/exploring-variance-inflation-factor-vif-in-r-a-practical-guide/

# Predict goal differences
predict(hb.lm) -> hb.lm.pred

data_frame(actual_scorediff = handball_wide$scorediff, 
           predicted_scorediff = hb.lm.pred) %>%
  mutate(pred_error = actual_scorediff - predicted_scorediff, 
         abs_pred_error = abs(pred_error)) -> hb.lm.comparison

# density
hb.lm.comparison %>%
  ggplot()+
  geom_density(aes(x = pred_error))

# boxplot 
ggplot(data = hb.lm.comparison)+
  geom_boxplot(aes(x = abs_pred_error))+
  geom_vline(aes(xintercept = mean(hb.lm.comparison$abs_pred_error), color = "red") )+
  labs(caption = "Red line is mean absolute error.")+
  labs(title = "Plain lm() Absolute Prediction Error")

#        observation: we have a skewed distribution of absolute prediction error 


# Model Improvement: how might we change the model accordingly to better address this issue? 
#        Idea 1: change the assumption about normally distributed errors to something with fatter tails, t()
#        Idea 2: log-transform the y data first? 
#               BUT we have negative response values here... so yjPower family?
#               https://bookdown.org/pbaumgartner/car-fox-weisberg/03-exploring-data.html#sec-chap03-4-2
#        Idea 3: maybe GLM of some sort? (none of families seemed right: poisson pos. only --> SKIP)
#        Idea 4: Non-constant variable/weighted least squares? (ncvTest --> SKIP )
#        Idea 5: What about huber regression? (don't think it's caused by outliers--SKIP)
#        Idea 6: Preprocessing the seasonal regressor (turns out I did already)
#        Idea 7: Reevaluate relationships between predictors & response

# Idea 2: the box-cox transforms (but generalized for response ata including neg values)
#     Conclusion: not a good idea to do a transformation... 
boxCox(hb.lm, family = "yjPower", plotit = T, interp = F)


# Idea 3: GLM idea (SKIP)
#         Poisson: only takes positive values as well... 

# Idea 1: Bayesian t-distributed errors
#       Eventual conclusion: no difference really... maybe a better fitted qq plot
#       Exploring the "change the distrubtion to a t-distribution" idea
#       Looks like it's possible in bayesian set up with brms
#       https://michael-franke.github.io/Bayesian-Regression/practice-sheets/04d-distributional-models.html
#       https://paulbuerkner.com/brms/
#       I don't really understand fully what I'm doing here but I'm playing around anyway

fit_t_regression <- 
  brm(data = hb_lm_dat, 
    scorediff ~ season + bet.probs.diff,
    family = student()
  )
summary(fit_t_regression)
#       so scorediff = .05*season + 7.11*bet.probs.diff - .14 
#       if we're willing to assume eps | X ~ t(sigma = 4.21, nu = 23.24)

#       trying to make a qq-plot with the original lm residuals vs this estimated t-distribution
#           https://library.virginia.edu/data/articles/understanding-q-q-plots
#           used ?qqplot() example
qqnorm(hb.lm$residuals)
qqplot(qt(ppoints(500), df = 23.24), hb.lm.comparison$pred_error)
qqline(hb.lm.comparison$pred_error, 
       distribution = function(p) qt(p, df = 23.24),
       probs = c(0.1, 0.6), col = 2)

plot(fit_t_regression)

#         looking at the boxplot afterwards
predict(fit_t_regression)-> t_reg_pred

data_frame(t_reg_pred = t_reg_pred[,1], 
           actual = handball_wide$scorediff) %>%
  mutate(pred_error = actual - t_reg_pred, 
         abs_pred_error = abs(pred_error)) -> t_reg_compare

ggplot(data = t_reg_compare)+
  geom_boxplot(aes(x = abs_pred_error))+
  geom_vline(aes(xintercept = mean(t_reg_compare$abs_pred_error), color = "red") )+
  labs(caption = "Red line is mean absolute error.")+
  labs(title = "t-distr. based bayesian regression")
#        Conclusion: almost nooooooo difference. 



#  Idea 6: How about preprocessing the season data? 
summary(hb_lm_dat$season)
handball_wide$season 
#          It looks like I already have by accident by converting to a factor and back



# Idea 7: looking at relationships with predictors finally
#         Yes, I'm dumb/skipped this step earlier...
summary(hb.lm)

#         Plotting betting prob vs scorediff
ggplot(data = hb_lm_dat, aes(x = bet.probs.diff, y = scorediff))+
  geom_point()+
  geom_smooth(formula = y ~ poly(x, 3))
#         conclusion: a cubic relationship makes a lot of sense

#         What about interactions between season & bet.prob.diff?
ggplot(data = handball_wide, aes(x = bet.probs.diff, y = scorediff))+
  geom_point()+
  facet_wrap(~season)

#         Plotting seasons vs scorediff: not much difference
ggplot(data = hb_lm_dat)+
  geom_point(aes(x = season, y = scorediff))

#         polynomial regression: https://www.geeksforgeeks.org/r-language/polynomial-regression-in-r-programming/
lm(data = hb_lm_dat, formula = scorediff ~ poly(bet.probs.diff, 3) + season) -> poly_mod

summary(poly_mod)
plot(poly_mod)

# checking the absolute pred errors: polynomial model
#        Conclusion: no better than the plain linear model
predict(poly_mod)-> poly_mod_pred

data_frame(poly_mod_pred = poly_mod_pred, 
           actual = handball_wide$scorediff) %>%
  mutate(pred_error = actual - poly_mod_pred, 
         abs_pred_error = abs(pred_error)) -> poly_compare

ggplot(data = poly_compare)+
  geom_boxplot(aes(x = abs_pred_error))+
  geom_vline(aes(xintercept = mean(.data[["abs_pred_error"]]), color = "red") )+
  labs(caption = "Red line is mean absolute error.")+
  labs(title = "Polynomial Model Absolute Prediction errors")

# model comparison between polynomial and plain linear
lm(hb_lm_dat, formula = scorediff ~ poly(bet.probs.diff,3)*season) -> poly_interact_mod
plot(poly_interact_mod)
rbind(poly = glance(poly_mod),
      linear=glance(hb.lm),
      poly_interactions = glance(poly_interact_mod))


# Section 11: Poisson Regression: Football-----

# CARS(fox) textbook Notes on Poisson regression
#                  : - poisson's default link function is log(though sqrt and identity are possible)
#                     - GLMs are typically fit by MLE using iterative weighted least squares 
#                     - eta = log(mu(x))
#                     - hat_eta = b_0 +... + b_k x_k 
#                     - Variance has a dispersion factor? var(y|x) = psi*V[mu(x)]

# Poisson distribution Wiki Notes: https://en.wikipedia.org/wiki/Poisson_distribution
#           Key assumptions: 1) number of events in a fixed interval of time if occuring 
#                              with a known constat mean rate and independently of 
#                              time since last event
#                            Are goals happening with a fixed known constant mean rate? 
#                            Are goals happening independent of the last goal time? 

# Initial Poisson Fit
glm(data = football_train,
    formula = Goals ~ Team + Opponent + Year + GroupStage + GDP
    + MarketValue + FifaRank + UefaPoints + CLPlayers, 
    family = poisson(),
    ) -> pois_football

summary(pois_football)
data_frame(coef_name = names(pois_football$coefficients),
           coef_value = pois_football$coefficients, 
           multiplicative_event_expect = exp(pois_football$coefficients)) %>%
  mutate(coef_category = case_when(
    str_detect(coef_name,"Year") ~ "Year", 
    str_detect(coef_name,"Team") ~ "Team",
    str_detect(coef_name, "Opponent") ~ "Opponent",
    .default = "Other"),
    coef_name = fct_reorder(factor(coef_name), rank(coef_category) + rank(multiplicative_event_expect)/1000))%>%
  ggplot()+
  geom_col(aes(y = coef_name,
               x = multiplicative_event_expect, 
               fill = coef_category))+
  labs(title = "Football Pois. Reg.: Interpreting a Unit increase in X_i for Expected # of Goal", 
       x = "Multiplicative Factor: exp(beta_i)",
       y = "Poisson Regressor Coefficient", 
       fill = "Regressor Category if Discrete")+
  geom_vline(xintercept = 1)

# multicolinearity
#        https://en.wikipedia.org/wiki/Variance_inflation_factor
#        Appears GDP, MarketValue and UefaPoints might be collinear
#        Look at Section 8: correlograms....
#                       MarketValue & FifaRank are strongly negatively correlated
#                       MarketValue & UefaPoints are stronly positively correlated
#                       Market Value & CLPlayers are strongly positively correlated
#                       FifaRank & UefaPoints are strongly negtaively correlated
#                       FifaRank & CLPlayers are moderately negatively correlated
#                       UefaPoints & CLPlayers are strongly positively correlated 
vif(pois_football)

# Checking model assumptions for poisson regression
plot(pois_football)


# Alternative model Ideas
#     Idea 1: when maybe to use multinomial? 
#     Idea 2: try a sqrt link or identity link instead
#     Idea 3: 

# Section 12: Logistic Regression: Tennis----

#TBD

