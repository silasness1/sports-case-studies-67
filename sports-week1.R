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

# read in the csv----
football_test = read_csv("raw-data/EuroTest.csv")
football_train = read_csv("raw-data/EuroTraining.csv")
handball_long = read_csv("raw-data/handball_long.csv")
handball_wide = read_csv("raw-data/handball_wide.csv")
tennis = read_csv("raw-data/SlamsData.csv")

# look at read in types----
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


# check for missing values ----
check_for_na <- function(df) {
  df %>% 
    summarise(across(everything(), function(x) sum(is.na(x))))
}

map(list(football_train, football_test, handball_long, handball_wide, tennis), check_for_na)

# Visualizing handball----

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

  
# Visualizing Euro (football data) avg number of goals per team and match----

# TBD:  

# Visualizing tennis data share of won matches ----

# TBD: 

# Visualizing (handball: scorediff vs regressors) -----

# TBD

# Visualizing (football: Goals vs regressors)-----

# TBD

# Visualizing (tennis: win vs regressors)-----

#TBD

# Linear Regression: Handball----

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

#    observation: we have a skewed distribution of absolute prediction error 

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



# Poisson Regression: Football-----

# TBD

# Logistic Regression: Tennis----

#TBD

