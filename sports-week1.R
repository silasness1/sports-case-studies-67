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
# linear models as baselines. The 3 datasets have different response types {ordinal, binary, metric}

# Section 1: Read in the csv----
football_test = read_csv("raw-data/EuroTest.csv")
football_train = read_csv("raw-data/EuroTraining.csv")
handball_long = read_csv("raw-data/handball_long.csv")
handball_wide = read_csv("raw-data/handball_wide.csv")
tennis = read_csv("raw-data/SlamsData.csv")

# Section 2: Look at read in types----
str(football_test)
str(football_train)

# convert vars to factors and integers where necessary 
clean_football_types <- function(df) {
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

# (hb graph 1) Team-level average {goals, betting probabilities} over time
#             Observation: Some teams aren't represented in all seasons!
#                          Avg goals within a team across seasons is pretty stable  
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
    labs(title = paste0("Team-level Average ", var_name, " Over Seasons"), 
         y = paste0("Average Seasonal ", var_name), 
         fill = "Number of Games\n Averaged Over in Season", 
         x = "Season")+
    theme(axis.text.x = element_text(angle = -90))
}
plot_season(handball_long, goals, "Goals")
plot_season(handball_long, bet.probs, "Betting Probabilities")

# (hb graph 2) Avg Betting probabilities vs Avg goals by seasons
#              Observation: No need for interaction coef between season & probabilities
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

# (hb graph 3) Home vs Away's impact on {betting, goals}
#             Observation: Sports betting seems to favor home games
#                          more than the home game advantage in # of goals suggests
#                          But maybe that makes sense because winning 
#                             happens as soon as you have just 1 more goal
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

# (hb graph 4) handball long average goals per team
#              Obs: number of avg goals for a team is roughly btw 23 and 32
handball_long %>% 
  group_by(team) %>%  
  summarise(team_mean_goals = mean(goals), .groups = "drop") %>%
  mutate(team = fct_reorder(team, team_mean_goals)) -> hb_team_means
  ggplot(hb_team_means)+
  geom_col(aes(y = team, x = team_mean_goals))+
  labs(title = "Handball Team Avg Goals per Match")

# (hb graph 5) violin plot goals
handball_long %>% 
  group_by(team) %>%
  mutate(team_median_goals = median(goals)) %>%
  ungroup() %>%
  mutate(team = fct_reorder(team, team_median_goals)) %>%
  ggplot()+
  geom_violin(aes(y = team, x = goals))+
  labs(title = "Goal Distribution over Matches Sorted by Median")

# (hb graph 6) Average score diff between unique combinations of teams
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
  dplyr::select(-team3) %>%
  rbind(tmp) %>% 
  # left_join(hb_team_means, by = join_by(team1 == team)) %>%   #couldn't get it to sort properly
  # mutate(team1 = fct_reorder(team1, team_mean_goals),
  #        team2 = fct_reorder(team2, team_mean_goals)) %>%
  ggplot()+
  geom_tile(aes(y = team1, x = team2, fill = mean_scorediff))+
  scale_fill_gradient2(low = "red", high ="blue",
                       mid = "white")+
  theme(axis.text.x=element_text(angle = -90))+
  labs(fill = "Team 1 Average Score\n Difference over Team 2",
       title = "Avg. Goal Differences for Handball Matchups", 
       caption = "Note: Score diff of team1 - team2 is consistent among the unique pairs of teams")

# (hb graph 7) Betting prob vs scorediff
#         conclusion: a cubic relationship makes a lot of sense

ggplot(data = handball_wide, aes(x = bet.probs.diff, y = scorediff))+
  geom_point()+
  geom_smooth(formula = y ~ poly(x, 3))+
  scale_x_continuous(labels = scales::label_percent())+
  labs(title = "Handball Betting Probabilities vs Score Difference",
       x = "Difference in Betting Probabilities", 
       y = "Game Score Difference")+
  theme_bw()

# (hb graph 8) Interactions between season & bet.prob.diff
#         Conclusion: interaction doesn't seem so necessary
#          Approximately same relationship btw bets & scorediff across diff seasons
ggplot(data = handball_wide, aes(x = bet.probs.diff, y = scorediff))+
  geom_point()+
  geom_smooth()+
  facet_wrap(~season)

# (hb graph 9) Plotting seasons vs scorediff: 
#         Conclusion: not much difference..maybe a little less variance in season 2019
ggplot(data = handball_wide)+
  geom_jitter(aes(x = season, y = scorediff), width = .2)

  
# Section 5: Visualizing football (euro data)----

# (football graph1) avg goals per team
# TBD: 

# (football graph2) avg goals per match
# TBD:  

# (football graph3) pairwise scatterplots
scatterplotMatrix(football_train) #pre-built correlogram

# (football graph4) correlogram 
#        Obs: Lots of correlation between MarketValue, FifaRank, UefaPoints, CLPlayers
football_train %>%
  mutate(across(everything(), as.numeric)) %>%   #https://dplyr.tidyverse.org/reference/across.html
  cor() %>%
  corrplot() # https://cran.r-project.org/web/packages/corrplot/vignettes/corrplot-intro.html

# (football graph5 - graph12) variable plotted against Goals 
#           Obs: Team vs goal:  Some countries do better than others and havehave fewer games
#                               !!! Teams that do better have more observations

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

# (football graph13) Number of team obs vs avg goals
#         Obs: Groupstage 1 teams with low number of goals get less observations
football_train %>%
  group_by(Team) %>%
  summarize(mean_goals = mean(Goals), 
            num_games = n(),
            mean_group_stage = mean(as.numeric(GroupStage)-1)) %>%  # factor had levels 1,2
  ungroup() %>%
  ggplot(aes(x = mean_goals, y = num_games, color = mean_group_stage))+
  geom_point(size = 2.5)+
  scale_color_continuous(palette = c("purple", "#FC9272", "#DE2D26"))+
  labs(x = "Team Mean # of Goals", 
       y = "Team Total # of Observed Games",
       title = "Football Train: Fewer Observations for Poor Performing Teams",
       color = "% of Matches \nwhere GroupStage = 1")+
  theme_bw()
  
# (football graph 14) Distribution of Goals in General
#      Seems reasonable to be poisson I guess? 
ggplot(data = football_train)+
  geom_histogram(aes(x = Goals))

# Section 6: Visualizing tennis data  ----

# (tennis graph 1) win vs regressors
# TBD:



# Section 7: Linear Regression & Assumption Checks: Handball----

# Fitting Initial hb model
handball_wide %>%
  mutate(season = as.numeric(season)) -> hb_lm_dat
lm(hb_lm_dat, formula = scorediff ~ bet.probs.diff + season) -> hb_lm
tidy(hb_lm)
glance(hb_lm)

# Checking assumptions
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
plot(hb_lm)


# Testing normality based on Q-Q plot
#       Conclusion: clearly it deviates from a normal distribution
# shapiro.test(hb_lm_comparison$pred_error)
shapiro.test(hb_lm$residuals)


# Testing Homoskedasticity after observing scale-location
#       Conclusion: test does not rejecting equal variances
ncvTest(hb_lm) 


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

# exogeneity: excluded variables
#     talent, reputation, injuries? 

# Multicollinearity & Rank(X) Assumptions: 
# using VIF - how well other regressors explain one left out
#     conclusion: multicolinearity not likely to be an issue here
#                 https://www.r-bloggers.com/2023/12/exploring-variance-inflation-factor-vif-in-r-a-practical-guide/
car::vif(mod = hb_lm)

# Predict goal differences
predict(hb_lm) -> hb_lm.pred

data_frame(actual_scorediff = handball_wide$scorediff, 
           predicted_scorediff = hb_lm.pred) %>%
  mutate(pred_error = actual_scorediff - predicted_scorediff, 
         abs_pred_error = abs(pred_error)) -> hb_lm_comparison

# Mean abs error 
mean(abs(hb_lm$residuals))-> mean_abs_error_hb_simple

# Density of prediction errors
hb_lm_comparison %>%
  ggplot()+
  geom_density(aes(x = pred_error))

# Boxplot of absolute prediction errors
#        observation: we have a skewed distribution of absolute prediction error 
ggplot(data = hb_lm_comparison)+
  geom_boxplot(aes(x = abs_pred_error))+
  geom_vline(aes(xintercept = mean(hb_lm_comparison$abs_pred_error), color = "red") )+
  labs(caption = "Red line is mean absolute error.",
       title = "Handball lm() Absolute Prediction Error", 
       x = "Absolute Prediction Error")+
  theme_bw()+
  theme(legend.position = "none")


# Section 8: Handball Model Improvement -----

#   How might we change the model accordingly to better address this issue? 
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
boxCox(hb_lm, family = "yjPower", plotit = T, interp = F)


# Idea 3: GLM idea (SKIP)
#         Poisson: only takes positive values as well... 

# Idea 1: Bayesian t-distributed errors
#       Eventual conclusion: no difference really... maybe a better fitted qq plot
#       Exploring the "change the distrubtion to a t-distribution" idea
#       Looks like it's possible in bayesian set up with brms
#       https://michael-franke.github.io/Bayesian-Regression/practice-sheets/04d-distributional-models.html
#       https://paulbuerkner.com/brms/
#       I don't really understand fully what I'm doing here but I'm playing around anyway

#       Fitting Bayesian T-Distributed Errors
#              scorediff = .05*season + 7.11*bet.probs.diff - .14 
#              eps | X ~ t(sigma = 4.21, nu = 23.24)
fit_t_regression <- 
  brm(data = hb_lm_dat, 
    scorediff ~ season + bet.probs.diff,
    family = student()
  )
summary(fit_t_regression)



#       Bayesian estimation graphs
plot(fit_t_regression)

#       T-regression predictions
predict(fit_t_regression)-> t_reg_pred

data_frame(t_reg_pred = t_reg_pred[,1], 
           actual = handball_wide$scorediff) %>%
  mutate(pred_error = actual - t_reg_pred, 
         abs_pred_error = abs(pred_error)) -> t_reg_compare

#      T-regression mean abs error
mean(t_reg_compare$abs_pred_error) -> mean_abs_error_hb_t_reg #3.486

#       T-distributed Errors Residual Boxplot
#                Conclusion: almost nooooooo difference. 
ggplot(data = t_reg_compare)+
  geom_boxplot(aes(x = abs_pred_error))+
  geom_vline(aes(xintercept = mean(t_reg_compare$abs_pred_error), color = "red") )+
  labs(caption = "Red line is mean absolute error.")+
  labs(title = "t-distr. based bayesian regression")


#       QQ-Plot for Bayesian t-distributed Errors
#           https://library.virginia.edu/data/articles/understanding-q-q-plots
#           used ?qqplot() example
qqplot(qt(ppoints(500), df = 23.24), t_reg_compare$pred_error)
qqline(t_reg_compare$pred_error, 
       distribution = function(p) qt(p, df = 23.24),
       probs = c(0.1, 0.6), col = 2)



#  Idea 6: How about preprocessing the season data? 
#          It looks like I already have by accident by converting to a factor and back
summary(hb_lm_dat$season)
handball_wide$season 

# Idea 7: looking at relationships with predictors finally
#         Yes, I'm dumb/skipped this step earlier...
#         Turns out betting prob. is cubic relationship with scorediff

#         polynomial regression: https://www.geeksforgeeks.org/r-language/polynomial-regression-in-r-programming/
lm(data = hb_lm_dat, formula = scorediff ~ poly(bet.probs.diff, 3) + season) -> poly_mod

summary(poly_mod)
plot(poly_mod)

#        checking the absolute pred errors: polynomial model
#        Conclusion: no better than the plain linear model
predict(poly_mod)-> poly_mod_pred


data_frame(poly_mod_pred = poly_mod_pred, 
           actual = handball_wide$scorediff) %>%
  mutate(pred_error = actual - poly_mod_pred, 
         abs_pred_error = abs(pred_error)) -> poly_compare

#        mean abs error for hb polynomial model
mean(poly_compare$abs_pred_error) -> mean_abs_error_hb_poly

#        Boxplot for hb polynomial model
ggplot(data = poly_compare)+
  geom_boxplot(aes(x = abs_pred_error))+
  geom_vline(aes(xintercept = mean(.data[["abs_pred_error"]]), color = "red") )+
  labs(caption = "Red line is mean absolute error.")+
  labs(title = "Polynomial Model Absolute Prediction errors")


# HB model comparison summary between polynomial & plain linear & t-regression
lm(hb_lm_dat, formula = scorediff ~ poly(bet.probs.diff,3)*season) -> poly_interact_mod
plot(poly_interact_mod)
rbind(poly = glance(poly_mod),
      linear=glance(hb_lm),
      poly_interactions = glance(poly_interact_mod))

data_frame(mean_abs_error = c(mean_abs_error_hb_simple, mean_abs_error_hb_t_reg, mean_abs_error_hb_poly),
           model = c("lm(scorediff ~ bet + season)", "brms(family = student())", "lm(scorediff ~ poly(bet,3) + season)"))-> hb_compare_mean_abs_err
print(tibble(hb_compare_mean_abs_err))

# Section 9: Poisson Regression: Football-----

# CARS(fox) textbook Notes on Poisson regression
#                  : - poisson's default link function is log(though sqrt and identity are possible)
#                     - GLMs are typically fit by MLE using iterative weighted least squares 
#                             See associated wiki: https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
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

# compare the mean and the variance of the number of goals (equidispersion).
#      See wiki for context: https://en.wikipedia.org/wiki/Poisson_regression#Overdispersion_and_zero_inflation

# TBD

# Alternative model Ideas
#     Idea 1: when maybe to use multinomial? 
#     Idea 2: try a sqrt link or identity link instead
#     Idea 3: try to implement robust version of poisson? https://stats.oarc.ucla.edu/r/dae/poisson-regression/
#     Idea 4: Possible overdispersion -> negative binomial model or quasi-likelihood (what/how??)

# Section 10: Logistic Regression: Tennis----

#TBD

