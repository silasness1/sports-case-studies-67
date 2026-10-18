# load libraries -----
library(readr)
library(dplyr)
library(purrr)
library(ggplot2)
library(forcats)
library(gridExtra)
library(broom)

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
  mutate(season = as.numeric(season)) %>%
  lm(formula = scorediff ~ bet.probs.diff + season) -> hb.lm
tidy(hb.lm)
glance(hb.lm)

# checking assumptions
plot(hb.lm)

# Predict goal differences
predict(hb.lm) -> hb.lm.pred

data_frame(actual_scorediff = handball_wide$scorediff, 
           predicted_scorediff = hb.lm.pred) %>%
  mutate(pred_error = actual_scorediff - predicted_scorediff, 
         abs_pred_error = abs(pred_error)) -> hb.lm.comparison

# boxplot 
ggplot(data = hb.lm.comparison)+
  geom_boxplot(aes(x = abs_pred_error))+
  geom_vline(aes(xintercept = mean(hb.lm.comparison$abs_pred_error), color = "red") )+
  labs(caption = "Red line is mean absolute error.")

# Poisson Regression: Football-----

# TBD

# Logistic Regression: Tennis----

#TBD

