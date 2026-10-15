# load libraries -----
library(readr)
library(dplyr)
library(purrr)

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

