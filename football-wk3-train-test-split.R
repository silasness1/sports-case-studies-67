
library(dplyr)
library(readr)

# combine existing
football_test = read_csv("raw-data/EuroTest.csv")
football_train = read_csv("raw-data/EuroTraining.csv")

rbind(football_train, football_test) %>% arrange(Year) -> football
football$match_id = rep(1:(nrow(football)/2), each =2)

# counts by year
football %>%
  group_by(Year) %>%
  summarize(n = n())

# write separate for each year
filter_and_write_year <- function(this_year) {
  football %>%
    filter(Year == this_year) %>%
    write_csv(file = paste0("raw-data/euro_",this_year,".csv"))
}

sapply(seq(2004,2024,4), filter_and_write_year)


