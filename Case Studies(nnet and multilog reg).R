
football_test = read_csv("C:/Users/962ni/Downloads/EuroTest.csv")
football_train = read_csv("C:/Users/962ni/Downloads/EuroTraining.csv")
eurodata=rbind(football_test,football_train)

euro.matches <- eurodata %>%
  
  select(
    Year, Team, Opponent, Goals,
    GroupStage, GDP, MarketValue,
    FifaRank, UefaPoints, CLPlayers
  ) %>%
  
  rename(
    Team_A = Team,
    Goals_A = Goals
  ) %>%
  
  inner_join(
    eurodata %>%
      select(Year, Team, Opponent, Goals) %>%
      rename(
        Team_B = Team,
        Goals_B = Goals
      ),
    by = c("Year", "Opponent" = "Team_B",
           "Team_A" = "Opponent")
  )
View(euro.matches)
euro.matches <- euro.matches %>%
  filter(Team_A < Opponent) %>%
  distinct(Year, Team_A, Opponent, .keep_all = TRUE)
View(euro.matches)


euro <- euro.matches %>%
  mutate(
    Result = case_when(
      Goals_A > Goals_B ~ "Win",
      Goals_A == Goals_B ~ "Draw",
      Goals_A < Goals_B ~ "Loss"
    ),
    Result = factor(
      Result,
      levels = c("Loss", "Draw", "Win")
    ),
    GroupStage = factor(GroupStage)
  )

# Inspect outcome distribution
table(euro$Result)

# Number of matches per tournament
table(euro$Year)
predictor_formula <- ~ GroupStage +
  GDP +
  MarketValue +
  FifaRank +
  UefaPoints +
  CLPlayers

evaluate_model <- function(probabilities, actual) {
  
  classes <- levels(actual)
  probabilities <- probabilities[, classes, drop = FALSE]
  
  predicted <- classes[max.col(probabilities)]
  
  accuracy <- mean(predicted == as.character(actual))
  
  actual_index <- match(actual, classes)
  
  log_loss <- -mean(
    log(pmax(
      probabilities[cbind(seq_along(actual), actual_index)],
      1e-15
    ))
  )
  
  actual_matrix <- sapply(
    classes,
    function(cl) as.numeric(actual == cl)
  )
  
  brier <- mean(
    rowSums((probabilities - actual_matrix)^2)
  )
  
  c(
    Accuracy = accuracy,
    LogLoss = log_loss,
    BrierScore = brier
  )
}

prepare_data <- function(train, test) {
  
  X_train <- model.matrix(
    predictor_formula,
    data = train
  )[, -1, drop = FALSE]
  
  X_test <- model.matrix(
    predictor_formula,
    data = test
  )[, -1, drop = FALSE]
  
  # Align test columns with training columns
  missing_cols <- setdiff(
    colnames(X_train),
    colnames(X_test)
  )
  
  for (col in missing_cols) {
    X_test <- cbind(
      X_test,
      setNames(data.frame(rep(0, nrow(X_test))), col)
    )
  }
  
  extra_cols <- setdiff(
    colnames(X_test),
    colnames(X_train)
  )
  
  X_test <- X_test[
    , setdiff(colnames(X_test), extra_cols),
    drop = FALSE
  ]
  
  X_test <- X_test[, colnames(X_train), drop = FALSE]
  
  # Standardise using training data
  means <- colMeans(X_train)
  sds <- apply(X_train, 2, sd)
  
  sds[!is.finite(sds) | sds == 0] <- 1
  
  X_train <- scale(X_train, center = means, scale = sds)
  X_test <- scale(X_test, center = means, scale = sds)
  
  list(
    X_train = X_train,
    X_test = X_test
  )
}

set.seed(123)

years <- sort(unique(euro$Year))

results <- data.frame()

for (yr in years) {
  
  cat("\nLeaving out tournament:", yr, "\n")
  
  train <- euro %>%
    filter(Year != yr)
  
  test <- euro %>%
    filter(Year == yr)
  
  # Check that all three outcome classes occur in training
  if (!all(levels(euro$Result) %in%
           unique(as.character(train$Result)))) {
    warning(paste("Skipping year", yr,
                  "- a class is missing from training"))
    next
  }
  
  prepared <- prepare_data(train, test)
  
  X_train <- prepared$X_train
  X_test <- prepared$X_test
  
  y_train <- train$Result
  y_test <- test$Result
  
  # =====================================
  # MODEL 1: MULTINOMIAL LOGISTIC REGRESSION
  # =====================================
  
  train_logit <- as.data.frame(X_train)
  train_logit$Result <- y_train
  
  model_logit <- multinom(
    Result ~ .,
    data = train_logit,
    maxit = 1000,
    decay = 0.01,
    trace = FALSE
  )
  
  prob_logit <- predict(
    model_logit,
    newdata = as.data.frame(X_test),
    type = "probs"
  )
  
  prob_logit <- as.matrix(prob_logit)
  
  # =====================================
  # MODEL 2: NEURAL NETWORK
  # =====================================
  
  model_nn <- nnet(
    x = X_train,
    y = class.ind(y_train),
    size = 5,
    softmax = TRUE,
    decay = 0.01,
    maxit = 1000,
    trace = FALSE
  )
  
  prob_nn <- predict(
    model_nn,
    newdata = X_test,
    type = "raw"
  )
  
  prob_nn <- as.matrix(prob_nn)
  colnames(prob_nn) <- levels(y_train)
  
  # =====================================
  # EVALUATION
  # =====================================
  
  metrics_logit <- evaluate_model(prob_logit, y_test)
  metrics_nn <- evaluate_model(prob_nn, y_test)
  
  results <- rbind(
    results,
    
    data.frame(
      Year = yr,
      Model = "Multinomial Logistic Regression",
      Accuracy = metrics_logit["Accuracy"],
      LogLoss = metrics_logit["LogLoss"],
      BrierScore = metrics_logit["BrierScore"]
    ),
    
    data.frame(
      Year = yr,
      Model = "Neural Network",
      Accuracy = metrics_nn["Accuracy"],
      LogLoss = metrics_nn["LogLoss"],
      BrierScore = metrics_nn["BrierScore"]
    )
  )
}

print(results)

average_results <- results %>%
  group_by(Model) %>%
  summarise(
    Mean_Accuracy = mean(Accuracy),
    Mean_LogLoss = mean(LogLoss),
    Mean_BrierScore = mean(BrierScore),
    .groups = "drop"
  )

print(average_results)

ggplot(
  results,
  aes(x = factor(Year), y = Accuracy, fill = Model)
) +
  geom_col(position = "dodge") +
  labs(
    title = "Model Accuracy Across UEFA Euro Tournaments",
    x = "Tournament Year",
    y = "Accuracy"
  ) +
  theme_minimal()
