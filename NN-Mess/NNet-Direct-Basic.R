#!/usr/bin/env Rscript
library(nnet)
library(dplyr)
library(purrr)
library(mgcv)
library(ggplot2)


football_test = read_csv("C:/Users/962ni/Downloads/EuroTest.csv")
football_train = read_csv("C:/Users/962ni/Downloads/EuroTraining.csv")
eurodata=rbind(football_test,football_train)
View(eurodata)
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
  # MODEL 2: NEURAL NETWORK
  # =====================================
  
  model_nn <- nnet(
    x = X_train,
    y = class.ind(y_train),
    size = 2,
    softmax = TRUE,
    decay = 0.5,
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
  
  
  metrics_nn <- evaluate_model(prob_nn, y_test)
  
  results <- rbind(
    results,
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


euro <- euro.matches %>%
  mutate(
    Result = case_when(
      Goals_A > Goals_B ~ "Win",
      Goals_A == Goals_B ~ "Draw",
      Goals_A < Goals_B ~ "Loss"
    ),
    Result = factor(Result, levels = c("Loss", "Draw", "Win")),
    GroupStage = factor(GroupStage)
  )

# Check the outcome distribution
table(euro$Result)

# Check how many matches each tournament contains
table(euro$Year)

predictors <- c(
  "GroupStage",
  "GDP",
  "MarketValue",
  "FifaRank",
  "UefaPoints",
  "CLPlayers"
)

classes <- levels(euro$Result)



#
#
#
#
#
#





# Create one-hot encoded outcome matrix
make_y <- function(y, classes) {
  
  y_matrix <- matrix(
    0,
    nrow = length(y),
    ncol = length(classes),
    dimnames = list(NULL, classes)
  )
  
  y_matrix[cbind(
    seq_along(y),
    match(as.character(y), classes)
  )] <- 1
  
  y_matrix
}


# Prepare predictors using training data only
prepare_x <- function(train, test) {
  
  formula_x <- as.formula(
    paste("~", paste(predictors, collapse = " + "))
  )
  
  X_train <- model.matrix(formula_x, data = train)
  X_test  <- model.matrix(formula_x, data = test)
  
  # Make sure both matrices have the same columns
  missing_cols <- setdiff(colnames(X_train), colnames(X_test))
  
  if (length(missing_cols) > 0) {
    for (col in missing_cols) {
      X_test <- cbind(X_test, 0)
      colnames(X_test)[ncol(X_test)] <- col
    }
  }
  
  X_test <- X_test[, colnames(X_train), drop = FALSE]
  
  # Calculate scaling parameters using training data only
  x_mean <- colMeans(X_train)
  x_sd <- apply(X_train, 2, sd)
  
  x_sd[is.na(x_sd) | x_sd == 0] <- 1
  
  X_train <- scale(
    X_train,
    center = x_mean,
    scale = x_sd
  )
  
  X_test <- scale(
    X_test,
    center = x_mean,
    scale = x_sd
  )
  
  list(
    train = as.matrix(X_train),
    test = as.matrix(X_test)
  )
}

fit_nn <- function(train, test, size, decay, seed) {
  
  set.seed(seed)
  
  x <- prepare_x(train, test)
  
  y_train <- make_y(train$Result, classes)
  
  model <- nnet(
    x = x$train,
    y = y_train,
    size = size,
    softmax = TRUE,
    decay = decay,
    maxit = 1000,
    MaxNWts = 10000,
    trace = FALSE
  )
  
  probabilities <- predict(
    model,
    newdata = x$test,
    type = "raw"
  )
  
  probabilities <- as.matrix(probabilities)
  colnames(probabilities) <- classes
  
  # Avoid numerical problems in log loss
  probabilities <- pmax(probabilities, 1e-15)
  probabilities <- probabilities / rowSums(probabilities)
  
  probabilities
}


# Multiclass log loss
calculate_logloss <- function(actual, probabilities) {
  
  actual_index <- match(as.character(actual), classes)
  
  -mean(log(
    probabilities[cbind(seq_along(actual_index), actual_index)]
  ))
}


# Classification metrics
calculate_metrics <- function(actual, probabilities) {
  
  predicted <- factor(
    classes[max.col(probabilities)],
    levels = classes
  )
  
  confusion <- table(
    factor(actual, levels = classes),
    predicted
  )
  
  accuracy <- mean(predicted == actual)
  
  recall <- diag(confusion) / rowSums(confusion)
  precision <- diag(confusion) / colSums(confusion)
  
  recall[!is.finite(recall)] <- NA
  precision[!is.finite(precision)] <- NA
  
  f1 <- 2 * precision * recall / (precision + recall)
  f1[!is.finite(f1)] <- NA
  
  c(
    Accuracy = accuracy,
    Balanced_Accuracy = mean(recall, na.rm = TRUE),
    Macro_F1 = mean(f1, na.rm = TRUE),
    Log_Loss = calculate_logloss(actual, probabilities)
  )
}

#
#
#
#
#
#
#

tune_nn <- function(outer_train) {
  
  inner_years <- sort(unique(outer_train$Year))
  
  if (length(inner_years) < 2) {
    stop("Not enough tournament years for inner cross-validation.")
  }
  
  # Candidate hyperparameters
  grid <- expand.grid(
    size = c(2, 4, 6),
    decay = c(0.001, 0.01, 0.1)
  )
  
  tuning_results <- data.frame()
  
  for (g in seq_len(nrow(grid))) {
    
    size_value <- grid$size[g]
    decay_value <- grid$decay[g]
    
    fold_losses <- c()
    
    for (validation_year in inner_years) {
      
      inner_train <- outer_train %>%
        filter(Year != validation_year)
      
      inner_valid <- outer_train %>%
        filter(Year == validation_year)
      
      # Repeat each configuration with different initialisations
      for (seed in 1:3) {
        
        probabilities <- fit_nn(
          train = inner_train,
          test = inner_valid,
          size = size_value,
          decay = decay_value,
          seed = seed
        )
        
        fold_losses <- c(
          fold_losses,
          calculate_logloss(
            inner_valid$Result,
            probabilities
          )
        )
      }
    }
    
    tuning_results <- rbind(
      tuning_results,
      data.frame(
        size = size_value,
        decay = decay_value,
        mean_logloss = mean(fold_losses)
      )
    )
  }
  
  # Select the configuration with the lowest validation log loss
  best <- tuning_results %>%
    arrange(mean_logloss) %>%
    slice(1)
  
  list(
    best = best,
    results = tuning_results
  )
}


#
#
#
#Outer CV
#
#
#

outer_years <- sort(unique(euro$Year))

outer_predictions <- list()
outer_tuning <- list()

for (yr in outer_years) {
  
  cat("\nOuter test tournament:", yr, "\n")
  
  # Split outer training and test data
  outer_train <- euro %>%
    filter(Year != yr)
  
  outer_test <- euro %>%
    filter(Year == yr)
  
  # 1. Tune using inner CV
  tuning <- tune_nn(outer_train)
  
  best_size <- tuning$best$size
  best_decay <- tuning$best$decay
  
  cat("Best size:", best_size, "\n")
  cat("Best decay:", best_decay, "\n")
  cat("Inner log loss:", tuning$best$mean_logloss, "\n")
  
  # Store tuning results
  outer_tuning[[as.character(yr)]] <- tuning$results %>%
    mutate(Outer_Test_Year = yr)
  
  # 2. Refit selected configuration on all outer training data
  probability_list <- lapply(101:105, function(seed) {
    
    fit_nn(
      train = outer_train,
      test = outer_test,
      size = best_size,
      decay = best_decay,
      seed = seed
    )
  })
  
  # 3. Average probabilities across initialisations
  average_probabilities <- Reduce(
    "+",
    probability_list
  ) / length(probability_list)
  
  # 4. Save predictions for the held-out tournament
  predicted_classes <- classes[
    max.col(average_probabilities)
  ]
  
  outer_predictions[[as.character(yr)]] <- data.frame(
    Year = yr,
    Actual = outer_test$Result,
    Predicted = factor(
      predicted_classes,
      levels = classes
    ),
    Prob_Loss = average_probabilities[, "Loss"],
    Prob_Draw = average_probabilities[, "Draw"],
    Prob_Win = average_probabilities[, "Win"]
  )
}

# Combine all outer test predictions
predictions_nested <- bind_rows(outer_predictions)

# Combine tuning results
tuning_summary <- bind_rows(outer_tuning)

prob_matrix <- as.matrix(
  predictions_nested[, c(
    "Prob_Loss",
    "Prob_Draw",
    "Prob_Win"
  )]
)

colnames(prob_matrix) <- classes

final_metrics <- calculate_metrics(
  predictions_nested$Actual,
  prob_matrix
)

print(final_metrics)

table(
  Actual = predictions_nested$Actual,
  Predicted = predictions_nested$Predicted
)


library(ggplot2)
library(dplyr)
library(tidyr)

classes <- c("Loss", "Draw", "Win")

# Create confusion matrix
cm_direct <- table(
  Actual = factor(predictions_nested$Actual, levels = classes),
  Predicted = factor(predictions_nested$Predicted, levels = classes)
)

# Convert to percentages
cm_direct_df <- as.data.frame(cm_direct) %>%
  group_by(Actual) %>%
  mutate(
    Percentage = Freq / sum(Freq) * 100
  ) %>%
  ungroup()

# Plot
ggplot(cm_direct_df,
       aes(x = Predicted, y = Actual, fill = Percentage)) +
  geom_tile(colour = "white", linewidth = 1.5) +
  geom_text(
    aes(label = sprintf("%.1f%%\n(%d)", Percentage, Freq)),
    size = 5
  ) +
  scale_fill_gradient(
    low = "white",
    high = "steelblue",
    limits = c(0, 100)
  ) +
  labs(
    title = "Direct Classification NN",
    subtitle = "Nested CV outer test predictions",
    x = "Predicted outcome",
    y = "Actual outcome",
    fill = "Recall (%)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid = element_blank()
  )
year_metrics <- predictions_nested %>%
  group_by(Year) %>%
  group_modify(~ {
    
    p <- as.matrix(
      .x[, c("Prob_Loss", "Prob_Draw", "Prob_Win")]
    )
    
    colnames(p) <- classes
    
    as.data.frame(
      as.list(calculate_metrics(.x$Actual, p))
    )
  })

print(year_metrics)


baseline_predictions <- bind_rows(
  lapply(outer_years, function(yr) {
    
    train <- euro %>% filter(Year != yr)
    test  <- euro %>% filter(Year == yr)
    
    majority_class <- names(
      which.max(table(factor(train$Result, levels = classes)))
    )
    
    data.frame(
      Year = yr,
      Actual = test$Result,
      Predicted = majority_class
    )
  })
)

baseline_accuracy <- mean(
  baseline_predictions$Actual ==
    baseline_predictions$Predicted
)

baseline_accuracy


