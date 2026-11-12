library(dplyr)
library(tidyr)
library(nnet)
library(caret)

# Your original dataset
football_test = read_csv("C:/Users/962ni/Downloads/EuroTest.csv")
football_train = read_csv("C:/Users/962ni/Downloads/EuroTraining.csv")
eurodata=rbind(football_test,football_train)


data <-eurodata

# Predictors for each team's expected goals
features <- c(
  "GDP",
  "MarketValue",
  "FifaRank",
  "UefaPoints",
  "CLPlayers"
)

# Neural network tuning grid
tuning_grid <- expand.grid(
  size = c(1, 2, 3, 4, 5),
  decay = c(0.5, 0.7, 0.9, 1.0, 1.1),
  skip = c(FALSE, TRUE)
)

# Candidate draw thresholds
threshold_grid <- seq(0.10, 0.50, by = 0.025)

# Tournament years
years <- sort(unique(data$Year))

# Outcome classes
classes <- c("Loss", "Draw", "Win")

prepare_team_data <- function(data) {
  
  opponent_data <- data %>%
    select(
      Year,
      Team,
      all_of(features)
    ) %>%
    rename(
      Opponent = Team,
      GDP_opp = GDP,
      MarketValue_opp = MarketValue,
      FifaRank_opp = FifaRank,
      UefaPoints_opp = UefaPoints,
      CLPlayers_opp = CLPlayers
    )
  
  data_prepared <- data %>%
    left_join(
      opponent_data,
      by = c("Year", "Opponent")
    )
  
  return(data_prepared)
}

data <- prepare_team_data(data)

fit_goal_nn <- function(train, test,
                        size,
                        decay,
                        skip,
                        maxit = 1000) {
  
  # Predictor variables
  predictors <- c(
    "GroupStage",
    features,
    paste0(features, "_opp")
  )
  
  # Keep the factor levels consistent
  train$GroupStage <- factor(train$GroupStage)
  test$GroupStage <- factor(
    test$GroupStage,
    levels = levels(train$GroupStage)
  )
  
  # Create model matrices
  formula <- as.formula(
    paste("~", paste(predictors, collapse = " + "))
  )
  
  x_train <- model.matrix(formula, data = train)[, -1, drop = FALSE]
  x_test <- model.matrix(formula, data = test)[, -1, drop = FALSE]
  
  # Ensure the same columns exist in both matrices
  missing_cols <- setdiff(colnames(x_train), colnames(x_test))
  
  for (col in missing_cols) {
    x_test <- cbind(x_test, setNames(data.frame(rep(0, nrow(x_test))), col))
  }
  
  x_test <- as.matrix(x_test[, colnames(x_train), drop = FALSE])
  
  # Standardise using training data only
  train_means <- colMeans(x_train)
  train_sds <- apply(x_train, 2, sd)
  
  train_sds[is.na(train_sds) | train_sds == 0] <- 1
  
  x_train <- scale(
    x_train,
    center = train_means,
    scale = train_sds
  )
  
  x_test <- scale(
    x_test,
    center = train_means,
    scale = train_sds
  )
  
  # Transform response
  y_train <- log1p(train$Goals)
  
  # Fit neural network
  set.seed(123)
  
  model <- nnet(
    x = x_train,
    y = y_train,
    size = size,
    decay = decay,
    skip = skip,
    linout = TRUE,
    maxit = maxit,
    trace = FALSE
  )
  
  # Predict on training data for smearing correction
  train_pred <- as.vector(predict(model, x_train))
  
  residuals <- y_train - train_pred
  
  smearing <- mean(exp(residuals))
  
  # Predict test data
  test_pred_log <- as.vector(
    predict(model, x_test)
  )
  
  # Transform predictions back to goals
  test_pred <- exp(test_pred_log) * smearing - 1
  
  test_pred <- pmax(test_pred, 0)
  
  # Return predictions
  results <- test %>%
    mutate(
      Predicted_Goals = test_pred,
      Actual_Goals = Goals
    ) %>%
    select(
      Year,
      Team,
      Opponent,
      Actual_Goals,
      Predicted_Goals
    )
  
  return(results)
}


pair_team_predictions <- function(predictions) {
  
  matches <- predictions %>%
    mutate(
      Team = as.character(Team),
      Opponent = as.character(Opponent),
      Team_A = pmin(Team, Opponent),
      Team_B = pmax(Team, Opponent)
    ) %>%
    group_by(Year, Team_A, Team_B) %>%
    summarise(
      
      lambda_A = Predicted_Goals[Team == Team_A][1],
      lambda_B = Predicted_Goals[Team == Team_B][1],
      
      Goals_A = Actual_Goals[Team == Team_A][1],
      Goals_B = Actual_Goals[Team == Team_B][1],
      
      .groups = "drop"
    )
  
  return(matches)
}



predict_match_probabilities <- function(lambda_A,
                                        lambda_B,
                                        max_goals = 20) {
  
  goals <- 0:max_goals
  
  prob_A <- dpois(goals, lambda_A)
  prob_B <- dpois(goals, lambda_B)
  
  # Joint scoreline probability matrix
  score_matrix <- outer(prob_A, prob_B)
  
  # Normalise to account for the tiny truncated tail
  score_matrix <- score_matrix / sum(score_matrix)
  
  # Row index = goals scored by A
  # Column index = goals scored by B
  
  p_win <- sum(
    score_matrix[row(score_matrix) > col(score_matrix)]
  )
  
  p_draw <- sum(
    score_matrix[row(score_matrix) == col(score_matrix)]
  )
  
  p_loss <- sum(
    score_matrix[row(score_matrix) < col(score_matrix)]
  )
  
  return(
    c(
      Loss = p_loss,
      Draw = p_draw,
      Win = p_win
    )
  )
}

classify_match <- function(p_loss,
                           p_draw,
                           p_win,
                           draw_threshold) {
  
  if (p_draw >= draw_threshold) {
    
    return("Draw")
    
  } else if (p_win > p_loss) {
    
    return("Win")
    
  } else {
    
    return("Loss")
    
  }
}

calculate_macro_f1 <- function(actual, predicted) {
  
  actual <- factor(actual, levels = classes)
  predicted <- factor(predicted, levels = classes)
  
  cm <- table(
    factor(predicted, levels = classes),
    factor(actual, levels = classes)
  )
  
  f1_values <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    
    tp <- cm[i, i]
    
    fp <- sum(cm[i, ]) - tp
    fn <- sum(cm[, i]) - tp
    
    precision <- ifelse(
      tp + fp == 0,
      0,
      tp / (tp + fp)
    )
    
    recall <- ifelse(
      tp + fn == 0,
      0,
      tp / (tp + fn)
    )
    
    f1_values[i] <- ifelse(
      precision + recall == 0,
      0,
      2 * precision * recall / (precision + recall)
    )
  }
  
  return(mean(f1_values))
}


tune_goal_nn <- function(outer_train,
                         tuning_grid,
                         years) {
  
  results <- list()
  
  counter <- 1
  
  for (i in seq_len(nrow(tuning_grid))) {
    
    size_value <- tuning_grid$size[i]
    decay_value <- tuning_grid$decay[i]
    skip_value <- tuning_grid$skip[i]
    
    fold_rmse <- c()
    fold_mae <- c()
    
    for (valid_year in years) {
      
      inner_train <- outer_train %>%
        filter(Year != valid_year)
      
      inner_valid <- outer_train %>%
        filter(Year == valid_year)
      
      predictions <- fit_goal_nn(
        train = inner_train,
        test = inner_valid,
        size = size_value,
        decay = decay_value,
        skip = skip_value
      )
      
      rmse <- sqrt(
        mean(
          (predictions$Actual_Goals -
             predictions$Predicted_Goals)^2
        )
      )
      
      mae <- mean(
        abs(
          predictions$Actual_Goals -
            predictions$Predicted_Goals
        )
      )
      
      fold_rmse <- c(fold_rmse, rmse)
      fold_mae <- c(fold_mae, mae)
    }
    
    results[[counter]] <- data.frame(
      size = size_value,
      decay = decay_value,
      skip = skip_value,
      mean_RMSE = mean(fold_rmse),
      mean_MAE = mean(fold_mae)
    )
    
    counter <- counter + 1
  }
  
  results <- bind_rows(results)
  
  best_params <- results %>%
    arrange(mean_RMSE) %>%
    slice(1)
  
  return(
    list(
      best_params = best_params,
      all_results = results
    )
  )
}

tune_draw_threshold <- function(outer_train,
                                best_params,
                                years,
                                threshold_grid) {
  
  # Store predictions from inner validation folds
  inner_predictions <- list()
  
  counter <- 1
  
  for (valid_year in years) {
    
    inner_train <- outer_train %>%
      filter(Year != valid_year)
    
    inner_valid <- outer_train %>%
      filter(Year == valid_year)
    
    # Fit goal model using selected hyperparameters
    goal_predictions <- fit_goal_nn(
      train = inner_train,
      test = inner_valid,
      size = best_params$size,
      decay = best_params$decay,
      skip = best_params$skip
    )
    
    # Pair the two team predictions
    matches <- pair_team_predictions(goal_predictions)
    
    # Calculate actual match outcome
    matches <- matches %>%
      mutate(
        Actual = case_when(
          Goals_A > Goals_B ~ "Win",
          Goals_A == Goals_B ~ "Draw",
          TRUE ~ "Loss"
        )
      )
    
    # Calculate Poisson probabilities
    probabilities <- t(
      sapply(
        seq_len(nrow(matches)),
        function(i) {
          predict_match_probabilities(
            matches$lambda_A[i],
            matches$lambda_B[i]
          )
        }
      )
    )
    
    matches <- bind_cols(
      matches,
      as.data.frame(probabilities)
    )
    
    inner_predictions[[counter]] <- matches
    
    counter <- counter + 1
  }
  
  # Combine all inner validation predictions
  inner_predictions <- bind_rows(inner_predictions)
  
  # Evaluate every threshold
  threshold_results <- list()
  
  for (threshold in threshold_grid) {
    
    fold_scores <- c()
    
    for (valid_year in years) {
      
      year_data <- inner_predictions %>%
        filter(Year == valid_year)
      
      predicted <- vapply(
        seq_len(nrow(year_data)),
        function(i) {
          classify_match(
            p_loss = year_data$Loss[i],
            p_draw = year_data$Draw[i],
            p_win = year_data$Win[i],
            draw_threshold = threshold
          )
        },
        character(1)
      )
      
      fold_scores <- c(
        fold_scores,
        calculate_macro_f1(
          actual = year_data$Actual,
          predicted = predicted
        )
      )
    }
    
    threshold_results[[as.character(threshold)]] <- data.frame(
      threshold = threshold,
      mean_macro_F1 = mean(fold_scores)
    )
  }
  
  threshold_results <- bind_rows(threshold_results)
  
  best_threshold <- threshold_results %>%
    arrange(desc(mean_macro_F1), threshold) %>%
    slice(1)
  
  return(
    list(
      best_threshold = best_threshold$threshold,
      all_results = threshold_results,
      inner_predictions = inner_predictions
    )
  )
}



# Store all outer predictions
outer_predictions <- list()

# Store tuning results
outer_tuning_results <- list()

# Store threshold tuning results
outer_threshold_results <- list()


for (test_year in years) {
  
  cat("\n============================\n")
  cat("Outer test year:", test_year, "\n")
  cat("============================\n")
  
  # Outer split
  outer_train <- data %>%
    filter(Year != test_year)
  
  outer_test <- data %>%
    filter(Year == test_year)
  
  inner_years <- sort(unique(outer_train$Year))
  
  
  # --------------------------------
  # 1. Tune expected-goals NN
  # --------------------------------
  
  goal_tuning <- tune_goal_nn(
    outer_train = outer_train,
    tuning_grid = tuning_grid,
    years = inner_years
  )
  
  best_params <- goal_tuning$best_params
  
  cat("\nBest NN parameters:\n")
  print(best_params)
  
  
  # --------------------------------
  # 2. Tune draw threshold
  # --------------------------------
  
  threshold_tuning <- tune_draw_threshold(
    outer_train = outer_train,
    best_params = best_params,
    years = inner_years,
    threshold_grid = threshold_grid
  )
  
  best_threshold <- threshold_tuning$best_threshold
  
  cat("\nBest draw threshold:\n")
  print(best_threshold)
  
  
  # --------------------------------
  # 3. Fit final goal NN
  # --------------------------------
  
  final_goal_predictions <- fit_goal_nn(
    train = outer_train,
    test = outer_test,
    size = best_params$size,
    decay = best_params$decay,
    skip = best_params$skip
  )
  
  
  # --------------------------------
  # 4. Pair team predictions
  # --------------------------------
  
  match_predictions <- pair_team_predictions(
    final_goal_predictions
  )
  
  
  # --------------------------------
  # 5. Calculate outcome probabilities
  # --------------------------------
  
  probabilities <- t(
    sapply(
      seq_len(nrow(match_predictions)),
      function(i) {
        
        predict_match_probabilities(
          match_predictions$lambda_A[i],
          match_predictions$lambda_B[i]
        )
        
      }
    )
  )
  
  match_predictions <- bind_cols(
    match_predictions,
    as.data.frame(probabilities)
  )
  
  
  # --------------------------------
  # 6. Actual outcomes
  # --------------------------------
  
  match_predictions <- match_predictions %>%
    mutate(
      Actual = case_when(
        Goals_A > Goals_B ~ "Win",
        Goals_A == Goals_B ~ "Draw",
        TRUE ~ "Loss"
      )
    )
  
  
  # --------------------------------
  # 7. Apply selected draw threshold
  # --------------------------------
  
  match_predictions <- match_predictions %>%
    rowwise() %>%
    mutate(
      Predicted = classify_match(
        p_loss = Loss,
        p_draw = Draw,
        p_win = Win,
        draw_threshold = best_threshold
      )
    ) %>%
    ungroup()
  
  
  # Save outer predictions
  outer_predictions[[as.character(test_year)]] <-
    match_predictions %>%
    mutate(
      Best_Threshold = best_threshold,
      Best_Size = best_params$size,
      Best_Decay = best_params$decay,
      Best_Skip = best_params$skip
    )
  
  
  # Save tuning results
  outer_tuning_results[[as.character(test_year)]] <-
    goal_tuning$all_results %>%
    mutate(Outer_Test_Year = test_year)
  
  outer_threshold_results[[as.character(test_year)]] <-
    threshold_tuning$all_results %>%
    mutate(Outer_Test_Year = test_year)
  
  
  cat("\nOuter test predictions completed.\n")
}


# Combine all outer-fold predictions
predictions_nested_outcomes <- bind_rows(
  outer_predictions
)

# Combine tuning results
tuning_results_nested <- bind_rows(
  outer_tuning_results
)

threshold_results_nested <- bind_rows(
  outer_threshold_results
)


# Ensure consistent class levels
predictions_nested_outcomes$Actual <- factor(
  predictions_nested_outcomes$Actual,
  levels = classes
)

predictions_nested_outcomes$Predicted <- factor(
  predictions_nested_outcomes$Predicted,
  levels = classes
)

# Confusion matrix
confusionMatrix(
  data = predictions_nested_outcomes$Predicted,
  reference = predictions_nested_outcomes$Actual
)

overall_macro_f1 <- calculate_macro_f1(
  actual = predictions_nested_outcomes$Actual,
  predicted = predictions_nested_outcomes$Predicted
)

overall_macro_f1
predictions_nested_outcomes %>%
  count(Predicted) %>%
  mutate(
    Proportion = n / sum(n)
  )

library(ggplot2)

ggplot(
  threshold_results_nested,
  aes(
    x = threshold,
    y = mean_macro_F1,
    group = factor(Outer_Test_Year),
    colour = factor(Outer_Test_Year)
  )
) +
  geom_line() +
  geom_point() +
  labs(
    title = "Inner CV: Draw Threshold Tuning",
    x = "Draw probability threshold",
    y = "Mean inner CV Macro F1",
    colour = "Outer test year"
  ) +
  theme_minimal()

predictions_nested_outcomes %>%
  distinct(Year, Best_Threshold) %>%
  ggplot(
    aes(
      x = factor(Year),
      y = Best_Threshold
    )
  ) +
  geom_col() +
  labs(
    title = "Selected Draw Threshold by Outer Fold",
    x = "Outer test year",
    y = "Selected draw threshold"
  ) +
  theme_minimal()

# Create team-level goal evaluation data
goal_eval <- bind_rows(
  predictions_nested_outcomes %>%
    transmute(
      Year,
      Actual_Goals = Goals_A,
      Predicted_Goals = lambda_A
    ),
  
  predictions_nested_outcomes %>%
    transmute(
      Year,
      Actual_Goals = Goals_B,
      Predicted_Goals = lambda_B
    )
)

# MAE
goal_mae <- mean(
  abs(goal_eval$Actual_Goals - goal_eval$Predicted_Goals)
)

# RMSE
goal_rmse <- sqrt(
  mean(
    (goal_eval$Actual_Goals - goal_eval$Predicted_Goals)^2
  )
)

# Poisson Negative Log-Likelihood
goal_nll <- -mean(
  dpois(
    goal_eval$Actual_Goals,
    lambda = pmax(goal_eval$Predicted_Goals, 1e-10),
    log = TRUE
  )
)

# Results table
goal_metrics <- data.frame(
  Metric = c("MAE", "RMSE", "Poisson NLL"),
  Value = c(goal_mae, goal_rmse, goal_nll)
)

print(goal_metrics)

# Actual outcomes
actual <- factor(
  predictions_nested_outcomes$Actual,
  levels = classes
)

# Probability matrix
prob_matrix <- as.matrix(
  predictions_nested_outcomes[, classes]
)

storage.mode(prob_matrix) <- "double"

# Normalise probabilities
prob_matrix <- prob_matrix / rowSums(prob_matrix)

# Accuracy
accuracy <- mean(
  predictions_nested_outcomes$Actual ==
    predictions_nested_outcomes$Predicted,
  na.rm = TRUE
)

# One-hot encoded actual outcomes
actual_onehot <- sapply(
  classes,
  function(cl) as.numeric(actual == cl)
)

# Log Loss
actual_probability <- prob_matrix[
  cbind(seq_along(actual), as.integer(actual))
]

log_loss <- -mean(
  log(pmax(actual_probability, 1e-15))
)

# Brier Score
brier_score <- mean(
  rowSums((prob_matrix - actual_onehot)^2)
)

# Ranked Probability Score
predicted_cumulative <- cbind(
  prob_matrix[, "Loss"],
  prob_matrix[, "Loss"] + prob_matrix[, "Draw"]
)

actual_cumulative <- cbind(
  as.numeric(actual == "Loss"),
  as.numeric(actual %in% c("Loss", "Draw"))
)

rps <- mean(
  rowMeans(
    (predicted_cumulative - actual_cumulative)^2
  )
)

# Results table
outcome_metrics <- data.frame(
  Metric = c(
    "Accuracy",
    "Macro F1",
    "Log Loss",
    "Brier Score",
    "RPS"
  ),
  
  Value = c(
    accuracy,
    overall_macro_f1,
    log_loss,
    brier_score,
    rps
  )
)

print(outcome_metrics)
1