
# ============================================================
# NESTED LEAVE-ONE-TOURNAMENT-OUT CROSS-VALIDATION
# RANDOM FOREST + SKELLAM + THRESHOLD CLASSIFICATION
# ============================================================


# ============================================================
# A. SETUP AND HELPER FUNCTIONS
# ============================================================

library(ranger)
library(dplyr)
library(ggplot2)
library(tidyr)
library(skellam)


# ------------------------------------------------------------
# A1. Settings
# ------------------------------------------------------------

N_TREES <- 1000
BASE_SEED <- 6769

target <- "Goals"

predictors <- c(
  "GroupStage",
  "GDP",
  "MarketValue",
  "FifaRank",
  "UefaPoints",
  "CLPlayers"
)

required_columns <- c(
  target,
  predictors,
  "match_id",
  "Team",
  "Opponent"
)

rf_formula <- as.formula(
  paste(target, "~", paste(predictors, collapse = " + "))
)

outcome_levels <- c("Win", "Draw", "Loss")

threshold_grid <- c(0.05, 0.10, 0.15, 0.20, 0.25)


# ------------------------------------------------------------
# A2. Combine the six tournament datasets
# ------------------------------------------------------------

euro_data <- list(
  "2004" = euro_2004,
  "2008" = euro_2008,
  "2012" = euro_2012,
  "2016" = euro_2016,
  "2020" = euro_2020,
  "2024" = euro_2024_fixed
)

EuroAll <- bind_rows(euro_data, .id = "Tournament")

EuroAll$Observation_ID <- seq_len(nrow(EuroAll))

tournament_years <- names(euro_data)

# Check data
print(table(EuroAll$Tournament))

stopifnot(
  all(required_columns %in% names(EuroAll)),
  !anyNA(EuroAll[, required_columns]),
  all(EuroAll$Goals >= 0),
  all(EuroAll$Goals == floor(EuroAll$Goals))
)

# Check that every tournament has two observations per match
match_counts <- EuroAll %>%
  count(Tournament, match_id)

stopifnot(all(match_counts$n == 2))


# ------------------------------------------------------------
# A3. Regression evaluation metrics
# ------------------------------------------------------------

calculate_metrics <- function(actual, predicted) {
  
  lambda <- pmax(predicted, 1e-10)
  
  data.frame(
    MAE = mean(abs(actual - predicted)),
    
    RMSE = sqrt(mean((actual - predicted)^2)),
    
    Poisson_NLL = -mean(
      dpois(
        x = actual,
        lambda = lambda,
        log = TRUE
      )
    )
  )
}


# ------------------------------------------------------------
# A4. Construct match-level data and Skellam probabilities
# ------------------------------------------------------------

build_match_data <- function(prediction_data) {
  
  match_data <- prediction_data %>%
    
    mutate(
      Team = as.character(Team),
      Opponent = as.character(Opponent),
      
      Team1_name = pmin(Team, Opponent),
      Team2_name = pmax(Team, Opponent),
      
      Is_Team1 = Team == Team1_name
    ) %>%
    
    group_by(Tournament, match_id) %>%
    
    summarise(
      
      N_Teams = n(),
      N_Team1 = sum(Is_Team1),
      N_Team2 = sum(!Is_Team1),
      
      Team1 = first(Team1_name),
      Team2 = first(Team2_name),
      
      Goals1 = first(Goals[Is_Team1]),
      Goals2 = first(Goals[!Is_Team1]),
      
      Lambda1 = first(Predicted_Goals[Is_Team1]),
      Lambda2 = first(Predicted_Goals[!Is_Team1]),
      
      .groups = "drop"
    )
  
  # Every match must contain exactly two team observations
  stopifnot(
    all(match_data$N_Teams == 2),
    all(match_data$N_Team1 == 1),
    all(match_data$N_Team2 == 1)
  )
  
  # Ensure positive Poisson rates
  match_data <- match_data %>%
    
    mutate(
      Lambda1 = pmax(Lambda1, 1e-10),
      Lambda2 = pmax(Lambda2, 1e-10)
    )
  
  # Calculate Skellam probabilities
  match_data <- match_data %>%
    
    mutate(
      
      # Team 1 wins: P(D > 0)
      P_Win = skellam::pskellam(
        0, Lambda1, Lambda2,
        lower.tail = FALSE
      ),
      
      # Draw: P(D = 0)
      P_Draw = skellam::dskellam(
        0, Lambda1, Lambda2
      ),
      
      # Team 1 loses: P(D < 0)
      P_Loss = skellam::pskellam(
        -1, Lambda1, Lambda2,
        lower.tail = TRUE
      ),
      
      Probability_Sum = P_Win + P_Draw + P_Loss,
      
      # Actual observed outcome
      Actual_Outcome = case_when(
        Goals1 > Goals2 ~ "Win",
        Goals1 == Goals2 ~ "Draw",
        Goals1 < Goals2 ~ "Loss"
      ),
      
      Actual_Outcome = factor(
        Actual_Outcome,
        levels = outcome_levels
      ),
      
      # Original maximum-probability classification
      Argmax_Outcome = outcome_levels[
        max.col(
          cbind(P_Win, P_Draw, P_Loss),
          ties.method = "first"
        )
      ],
      
      Argmax_Outcome = factor(
        Argmax_Outcome,
        levels = outcome_levels
      ),
      
      # Difference between maximum and minimum probabilities
      Probability_Range =
        pmax(P_Win, P_Draw, P_Loss) -
        pmin(P_Win, P_Draw, P_Loss)
    )
  
  # Probability checks
  stopifnot(
    all(match_data$P_Win >= 0 & match_data$P_Win <= 1),
    all(match_data$P_Draw >= 0 & match_data$P_Draw <= 1),
    all(match_data$P_Loss >= 0 & match_data$P_Loss <= 1),
    all(abs(match_data$Probability_Sum - 1) < 1e-6)
  )
  
  return(match_data)
}


# ------------------------------------------------------------
# A5. Apply threshold-based classification
# ------------------------------------------------------------

apply_threshold <- function(match_data, threshold) {
  
  match_data %>%
    
    mutate(
      Predicted_Outcome = ifelse(
        Probability_Range < threshold,
        "Draw",
        as.character(Argmax_Outcome)
      ),
      
      Predicted_Outcome = factor(
        Predicted_Outcome,
        levels = outcome_levels
      )
    )
}


# ------------------------------------------------------------
# A6. Classification evaluation metrics
# ------------------------------------------------------------

calculate_outcome_metrics <- function(match_data) {
  
  prob_matrix <- as.matrix(
    match_data[, c("P_Win", "P_Draw", "P_Loss")]
  )
  
  colnames(prob_matrix) <- outcome_levels
  
  actual_index <- match(
    as.character(match_data$Actual_Outcome),
    outcome_levels
  )
  
  # Probability assigned to the actual outcome
  true_probabilities <- prob_matrix[
    cbind(seq_len(nrow(prob_matrix)), actual_index)
  ]
  
  # Log Loss
  log_loss <- -mean(
    log(pmax(true_probabilities, 1e-15))
  )
  
  # One-hot encoded actual outcomes
  actual_matrix <- matrix(
    0,
    nrow = nrow(prob_matrix),
    ncol = 3
  )
  
  actual_matrix[
    cbind(seq_len(nrow(prob_matrix)), actual_index)
  ] <- 1
  
  # Multiclass Brier Score
  brier_score <- mean(
    rowSums((prob_matrix - actual_matrix)^2)
  )
  
  # Classification accuracy
  accuracy <- mean(
    match_data$Predicted_Outcome ==
      match_data$Actual_Outcome
  )
  
  data.frame(
    Log_Loss = log_loss,
    Brier_Score = brier_score,
    Accuracy = accuracy
  )
}


# ------------------------------------------------------------
# A7. Select threshold using inner CV predictions
# ------------------------------------------------------------

select_threshold <- function(inner_match_data, threshold_grid) {
  
  threshold_results <- lapply(
    threshold_grid,
    function(threshold) {
      
      predictions <- ifelse(
        inner_match_data$Probability_Range < threshold,
        "Draw",
        as.character(inner_match_data$Argmax_Outcome)
      )
      
      temp <- data.frame(
        Tournament = inner_match_data$Tournament,
        Correct = predictions ==
          as.character(inner_match_data$Actual_Outcome)
      )
      
      # Accuracy separately for each tournament
      tournament_accuracy <- temp %>%
        group_by(Tournament) %>%
        summarise(
          Accuracy = mean(Correct),
          .groups = "drop"
        )
      
      data.frame(
        Threshold = threshold,
        
        Mean_Tournament_Accuracy =
          mean(tournament_accuracy$Accuracy),
        
        Pooled_Accuracy = mean(temp$Correct)
      )
    }
  )
  
  threshold_results <- bind_rows(threshold_results)
  
  # Select threshold maximising mean tournament accuracy
  # which.max selects the first threshold in case of a tie
  best_threshold <- threshold_results$Threshold[
    which.max(threshold_results$Mean_Tournament_Accuracy)
  ]
  
  list(
    results = threshold_results,
    best_threshold = best_threshold
  )
}


# ------------------------------------------------------------
# A8. Construct actual match outcomes (for baseline)
# ------------------------------------------------------------

build_actual_matches <- function(data) {
  
  actual_matches <- data %>%
    
    mutate(
      Team = as.character(Team),
      Opponent = as.character(Opponent),
      
      Team1_name = pmin(Team, Opponent),
      Is_Team1 = Team == Team1_name
    ) %>%
    
    group_by(Tournament, match_id) %>%
    
    summarise(
      N_Teams = n(),
      N_Team1 = sum(Is_Team1),
      
      Goals1 = first(Goals[Is_Team1]),
      Goals2 = first(Goals[!Is_Team1]),
      
      .groups = "drop"
    ) %>%
    
    mutate(
      Actual_Outcome = case_when(
        Goals1 > Goals2 ~ "Win",
        Goals1 == Goals2 ~ "Draw",
        Goals1 < Goals2 ~ "Loss"
      ),
      
      Actual_Outcome = factor(
        Actual_Outcome,
        levels = outcome_levels
      )
    )
  
  stopifnot(
    all(actual_matches$N_Teams == 2),
    all(actual_matches$N_Team1 == 1)
  )
  
  actual_matches
}


# ------------------------------------------------------------
# A9. Original RF hyperparameter grid
# ------------------------------------------------------------

rf_grid <- expand.grid(
  mtry = 1:length(predictors),
  min.node.size = c(3, 5, 10, 15),
  sample.fraction = c(0.7, 0.85, 1.0)
)

stopifnot(nrow(rf_grid) == 72)

print(rf_grid)



# ============================================================
# B. NESTED CROSS-VALIDATION
# ============================================================

# Store outer-fold results
outer_results <- vector("list", length(tournament_years))

# Store team-level predictions
outer_predictions <- vector("list", length(tournament_years))

# Store match-level predictions
outer_match_predictions <- vector("list", length(tournament_years))

# Store selected RF parameters and thresholds
selected_parameters <- vector("list", length(tournament_years))

# Store inner RF tuning results
all_inner_results <- vector("list", length(tournament_years))

# Store threshold tuning results
all_threshold_results <- vector("list", length(tournament_years))


# ------------------------------------------------------------
# Outer loop: hold out one entire tournament
# ------------------------------------------------------------

for (outer_i in seq_along(tournament_years)) {
  
  outer_year <- tournament_years[outer_i]
  
  cat("\n========================================\n")
  cat("OUTER FOLD:", outer_year, "\n")
  cat("========================================\n")
  
  # Outer training and test data
  outer_train <- EuroAll %>%
    filter(Tournament != outer_year)
  
  outer_test <- EuroAll %>%
    filter(Tournament == outer_year)
  
  # Five tournaments available for inner CV
  inner_years <- tournament_years[
    tournament_years != outer_year
  ]
  
  inner_results <- vector("list", nrow(rf_grid))
  
  
  # ==========================================================
  # B1. INNER CV: RF HYPERPARAMETER TUNING
  # ==========================================================
  
  for (grid_i in seq_len(nrow(rf_grid))) {
    
    params <- rf_grid[grid_i, ]
    
    fold_results <- vector("list", length(inner_years))
    
    for (inner_j in seq_along(inner_years)) {
      
      inner_year <- inner_years[inner_j]
      
      # Inner training: four tournaments
      inner_train <- outer_train %>%
        filter(Tournament != inner_year)
      
      # Inner validation: one tournament
      inner_validation <- outer_train %>%
        filter(Tournament == inner_year)
      
      model_seed <- BASE_SEED +
        outer_i * 100000 +
        grid_i * 10 +
        inner_j
      
      rf_model <- ranger(
        formula = rf_formula,
        data = inner_train[, c(target, predictors)],
        num.trees = N_TREES,
        mtry = params$mtry,
        min.node.size = params$min.node.size,
        sample.fraction = params$sample.fraction,
        importance = "none",
        seed = model_seed
      )
      
      # Predict inner validation tournament
      predictions <- predict(
        rf_model,
        data = inner_validation[, predictors]
      )$predictions
      
      metrics <- calculate_metrics(
        inner_validation$Goals,
        predictions
      )
      
      fold_results[[inner_j]] <- data.frame(
        Tournament = inner_year,
        MAE = metrics$MAE,
        RMSE = metrics$RMSE,
        Poisson_NLL = metrics$Poisson_NLL
      )
    }
    
    fold_results_df <- bind_rows(fold_results)
    
    # Equal weight per tournament
    inner_results[[grid_i]] <- data.frame(
      mtry = params$mtry,
      min.node.size = params$min.node.size,
      sample.fraction = params$sample.fraction,
      
      Mean_MAE = mean(fold_results_df$MAE),
      Mean_RMSE = mean(fold_results_df$RMSE),
      Mean_Poisson_NLL = mean(fold_results_df$Poisson_NLL),
      
      SD_MAE = sd(fold_results_df$MAE)
    )
  }
  
  
  # ----------------------------------------------------------
  # B2. Select best RF hyperparameters using inner CV MAE
  # ----------------------------------------------------------
  
  inner_results_df <- bind_rows(inner_results) %>%
    arrange(Mean_MAE)
  
  best_params <- inner_results_df[1, ]
  
  all_inner_results[[outer_i]] <- inner_results_df
  
  cat("\nSelected RF hyperparameters:\n")
  print(best_params)
  
  
  # ==========================================================
  # B3. INNER CV: GENERATE OOF PREDICTIONS USING BEST RF
  # ==========================================================
  
  inner_oof_predictions <- vector("list", length(inner_years))
  
  for (inner_j in seq_along(inner_years)) {
    
    inner_year <- inner_years[inner_j]
    
    inner_train <- outer_train %>%
      filter(Tournament != inner_year)
    
    inner_validation <- outer_train %>%
      filter(Tournament == inner_year)
    
    inner_seed <- BASE_SEED +
      outer_i * 100000 +
      90000 +
      inner_j
    
    rf_inner_best <- ranger(
      formula = rf_formula,
      data = inner_train[, c(target, predictors)],
      num.trees = N_TREES,
      mtry = best_params$mtry,
      min.node.size = best_params$min.node.size,
      sample.fraction = best_params$sample.fraction,
      importance = "none",
      seed = inner_seed
    )
    
    inner_predictions <- predict(
      rf_inner_best,
      data = inner_validation[, predictors]
    )$predictions
    
    # Preserve match information
    inner_oof_predictions[[inner_j]] <- data.frame(
      Tournament = inner_validation$Tournament,
      match_id = inner_validation$match_id,
      Team = as.character(inner_validation$Team),
      Opponent = as.character(inner_validation$Opponent),
      Goals = inner_validation$Goals,
      Predicted_Goals = inner_predictions
    )
  }
  
  inner_oof_predictions <- bind_rows(inner_oof_predictions)
  
  # Convert team-level predictions into match-level probabilities
  inner_match_data <- build_match_data(inner_oof_predictions)
  
  cat("\nInner OOF matches by tournament:\n")
  print(table(inner_match_data$Tournament))
  
  
  # ==========================================================
  # B4. INNER CV: THRESHOLD SELECTION
  # ==========================================================
  
  threshold_selection <- select_threshold(
    inner_match_data,
    threshold_grid
  )
  
  threshold_results <- threshold_selection$results
  
  best_threshold <- threshold_selection$best_threshold
  
  all_threshold_results[[outer_i]] <- threshold_results
  
  cat("\nThreshold tuning results:\n")
  print(threshold_results)
  
  cat("\nSelected threshold:", best_threshold, "\n")
  
  # Apply selected threshold to inner predictions
  inner_match_data <- apply_threshold(
    inner_match_data,
    best_threshold
  )
  
  inner_classification_metrics <- calculate_outcome_metrics(
    inner_match_data
  )
  
  cat("\nInner classification performance:\n")
  print(inner_classification_metrics)
  
  
  # ----------------------------------------------------------
  # B5. Save selected RF parameters and threshold
  # ----------------------------------------------------------
  
  selected_parameters[[outer_i]] <- data.frame(
    Outer_Tournament = outer_year,
    
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    
    Inner_MAE = best_params$Mean_MAE,
    
    Best_Threshold = best_threshold,
    
    Inner_Classification_Accuracy =
      inner_classification_metrics$Accuracy
  )
  
  
  # ==========================================================
  # B6. REFIT RF ON ALL FIVE OUTER-TRAINING TOURNAMENTS
  # ==========================================================
  
  final_seed <- BASE_SEED + outer_i * 1000000
  
  rf_outer <- ranger(
    formula = rf_formula,
    data = outer_train[, c(target, predictors)],
    num.trees = N_TREES,
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    importance = "none",
    seed = final_seed
  )
  
  
  # ==========================================================
  # B7. PREDICT THE UNTOUCHED OUTER TEST TOURNAMENT
  # ==========================================================
  
  outer_pred <- predict(
    rf_outer,
    data = outer_test[, predictors]
  )$predictions
  
  # Regression metrics
  outer_metrics <- calculate_metrics(
    outer_test$Goals,
    outer_pred
  )
  
  # Save team-level predictions
  outer_predictions[[outer_i]] <- data.frame(
    Observation_ID = outer_test$Observation_ID,
    Tournament = outer_year,
    match_id = outer_test$match_id,
    Team = as.character(outer_test$Team),
    Opponent = as.character(outer_test$Opponent),
    Actual_Goals = outer_test$Goals,
    Predicted_Goals = outer_pred
  )
  
  
  # ==========================================================
  # B8. OUTER TEST: SKELLAM PROBABILITIES
  # ==========================================================
  
  outer_prediction_rows <- data.frame(
    Tournament = outer_test$Tournament,
    match_id = outer_test$match_id,
    Team = as.character(outer_test$Team),
    Opponent = as.character(outer_test$Opponent),
    Goals = outer_test$Goals,
    Predicted_Goals = outer_pred
  )
  
  outer_match_data <- build_match_data(
    outer_prediction_rows
  )
  
  
  # ==========================================================
  # B9. OUTER TEST: APPLY INNER-SELECTED THRESHOLD
  # ==========================================================
  
  outer_match_data <- apply_threshold(
    outer_match_data,
    best_threshold
  )
  
  
  # ==========================================================
  # B10. HISTORICAL FREQUENCY BASELINE
  # ==========================================================
  
  # Derive actual match outcomes from the five training tournaments
  training_match_outcomes <- build_actual_matches(outer_train)
  
  # Historical outcome frequencies
  baseline_probs <- prop.table(
    table(
      factor(
        as.character(training_match_outcomes$Actual_Outcome),
        levels = outcome_levels
      )
    )
  )
  
  # Assign probabilities to every match in the outer test tournament
  outer_match_data <- outer_match_data %>%
    
    mutate(
      Baseline_P_Win = unname(baseline_probs["Win"]),
      Baseline_P_Draw = unname(baseline_probs["Draw"]),
      Baseline_P_Loss = unname(baseline_probs["Loss"])
    )
  
  # Historical baseline classification
  baseline_matrix <- as.matrix(
    outer_match_data[
      , c(
        "Baseline_P_Win",
        "Baseline_P_Draw",
        "Baseline_P_Loss"
      )
    ]
  )
  
  outer_match_data$Baseline_Predicted <- factor(
    outcome_levels[
      max.col(baseline_matrix, ties.method = "first")
    ],
    levels = outcome_levels
  )
  
  
  # ==========================================================
  # B11. CLASSIFICATION METRICS
  # ==========================================================
  
  # RF + Skellam + threshold
  outer_classification_metrics <- calculate_outcome_metrics(
    outer_match_data
  )
  
  # Historical baseline probabilities
  baseline_evaluation <- outer_match_data %>%
    
    mutate(
      P_Win = Baseline_P_Win,
      P_Draw = Baseline_P_Draw,
      P_Loss = Baseline_P_Loss,
      
      Predicted_Outcome = Baseline_Predicted
    )
  
  baseline_classification_metrics <- calculate_outcome_metrics(
    baseline_evaluation
  )
  
  
  # ==========================================================
  # B12. SAVE OUTER-FOLD MATCH PREDICTIONS
  # ==========================================================
  
  outer_match_predictions[[outer_i]] <- outer_match_data
  
  
  # ==========================================================
  # B13. SAVE OUTER-FOLD RESULTS
  # ==========================================================
  
  outer_results[[outer_i]] <- data.frame(
    
    Outer_Tournament = outer_year,
    
    Train_Observations = nrow(outer_train),
    Test_Observations = nrow(outer_test),
    
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    
    Best_Threshold = best_threshold,
    
    # Regression metrics
    Outer_MAE = outer_metrics$MAE,
    Outer_RMSE = outer_metrics$RMSE,
    Outer_Poisson_NLL = outer_metrics$Poisson_NLL,
    
    # RF classification metrics
    Outer_Log_Loss = outer_classification_metrics$Log_Loss,
    Outer_Brier_Score = outer_classification_metrics$Brier_Score,
    Outer_Accuracy = outer_classification_metrics$Accuracy,
    
    # Historical baseline metrics
    Baseline_Log_Loss = baseline_classification_metrics$Log_Loss,
    Baseline_Brier_Score = baseline_classification_metrics$Brier_Score,
    Baseline_Accuracy = baseline_classification_metrics$Accuracy
  )
  
  
  cat("\nOuter test performance:\n")
  print(outer_results[[outer_i]])
  
  cat("\nCompleted outer fold", outer_i,
      "of", length(tournament_years), "\n")
}


# ============================================================
# C. COMBINE RESULTS FROM ALL SIX OUTER FOLDS
# ============================================================

outer_results <- bind_rows(outer_results)

outer_predictions <- bind_rows(outer_predictions)

outer_match_predictions <- bind_rows(outer_match_predictions)

selected_parameters <- bind_rows(selected_parameters)

cat("\nNESTED CV COMPLETED\n")


# ============================================================
# D. REGRESSION PERFORMANCE
# ============================================================

# ------------------------------------------------------------
# D1. Performance per outer tournament
# ------------------------------------------------------------

cat("\nOuter-fold regression performance:\n")

print(
  outer_results[
    , c(
      "Outer_Tournament",
      "Outer_MAE",
      "Outer_RMSE",
      "Outer_Poisson_NLL"
    )
  ],
  row.names = FALSE
)

mean_outer_MAE <- mean(outer_results$Outer_MAE)

mean_outer_RMSE <- mean(outer_results$Outer_RMSE)

mean_outer_Poisson_NLL <- mean(
  outer_results$Outer_Poisson_NLL
)

sd_outer_MAE <- sd(outer_results$Outer_MAE)


# ------------------------------------------------------------
# D2. Pooled regression performance
# ------------------------------------------------------------

stopifnot(
  nrow(outer_predictions) == nrow(EuroAll),
  !anyDuplicated(outer_predictions$Observation_ID),
  setequal(
    outer_predictions$Observation_ID,
    EuroAll$Observation_ID
  ),
  !anyNA(outer_predictions$Predicted_Goals)
)

pooled_metrics <- calculate_metrics(
  outer_predictions$Actual_Goals,
  outer_predictions$Predicted_Goals
)

cat("\nPooled nested CV regression performance:\n")
print(pooled_metrics)


# ============================================================
# E. CLASSIFICATION PERFORMANCE
# ============================================================

# ------------------------------------------------------------
# E1. Validate match-level predictions
# ------------------------------------------------------------

expected_matches <- nrow(EuroAll) / 2

stopifnot(
  nrow(outer_match_predictions) == expected_matches,
  
  !anyDuplicated(
    paste(
      outer_match_predictions$Tournament,
      outer_match_predictions$match_id
    )
  ),
  
  all(outer_match_predictions$N_Teams == 2)
)

cat("\nMatches per outer tournament:\n")
print(table(outer_match_predictions$Tournament))


# ------------------------------------------------------------
# E2. Pooled RF classification metrics
# ------------------------------------------------------------

pooled_classification_metrics <- calculate_outcome_metrics(
  outer_match_predictions
)

cat("\nPooled nested CV RF classification performance:\n")
print(pooled_classification_metrics)


# ------------------------------------------------------------
# E3. Pooled historical baseline metrics
# ------------------------------------------------------------

pooled_baseline_evaluation <- outer_match_predictions %>%
  
  mutate(
    P_Win = Baseline_P_Win,
    P_Draw = Baseline_P_Draw,
    P_Loss = Baseline_P_Loss,
    
    Predicted_Outcome = Baseline_Predicted
  )

pooled_baseline_metrics <- calculate_outcome_metrics(
  pooled_baseline_evaluation
)

cat("\nPooled historical baseline performance:\n")
print(pooled_baseline_metrics)


# ------------------------------------------------------------
# E4. Mean outer-fold classification performance
# ------------------------------------------------------------

mean_outer_Log_Loss <- mean(outer_results$Outer_Log_Loss)

mean_outer_Brier <- mean(outer_results$Outer_Brier_Score)

mean_outer_Accuracy <- mean(outer_results$Outer_Accuracy)

sd_outer_Accuracy <- sd(outer_results$Outer_Accuracy)


# ------------------------------------------------------------
# E5. Model comparison
# ------------------------------------------------------------

model_comparison <- data.frame(
  
  Model = c(
    "Random Forest + Skellam + Threshold",
    "Historical Frequency Baseline"
  ),
  
  Pooled_Log_Loss = c(
    pooled_classification_metrics$Log_Loss,
    pooled_baseline_metrics$Log_Loss
  ),
  
  Pooled_Brier_Score = c(
    pooled_classification_metrics$Brier_Score,
    pooled_baseline_metrics$Brier_Score
  ),
  
  Pooled_Accuracy = c(
    pooled_classification_metrics$Accuracy,
    pooled_baseline_metrics$Accuracy
  )
)

cat("\nNested CV model comparison:\n")
print(model_comparison, row.names = FALSE)


# ------------------------------------------------------------
# E6. Final combined summary
# ------------------------------------------------------------

final_nested_summary <- data.frame(
  
  Metric = c(
    "Pooled MAE",
    "Pooled RMSE",
    "Pooled Poisson NLL",
    
    "Mean tournament MAE",
    "Mean tournament RMSE",
    "Mean tournament Poisson NLL",
    "SD of tournament MAE",
    
    "Pooled Log Loss",
    "Pooled Brier Score",
    "Pooled Classification Accuracy",
    
    "Mean tournament Log Loss",
    "Mean tournament Brier Score",
    "Mean tournament Accuracy",
    "SD of tournament Accuracy",
    
    "Pooled Baseline Log Loss",
    "Pooled Baseline Brier Score",
    "Pooled Baseline Accuracy"
  ),
  
  Value = c(
    pooled_metrics$MAE,
    pooled_metrics$RMSE,
    pooled_metrics$Poisson_NLL,
    
    mean_outer_MAE,
    mean_outer_RMSE,
    mean_outer_Poisson_NLL,
    sd_outer_MAE,
    
    pooled_classification_metrics$Log_Loss,
    pooled_classification_metrics$Brier_Score,
    pooled_classification_metrics$Accuracy,
    
    mean_outer_Log_Loss,
    mean_outer_Brier,
    mean_outer_Accuracy,
    sd_outer_Accuracy,
    
    pooled_baseline_metrics$Log_Loss,
    pooled_baseline_metrics$Brier_Score,
    pooled_baseline_metrics$Accuracy
  )
)

cat("\nFINAL NESTED CV SUMMARY:\n")
print(final_nested_summary, row.names = FALSE)


# ============================================================
# F. SELECTED HYPERPARAMETERS AND THRESHOLDS
# ============================================================

# ------------------------------------------------------------
# F1. Selected parameters per outer fold
# ------------------------------------------------------------

cat("\nSelected RF parameters and thresholds:\n")

print(selected_parameters, row.names = FALSE)


# ------------------------------------------------------------
# F2. Frequency of selected RF parameters
# ------------------------------------------------------------

cat("\nSelected mtry values:\n")
print(table(selected_parameters$mtry))

cat("\nSelected minimum node sizes:\n")
print(table(selected_parameters$min.node.size))

cat("\nSelected sample fractions:\n")
print(table(selected_parameters$sample.fraction))


# ------------------------------------------------------------
# F3. Frequency of selected thresholds
# ------------------------------------------------------------

cat("\nSelected thresholds:\n")
print(table(selected_parameters$Best_Threshold))


# ============================================================
# G. REGRESSION VISUALISATIONS
# ============================================================

# ------------------------------------------------------------
# G1. MAE by outer validation tournament
# ------------------------------------------------------------

ggplot(
  outer_results,
  aes(x = Outer_Tournament, y = Outer_MAE)
) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Random Forest: Nested CV Performance",
    subtitle = "MAE on each completely held-out tournament",
    x = "Outer validation tournament",
    y = "MAE"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# G2. RMSE by outer validation tournament
# ------------------------------------------------------------

ggplot(
  outer_results,
  aes(x = Outer_Tournament, y = Outer_RMSE)
) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Random Forest: Nested CV Performance",
    subtitle = "RMSE on each completely held-out tournament",
    x = "Outer validation tournament",
    y = "RMSE"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# G3. Poisson NLL by outer validation tournament
# ------------------------------------------------------------

ggplot(
  outer_results,
  aes(x = Outer_Tournament, y = Outer_Poisson_NLL)
) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Random Forest: Nested CV Performance",
    subtitle = "Poisson NLL on each completely held-out tournament",
    x = "Outer validation tournament",
    y = "Poisson NLL"
  ) +
  theme_minimal()


# ============================================================
# H. CLASSIFICATION VISUALISATIONS
# ============================================================

# ------------------------------------------------------------
# H1. Classification accuracy by outer tournament
# ------------------------------------------------------------

ggplot(
  outer_results,
  aes(x = Outer_Tournament, y = Outer_Accuracy)
) +
  geom_col(fill = "steelblue") +
  geom_hline(
    yintercept = 1/3,
    linetype = "dashed",
    color = "red"
  ) +
  labs(
    title = "Match Outcome Classification Accuracy",
    subtitle = "Nested leave-one-tournament-out CV",
    x = "Held-out tournament",
    y = "Accuracy"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# H2. Pooled confusion matrix
# ------------------------------------------------------------

confusion_matrix <- table(
  Actual = outer_match_predictions$Actual_Outcome,
  Predicted = outer_match_predictions$Predicted_Outcome
)

cat("\nPooled outer-fold confusion matrix:\n")
print(confusion_matrix)

confusion_df <- as.data.frame(confusion_matrix)

ggplot(
  confusion_df,
  aes(x = Predicted, y = Actual, fill = Freq)
) +
  geom_tile(color = "white") +
  geom_text(
    aes(label = Freq),
    size = 5
  ) +
  scale_y_discrete(limits = rev(outcome_levels)) +
  labs(
    title = "Confusion Matrix: Nested CV",
    subtitle = "Predictions on all six held-out tournaments",
    x = "Predicted outcome",
    y = "Actual outcome",
    fill = "Matches"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# H3. Distribution of predicted probabilities
# ------------------------------------------------------------

probability_long <- outer_match_predictions %>%
  
  select(P_Win, P_Draw, P_Loss) %>%
  
  pivot_longer(
    cols = everything(),
    names_to = "Outcome",
    values_to = "Probability"
  ) %>%
  
  mutate(
    Outcome = recode(
      Outcome,
      P_Win = "Win",
      P_Draw = "Draw",
      P_Loss = "Loss"
    ),
    
    Outcome = factor(
      Outcome,
      levels = outcome_levels
    )
  )

ggplot(
  probability_long,
  aes(x = Probability, fill = Outcome)
) +
  geom_histogram(
    bins = 25,
    alpha = 0.75,
    color = "white"
  ) +
  facet_wrap(~Outcome, ncol = 1) +
  labs(
    title = "Distribution of Predicted Match Outcome Probabilities",
    subtitle = "Nested CV outer-fold predictions",
    x = "Predicted probability",
    y = "Number of matches"
  ) +
  theme_minimal() +
  theme(legend.position = "none")


# ------------------------------------------------------------
# H4. Predicted probabilities by actual outcome
# ------------------------------------------------------------

probability_by_actual <- outer_match_predictions %>%
  
  select(
    Actual_Outcome,
    P_Win,
    P_Draw,
    P_Loss
  ) %>%
  
  pivot_longer(
    cols = c(P_Win, P_Draw, P_Loss),
    names_to = "Probability_Type",
    values_to = "Probability"
  ) %>%
  
  mutate(
    Probability_Type = recode(
      Probability_Type,
      P_Win = "Win probability",
      P_Draw = "Draw probability",
      P_Loss = "Loss probability"
    )
  )

ggplot(
  probability_by_actual,
  aes(
    x = Actual_Outcome,
    y = Probability,
    fill = Actual_Outcome
  )
) +
  geom_boxplot(alpha = 0.75) +
  facet_wrap(~Probability_Type) +
  labs(
    title = "Predicted Probabilities by Actual Match Outcome",
    subtitle = "Nested CV outer-fold predictions",
    x = "Actual outcome",
    y = "Predicted probability"
  ) +
  theme_minimal() +
  theme(legend.position = "none")


# ------------------------------------------------------------
# H5. Mean predicted probabilities vs actual frequencies
# ------------------------------------------------------------

actual_frequencies <- prop.table(
  table(
    factor(
      outer_match_predictions$Actual_Outcome,
      levels = outcome_levels
    )
  )
)

mean_predicted_probabilities <- colMeans(
  outer_match_predictions[
    , c("P_Win", "P_Draw", "P_Loss")
  ]
)

calibration_summary <- data.frame(
  
  Outcome = outcome_levels,
  
  Actual_Frequency = as.numeric(actual_frequencies),
  
  Mean_Predicted_Probability =
    as.numeric(mean_predicted_probabilities)
)

calibration_long <- calibration_summary %>%
  
  pivot_longer(
    cols = c(
      Actual_Frequency,
      Mean_Predicted_Probability
    ),
    names_to = "Measure",
    values_to = "Probability"
  ) %>%
  
  mutate(
    Measure = recode(
      Measure,
      Actual_Frequency = "Actual frequency",
      Mean_Predicted_Probability = "Mean predicted probability"
    )
  )

ggplot(
  calibration_long,
  aes(
    x = Outcome,
    y = Probability,
    fill = Measure
  )
) +
  geom_col(
    position = "dodge",
    width = 0.7
  ) +
  labs(
    title = "Mean Predicted Probabilities vs Actual Frequencies",
    subtitle = "Nested CV outer-fold predictions",
    x = "Match outcome",
    y = "Probability / frequency",
    fill = ""
  ) +
  theme_minimal()


# ------------------------------------------------------------
# H6. Selected threshold by outer tournament
# ------------------------------------------------------------

ggplot(
  selected_parameters,
  aes(
    x = Outer_Tournament,
    y = Best_Threshold
  )
) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Selected Classification Thresholds",
    subtitle = "Each threshold selected exclusively through inner CV",
    x = "Outer validation tournament",
    y = "Selected threshold"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# H7. RF vs historical baseline accuracy
# ------------------------------------------------------------

accuracy_comparison <- outer_results %>%
  
  select(
    Outer_Tournament,
    Outer_Accuracy,
    Baseline_Accuracy
  ) %>%
  
  pivot_longer(
    cols = c(Outer_Accuracy, Baseline_Accuracy),
    names_to = "Model",
    values_to = "Accuracy"
  ) %>%
  
  mutate(
    Model = recode(
      Model,
      Outer_Accuracy = "RF + Skellam + Threshold",
      Baseline_Accuracy = "Historical Frequency"
    )
  )

ggplot(
  accuracy_comparison,
  aes(
    x = Outer_Tournament,
    y = Accuracy,
    fill = Model
  )
) +
  geom_col(position = "dodge") +
  labs(
    title = "Classification Accuracy: RF vs Historical Baseline",
    subtitle = "Nested leave-one-tournament-out CV",
    x = "Held-out tournament",
    y = "Accuracy",
    fill = "Model"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# H8. RF vs historical baseline probability metrics
# ------------------------------------------------------------

comparison_long <- model_comparison %>%
  
  select(
    Model,
    Pooled_Log_Loss,
    Pooled_Brier_Score,
    Pooled_Accuracy
  ) %>%
  
  pivot_longer(
    cols = -Model,
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  
  mutate(
    Metric = recode(
      Metric,
      Pooled_Log_Loss = "Log Loss",
      Pooled_Brier_Score = "Brier Score",
      Pooled_Accuracy = "Accuracy"
    )
  )

ggplot(
  comparison_long,
  aes(
    x = Model,
    y = Value,
    fill = Model
  )
) +
  geom_col() +
  facet_wrap(~Metric, scales = "free_y") +
  labs(
    title = "Nested CV: Model Comparison",
    subtitle = "Pooled outer-fold classification metrics",
    x = "",
    y = "Metric value",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 15,
      hjust = 1
    ),
    legend.position = "none"
  )


# ============================================================
# I. FINAL CHECKS
# ============================================================

cat("\n============================================\n")
cat("FINAL NESTED CV CHECKS\n")
cat("============================================\n")

cat("\nNumber of outer-fold predictions:",
    nrow(outer_predictions), "\n")

cat("Number of evaluated matches:",
    nrow(outer_match_predictions), "\n")

cat("\nSelected thresholds:\n")
print(selected_parameters[
  , c("Outer_Tournament", "Best_Threshold")
])

cat("\nPooled RF classification accuracy:",
    pooled_classification_metrics$Accuracy, "\n")

cat("Pooled historical baseline accuracy:",
    pooled_baseline_metrics$Accuracy, "\n")

cat("\nPooled RF Log Loss:",
    pooled_classification_metrics$Log_Loss, "\n")

cat("Pooled RF Brier Score:",
    pooled_classification_metrics$Brier_Score, "\n")

cat("\nNESTED CV PIPELINE FINISHED SUCCESSFULLY\n")
