
# ============================================================
# NESTED LEAVE-ONE-TOURNAMENT-OUT CROSS-VALIDATION
# RANDOM FOREST
# ============================================================

library(ranger)
library(dplyr)
library(ggplot2)

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

required_columns <- c(target, predictors)

rf_formula <- as.formula(
  paste(target, "~", paste(predictors, collapse = " + "))
)


# ------------------------------------------------------------
# A2. Combine the six tournament datasets
# ------------------------------------------------------------

euro_data <- list(
  "2004" = euro_2004,
  "2008" = euro_2008,
  "2012" = euro_2012,
  "2016" = euro_2016,
  "2020" = euro_2020,
  "2024" = euro_2024
)

EuroAll <- bind_rows(euro_data, .id = "Tournament")

# Give every observation a unique identifier
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


# ------------------------------------------------------------
# A3. Evaluation metrics
# ------------------------------------------------------------

calculate_metrics <- function(actual, predicted) {
  
  # Ensure strictly positive Poisson rate predictions
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
# A4. Original 72-combination hyperparameter grid
# ------------------------------------------------------------

rf_grid <- expand.grid(
  mtry = 1:length(predictors),
  min.node.size = c(3, 5, 10, 15),
  sample.fraction = c(0.7, 0.85, 1.0)
)

stopifnot(nrow(rf_grid) == 72)

print(rf_grid)


# ============================================================
# NESTED CROSS-VALIDATION
# ============================================================

# Store outer-fold performance
outer_results <- vector("list", length(tournament_years))

# Store outer-fold predictions
outer_predictions <- vector("list", length(tournament_years))

# Store selected hyperparameters
selected_parameters <- vector("list", length(tournament_years))

# Store all inner CV results
all_inner_results <- vector("list", length(tournament_years))


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
  
  # The five tournaments available for inner CV
  inner_years <- tournament_years[
    tournament_years != outer_year
  ]
  
  # Store results for the 72 combinations
  inner_results <- vector("list", nrow(rf_grid))
  
  
  # ----------------------------------------------------------
  # Inner loop: tune hyperparameters using the five
  # remaining tournaments
  # ----------------------------------------------------------
  
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
      
      # Fit candidate Random Forest
      model_seed <- BASE_SEED +
        outer_i * 100000 +
        grid_i * 10 +
        inner_j
      
      rf_model <- ranger(
        formula = rf_formula,
        data = inner_train[, required_columns],
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
      
      # Calculate inner validation metrics
      metrics <- calculate_metrics(
        inner_validation$Goals,
        predictions
      )
      
      # IMPORTANT: Save all three metrics
      fold_results[[inner_j]] <- data.frame(
        Tournament = inner_year,
        MAE = metrics$MAE,
        RMSE = metrics$RMSE,
        Poisson_NLL = metrics$Poisson_NLL
      )
    }
    
    # Combine the five inner-fold results
    fold_results_df <- bind_rows(fold_results)
    
    # Average across tournaments: equal weight per tournament
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
  # Select best hyperparameters using inner CV MAE
  # ----------------------------------------------------------
  
  inner_results_df <- bind_rows(inner_results) %>%
    arrange(Mean_MAE)
  
  best_params <- inner_results_df[1, ]
  
  all_inner_results[[outer_i]] <- inner_results_df
  
  selected_parameters[[outer_i]] <- data.frame(
    Outer_Tournament = outer_year,
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    Inner_MAE = best_params$Mean_MAE,
    Inner_RMSE = best_params$Mean_RMSE,
    Inner_Poisson_NLL = best_params$Mean_Poisson_NLL
  )
  
  cat("\nSelected hyperparameters:\n")
  print(selected_parameters[[outer_i]])
  
  
  # ----------------------------------------------------------
  # Refit on ALL five outer-training tournaments
  # ----------------------------------------------------------
  
  final_seed <- BASE_SEED + outer_i * 1000000
  
  rf_outer <- ranger(
    formula = rf_formula,
    data = outer_train[, required_columns],
    num.trees = N_TREES,
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    importance = "none",
    seed = final_seed
  )
  
  
  # ----------------------------------------------------------
  # Evaluate ONLY on the untouched outer test tournament
  # ----------------------------------------------------------
  
  outer_pred <- predict(
    rf_outer,
    data = outer_test[, predictors]
  )$predictions
  
  outer_metrics <- calculate_metrics(
    outer_test$Goals,
    outer_pred
  )
  
  outer_results[[outer_i]] <- data.frame(
    Outer_Tournament = outer_year,
    Train_Observations = nrow(outer_train),
    Test_Observations = nrow(outer_test),
    
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    
    Outer_MAE = outer_metrics$MAE,
    Outer_RMSE = outer_metrics$RMSE,
    Outer_Poisson_NLL = outer_metrics$Poisson_NLL
  )
  
  # Save predictions for later pooled evaluation
  outer_predictions[[outer_i]] <- data.frame(
    Observation_ID = outer_test$Observation_ID,
    Tournament = outer_year,
    Actual_Goals = outer_test$Goals,
    Predicted_Goals = outer_pred
  )
  
  cat("\nOuter test performance:\n")
  print(outer_results[[outer_i]])
  
  cat("\nCompleted outer fold", outer_i,
      "of", length(tournament_years), "\n")
}


# ============================================================
# B1. COMBINE RESULTS FROM ALL SIX OUTER FOLDS
# ============================================================

outer_results <- bind_rows(outer_results)

selected_parameters <- bind_rows(selected_parameters)

outer_predictions <- bind_rows(outer_predictions)

cat("\nNESTED CV COMPLETED\n")


# ============================================================
# C1. OUTER-FOLD PERFORMANCE
# ============================================================

# Display performance for each held-out tournament
print(outer_results, row.names = FALSE)

# Average performance across the six tournaments
mean_outer_MAE <- mean(outer_results$Outer_MAE)

mean_outer_RMSE <- mean(outer_results$Outer_RMSE)

mean_outer_Poisson_NLL <- mean(
  outer_results$Outer_Poisson_NLL
)

sd_outer_MAE <- sd(outer_results$Outer_MAE)

nested_summary <- data.frame(
  Mean_Outer_MAE = mean_outer_MAE,
  Mean_Outer_RMSE = mean_outer_RMSE,
  Mean_Outer_Poisson_NLL = mean_outer_Poisson_NLL,
  SD_Outer_MAE = sd_outer_MAE
)

cat("\nAverage outer-fold performance:\n")
print(nested_summary)


# ============================================================
# D1. POOLED OUTER-FOLD PERFORMANCE
# ============================================================

# Every observation should have exactly one outer-fold prediction
stopifnot(
  nrow(outer_predictions) == nrow(EuroAll),
  !anyDuplicated(outer_predictions$Observation_ID),
  setequal(
    outer_predictions$Observation_ID,
    EuroAll$Observation_ID
  ),
  !anyNA(outer_predictions$Predicted_Goals)
)

# Calculate pooled metrics
pooled_metrics <- calculate_metrics(
  outer_predictions$Actual_Goals,
  outer_predictions$Predicted_Goals
)

cat("\nPooled nested CV performance:\n")
print(pooled_metrics)


# Combined summary
final_nested_summary <- data.frame(
  Metric = c(
    "Pooled MAE",
    "Pooled RMSE",
    "Pooled Poisson NLL",
    "Mean tournament MAE",
    "Mean tournament RMSE",
    "Mean tournament Poisson NLL",
    "SD of tournament MAE"
  ),
  
  Value = c(
    pooled_metrics$MAE,
    pooled_metrics$RMSE,
    pooled_metrics$Poisson_NLL,
    mean_outer_MAE,
    mean_outer_RMSE,
    mean_outer_Poisson_NLL,
    sd_outer_MAE
  )
)

cat("\nFinal nested CV summary:\n")
print(final_nested_summary, row.names = FALSE)


# ============================================================
# E1. SELECTED HYPERPARAMETERS PER OUTER FOLD
# ============================================================

print(selected_parameters, row.names = FALSE)

# Frequency of selected mtry values
cat("\nFrequency of selected mtry values:\n")
print(table(selected_parameters$mtry))

# Frequency of selected minimum node sizes
cat("\nFrequency of selected minimum node sizes:\n")
print(table(selected_parameters$min.node.size))

# Frequency of selected sample fractions
cat("\nFrequency of selected sample fractions:\n")
print(table(selected_parameters$sample.fraction))


# ============================================================
# F1. MAE BY OUTER VALIDATION TOURNAMENT
# ============================================================

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


# ============================================================
# F2. RMSE BY OUTER VALIDATION TOURNAMENT
# ============================================================

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


# ============================================================
# F3. POISSON NLL BY OUTER VALIDATION TOURNAMENT
# ============================================================

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
