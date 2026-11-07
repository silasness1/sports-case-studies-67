# =============================================================
# Fallstudien 2 - UEFA Euro - XGBoost with nested cross-validation
#
# Outer loop (6 seasons) = evaluation   -> honest performance estimate
# Inner loop (5 seasons) = tuning       -> hyperparameter selection
# Final pass (all 6)     = tuning + fit -> the final model
#
# LOSO = leave-one-season-out: one fold per tournament.
#
# Every error is reported two ways:
#   mean   = unweighted mean of the 6 per-season errors
#   pooled = computed once over all 492 out-of-fold predictions,
#            i.e. weighted by season size (62 vs 102 rows)
# =============================================================

library(data.table)
library(xgboost)

set.seed(42)


# -------------------------------------------------------------
# 1. Data
# -------------------------------------------------------------

path <- "D:/Course/Case Studies/Football/"

euro_2004 <- fread(paste0(path, "euro_2004.csv"))
euro_2008 <- fread(paste0(path, "euro_2008.csv"))
euro_2012 <- fread(paste0(path, "euro_2012.csv"))
euro_2016 <- fread(paste0(path, "euro_2016.csv"))
euro_2020 <- fread(paste0(path, "euro_2020.csv"))
euro_2024 <- fread(paste0(path, "euro_2024.csv"))

euro <- rbind(euro_2004, euro_2008, euro_2012,
              euro_2016, euro_2020, euro_2024)

# sanity checks
stopifnot(nrow(euro) == 492, sum(is.na(euro)) == 0)
print(table(euro$Year))                                   # 62 62 62 102 102 102
print(c(mean = mean(euro$Goals), var = var(euro$Goals)))  # ~1.20 / ~1.25

# match key: both rows of a match get the same ID.
# GroupStage is included because three team pairs met twice in one
# tournament (group stage and knockout).
euro$MatchID <- paste(euro$Year,
                      pmin(euro$Team, euro$Opponent),
                      pmax(euro$Team, euro$Opponent),
                      euro$GroupStage,
                      sep = "_")
stopifnot(all(table(euro$MatchID) == 2))              # 246 matches, each twice

# design matrix (Team/Opponent excluded: the predictors are already
# differences team - opponent)
predictors <- c("GroupStage", "GDP", "MarketValue",
                "FifaRank", "UefaPoints", "CLPlayers")

X <- as.matrix(as.data.frame(euro)[, predictors])
y <- euro$Goals

# one fold per season - used by BOTH cv loops
seasons  <- c(2004, 2008, 2012, 2016, 2020, 2024)
fold_idx <- list()
for (yr in seasons) {
  fold_idx[[as.character(yr)]] <- which(euro$Year == yr)
}


# -------------------------------------------------------------
# Metrics
#
# Poisson NLL matches xgboost's "poisson-nloglik":
#   lambda - k*log(lambda) + log(k!)
# -------------------------------------------------------------

poisson_nll <- function(actual, lambda) {
  mean(lambda - actual * log(lambda) + lfactorial(actual))
}

# all three metrics over one set of predictions
metrics <- function(actual, pred) {
  c(mae  = mean(abs(pred - actual)),
    rmse = sqrt(mean((pred - actual)^2)),
    nll  = poisson_nll(actual, pred))
}


# -------------------------------------------------------------
# 2. Hyperparameter grid
#
# max_depth 1-3   : ESL ch.10 recommends 4-8 terminal nodes for boosting
# eta 0.01-0.1    : Friedman (2001); small shrinkage generalises better
# subsample .5/.8 : Friedman (2002), stochastic gradient boosting
# min_child_weight: main overfitting control (xgboost docs)
# nrounds is NOT in the grid - early stopping finds it per combination
# -------------------------------------------------------------

grid_base <- expand.grid(
  max_depth        = c(1, 2, 3),
  eta              = c(0.01, 0.05, 0.1),
  subsample        = c(0.5, 0.8),
  min_child_weight = c(1, 5, 10)
)


# -------------------------------------------------------------
# 3. Helper functions
# -------------------------------------------------------------

tune_metric <- "poisson-nloglik"     # or "mae"
metric_col  <- paste0("test_", gsub("-", "_", tune_metric), "_mean")


# INNER LOOP: LOSO tuning on a pool of seasons.
# Returns the single best hyperparameter combination.
tune_xgb <- function(pool_rows, grid) {
  
  dpool      <- xgb.DMatrix(X[pool_rows, ], label = y[pool_rows])
  pool_years <- euro$Year[pool_rows]
  
  # NOTE: xgb.cv indexes rows INSIDE dpool, not inside euro
  inner_folds <- list()
  for (yr in sort(unique(pool_years))) {
    inner_folds[[as.character(yr)]] <- which(pool_years == yr)
  }
  
  grid$cv_score <- NA
  grid$nrounds  <- NA
  
  for (i in 1:nrow(grid)) {
    cat("\r  combo", i, "of", nrow(grid), "  ")
    flush.console()
    
    cv <- xgb.cv(
      params = list(
        objective        = "count:poisson",
        eval_metric      = tune_metric,
        max_depth        = grid$max_depth[i],
        eta              = grid$eta[i],
        subsample        = grid$subsample[i],
        min_child_weight = grid$min_child_weight[i],
        colsample_bytree = 0.8
      ),
      data    = dpool,
      nrounds = 2000,
      folds   = inner_folds,
      early_stopping_rounds = 50,
      verbose = 0
    )
    
    log <- cv$evaluation_log
    stopifnot(metric_col %in% names(log))
    
    best <- which.min(log[[metric_col]])
    grid$cv_score[i] <- log[[metric_col]][best]
    grid$nrounds[i]  <- best
  }
  cat("\n")
  
  grid[which.min(grid$cv_score), ]
}


# REFIT: train with a chosen combination on the given rows
fit_xgb <- function(rows, combo) {
  xgb.train(
    params = list(
      objective        = "count:poisson",
      eval_metric      = tune_metric,
      max_depth        = combo$max_depth,
      eta              = combo$eta,
      subsample        = combo$subsample,
      min_child_weight = combo$min_child_weight,
      colsample_bytree = 0.8
    ),
    data    = xgb.DMatrix(X[rows, ], label = y[rows]),
    nrounds = combo$nrounds,
    verbose = 0
  )
}


# -------------------------------------------------------------
# 4. Nested CV  -> honest performance estimate
#
# 6 outer folds x (54 combinations x 5 inner folds + 1 refit) = 1,626 fits
# -------------------------------------------------------------

results    <- data.frame()
pred_outer <- rep(NA, nrow(euro))    # out-of-fold predictions

for (s in seasons) {
  cat("Outer fold:", s, "\n")
  
  test_rows <- fold_idx[[as.character(s)]]
  pool_rows <- setdiff(seq_len(nrow(euro)), test_rows)
  
  set.seed(42)
  combo <- tune_xgb(pool_rows, grid_base)
  fit   <- fit_xgb(pool_rows, combo)
  
  p <- predict(fit, xgb.DMatrix(X[test_rows, ]))
  pred_outer[test_rows] <- p
  
  m <- metrics(y[test_rows], p)
  
  results <- rbind(results, data.frame(
    season           = s,
    n                = length(test_rows),
    mae              = m["mae"],
    rmse             = m["rmse"],
    nll              = m["nll"],
    floor_mae        = mean(abs(mean(y[pool_rows]) - y[test_rows])),
    max_depth        = combo$max_depth,
    eta              = combo$eta,
    subsample        = combo$subsample,
    min_child_weight = combo$min_child_weight,
    nrounds          = combo$nrounds,
    row.names        = NULL
  ))
}

stopifnot(!anyNA(pred_outer))
print(results)

cat("\n--- nested CV performance ---\n")
cat("tuned by:", tune_metric, "\n")
nested_summary <- rbind(
  mean   = c(mae = mean(results$mae), rmse = mean(results$rmse),
             nll = mean(results$nll)),
  sd     = c(mae = sd(results$mae),   rmse = sd(results$rmse),
             nll = sd(results$nll)),
  pooled = metrics(y, pred_outer)
)
print(round(nested_summary, 4))

cat("\n'do nothing' floor (predict the pool mean), mean MAE:",
    round(mean(results$floor_mae), 4), "\n")


# -------------------------------------------------------------
# 5. Constant baselines
#
# MAE is minimised by the MEDIAN, RMSE by the MEAN. A Poisson model
# predicts the mean, so MAE structurally handicaps it: the constant
# median (1 goal) wins on MAE but loses on RMSE.
# -------------------------------------------------------------

baselines   <- data.frame()
pred_median <- rep(NA, nrow(euro))
pred_mean   <- rep(NA, nrow(euro))

for (s in seasons) {
  test_rows <- fold_idx[[as.character(s)]]
  pool_rows <- setdiff(seq_len(nrow(euro)), test_rows)
  
  pred_median[test_rows] <- median(y[pool_rows])
  pred_mean[test_rows]   <- mean(y[pool_rows])
  
  baselines <- rbind(baselines, data.frame(
    season      = s,
    mae_median  = mean(abs(pred_median[test_rows] - y[test_rows])),
    rmse_median = sqrt(mean((pred_median[test_rows] - y[test_rows])^2)),
    mae_mean    = mean(abs(pred_mean[test_rows] - y[test_rows])),
    rmse_mean   = sqrt(mean((pred_mean[test_rows] - y[test_rows])^2))
  ))
}

cat("\n--- constant baselines, per season ---\n")
print(baselines)

cat("\n--- model vs constants (pooled over all 492 rows) ---\n")
print(round(rbind(
  xgboost         = metrics(y, pred_outer)[c("mae", "rmse")],
  constant_median = metrics(y, pred_median)[c("mae", "rmse")],
  constant_mean   = metrics(y, pred_mean)[c("mae", "rmse")]
), 4))

cat("\n--- model vs constants, season by season ---\n")
print(data.frame(
  season          = results$season,
  mae_xgb         = round(results$mae, 3),
  mae_const_med   = round(baselines$mae_median, 3),
  rmse_xgb        = round(results$rmse, 3),
  rmse_const_mean = round(baselines$rmse_mean, 3)
))


# -------------------------------------------------------------
# 6. Final pass
#
# Step 1 - TUNING: LOSO over all six seasons, 54 x 6 = 324 fits.
#          Seasons are held out one at a time, so this gives a real cv
#          score, but every season took part in choosing the winner.
# Step 2 - FIT: one model on all 492 rows. No held-out data, no score.
# -------------------------------------------------------------

cat("\n--- final pass: tuning on all 6 seasons ---\n")
set.seed(42)
final_combo <- tune_xgb(seq_len(nrow(euro)), grid_base)   # step 1
final_fit   <- fit_xgb(seq_len(nrow(euro)), final_combo)  # step 2

cat("\n--- final hyperparameters ---\n")
print(final_combo)

cat("\n--- variable importance (report Gain) ---\n")
print(xgb.importance(model = final_fit))

trees  <- xgb.model.dt.tree(model = final_fit)
splits <- trees[Feature != "Leaf", .(Tree, Feature, Split)]
cat("\n--- stumps per predictor ---\n")
print(table(splits$Feature))


# -------------------------------------------------------------
# 7. How optimistic is the tuning score?
#
# Re-run the winning combination over the same 6 LOSO folds, keeping
# the out-of-fold predictions so it can be pooled exactly like the
# nested CV. Both train on 5 seasons and predict the 6th; they differ
# only in whether the predicted season helped choose the
# hyperparameters. The gap is the selection bias.
# -------------------------------------------------------------

all_folds <- list()
for (yr in seasons) all_folds[[as.character(yr)]] <- which(euro$Year == yr)

set.seed(42)
cv_final <- xgb.cv(
  params = list(
    objective        = "count:poisson",
    eval_metric      = c("poisson-nloglik", "mae", "rmse"),
    max_depth        = final_combo$max_depth,
    eta              = final_combo$eta,
    subsample        = final_combo$subsample,
    min_child_weight = final_combo$min_child_weight,
    colsample_bytree = 0.8
  ),
  data       = xgb.DMatrix(X, label = y),
  nrounds    = final_combo$nrounds,
  folds      = all_folds,
  prediction = TRUE,
  verbose    = 0
)

# the location of the predictions differs between xgboost versions
pred_tuning <- if (!is.null(cv_final$cv_predict$pred)) {
  cv_final$cv_predict$pred
} else {
  cv_final$pred
}
stopifnot(length(pred_tuning) == nrow(euro), !anyNA(pred_tuning))

# per-season errors for the tuning pass
tuning_fold <- t(sapply(seasons, function(s) {
  r <- fold_idx[[as.character(s)]]
  metrics(y[r], pred_tuning[r])
}))

nested_mean   <- colMeans(results[, c("mae", "rmse", "nll")])
nested_pooled <- metrics(y, pred_outer)
tuning_mean   <- colMeans(tuning_fold)
tuning_pooled <- metrics(y, pred_tuning)

comparison <- data.frame(
  metric        = c("MAE", "RMSE", "Poisson NLL"),
  nested_mean   = nested_mean,
  nested_pooled = nested_pooled,
  tuning_mean   = tuning_mean,
  tuning_pooled = tuning_pooled,
  gap_pooled    = nested_pooled - tuning_pooled,
  row.names     = NULL
)
comparison$gap_in_sds <- round(comparison$gap_pooled /
                                 c(sd(results$mae), sd(results$rmse),
                                   sd(results$nll)), 3)

cat("\n--- nested CV vs tuning pass ---\n")
print(comparison, digits = 4)


saveRDS(list(fit = final_fit, combo = final_combo,
             results = results, baselines = baselines,
             comparison = comparison,
             pred_outer = pred_outer, pred_tuning = pred_tuning,
             tune_metric = tune_metric),
        paste0(path, "xgb_", gsub("-", "_", tune_metric), ".rds"))


results[, c("season","max_depth","eta","nrounds", "subsample", "min_child_weight")]


# =============================================================
# CONTINUATION of xgboost_nested_cv_v5.R
#
# Requires from that script: euro, X, y, seasons, fold_idx,
#                            final_combo, path
# Nothing is reloaded or retuned.
#
#   8. out-of-fold predictions with the selected hyperparameters
#   9. pair the two rows of each match
#   10. Skellam win/draw/loss probabilities
#   11. actual and predicted outcomes
#   12. log loss, Brier, RPS, accuracy
#   13. historical-frequency baseline
#   14. plots
# =============================================================

library(skellam)
library(ggplot2)
library(tidyr)
library(dplyr)

stopifnot(exists("euro"), exists("X"), exists("y"),
          exists("final_combo"), exists("fold_idx"))


# -------------------------------------------------------------
# 8. Out-of-fold predictions with the selected hyperparameters
#
# For each season: fit on the other five, predict the held-out one.
# The final model is NOT used - it has seen every season.
# -------------------------------------------------------------

pred_oof <- rep(NA_real_, nrow(euro))

for (s in seasons) {
  test_rows <- fold_idx[[as.character(s)]]
  pool_rows <- setdiff(seq_len(nrow(euro)), test_rows)
  
  set.seed(42)
  fit <- fit_xgb(pool_rows, final_combo)
  
  pred_oof[test_rows] <- predict(fit, xgb.DMatrix(X[test_rows, ]))
}

stopifnot(!anyNA(pred_oof))

# should reproduce the tuning-pass figures (~0.828 / ~1.035)
cat("\ngoal-level check:\n")
print(c(MAE  = mean(abs(pred_oof - y)),
        RMSE = sqrt(mean((pred_oof - y)^2))))


# -------------------------------------------------------------
# 9. Pair the two rows of each match
#
# Team1 = alphabetically first. MatchID already includes GroupStage,
# because three team pairs met twice in the same tournament.
# -------------------------------------------------------------

oof <- data.frame(
  Tournament     = euro$Year,
  match_id       = euro$MatchID,
  Team           = as.character(euro$Team),
  Opponent       = as.character(euro$Opponent),
  Goals          = y,
  PredictedGoals = pred_oof
)

match_data <- oof %>%
  mutate(
    Team1_name = pmin(Team, Opponent),
    Team2_name = pmax(Team, Opponent),
    Is_Team1   = Team == Team1_name
  ) %>%
  group_by(Tournament, match_id) %>%
  summarise(
    N_Teams = n(),
    Team1   = first(Team1_name),
    Team2   = first(Team2_name),
    Goals1  = first(Goals[Is_Team1]),
    Goals2  = first(Goals[!Is_Team1]),
    Lambda1 = first(PredictedGoals[Is_Team1]),
    Lambda2 = first(PredictedGoals[!Is_Team1]),
    .groups = "drop"
  )

stopifnot(all(match_data$N_Teams == 2), nrow(match_data) == 246)


# -------------------------------------------------------------
# 10. Skellam win/draw/loss probabilities
#
# D = G1 - G2 with G1 ~ Pois(Lambda1), G2 ~ Pois(Lambda2)
# -------------------------------------------------------------

match_data <- match_data %>%
  mutate(
    P_Win  = pskellam(0,  Lambda1, Lambda2, lower.tail = FALSE),  # D >= 1
    P_Draw = dskellam(0,  Lambda1, Lambda2),                      # D = 0
    P_Loss = pskellam(-1, Lambda1, Lambda2)                       # D <= -1
  )

stopifnot(all(abs(match_data$P_Win + match_data$P_Draw +
                    match_data$P_Loss - 1) < 1e-8))

print(head(as.data.frame(match_data[, c("Tournament", "Team1", "Team2",
                                        "Lambda1", "Lambda2",
                                        "P_Win", "P_Draw", "P_Loss")]), 10),
      digits = 3)


# -------------------------------------------------------------
# 11. Actual and predicted outcomes
# -------------------------------------------------------------

outcome_levels <- c("Win", "Draw", "Loss")

match_data <- match_data %>%
  mutate(
    Actual_Outcome = factor(
      case_when(Goals1 >  Goals2 ~ "Win",
                Goals1 == Goals2 ~ "Draw",
                TRUE             ~ "Loss"),
      levels = outcome_levels),
    Predicted_Outcome = factor(
      case_when(P_Win  >= P_Draw & P_Win >= P_Loss ~ "Win",
                P_Draw >= P_Win  & P_Draw >= P_Loss ~ "Draw",
                TRUE                                ~ "Loss"),
      levels = outcome_levels)
  )

# draws are almost never the single most likely outcome
cat("\npredicted (rows) vs actual (columns):\n")
print(table(predicted = match_data$Predicted_Outcome,
            actual    = match_data$Actual_Outcome))


# -------------------------------------------------------------
# 12. Scoring rules
# -------------------------------------------------------------

# Rank Probability Score - as given on the exercise sheet
rank_prob_score <- function(pred_matrix, obs) {
  n <- nrow(pred_matrix)
  C <- ncol(pred_matrix)
  obs_mat <- matrix(0, n, C)
  obs_mat[cbind(1:n, as.integer(obs))] <- 1
  P_cum <- t(apply(pred_matrix, 1, cumsum))
  O_cum <- t(apply(obs_mat, 1, cumsum))
  mean(rowSums((P_cum - O_cum)^2) / (C - 1))
}

outcome_metrics <- function(prob_matrix, actual) {
  n   <- nrow(prob_matrix)
  idx <- as.integer(actual)
  
  actual_matrix <- matrix(0, n, 3)
  actual_matrix[cbind(1:n, idx)] <- 1
  
  true_probabilities <- prob_matrix[cbind(1:n, idx)]
  
  data.frame(
    Log_Loss    = -mean(log(pmax(true_probabilities, 1e-15))),
    Brier_Score = mean(rowSums((prob_matrix - actual_matrix)^2)),
    RPS         = rank_prob_score(prob_matrix, actual),
    Accuracy    = mean(max.col(prob_matrix, ties.method = "first") == idx)
  )
}

prob_matrix <- as.matrix(match_data[, c("P_Win", "P_Draw", "P_Loss")])
colnames(prob_matrix) <- outcome_levels

xgb_metrics <- outcome_metrics(prob_matrix, match_data$Actual_Outcome)

# per tournament, for paired comparisons with the other methods
per_tournament <- match_data %>%
  group_by(Tournament) %>%
  group_modify(~ outcome_metrics(
    as.matrix(.x[, c("P_Win", "P_Draw", "P_Loss")]),
    .x$Actual_Outcome)) %>%
  ungroup()

cat("\nper tournament:\n")
print(as.data.frame(per_tournament), digits = 4)


# -------------------------------------------------------------
# 13. Baseline: historical outcome frequencies
# -------------------------------------------------------------

baseline_matrix <- matrix(NA_real_, nrow(match_data), 3,
                          dimnames = list(NULL, outcome_levels))

for (s in seasons) {
  idx  <- which(match_data$Tournament == s)
  freq <- prop.table(table(
    match_data$Actual_Outcome[match_data$Tournament != s]))[outcome_levels]
  baseline_matrix[idx, ] <- matrix(as.numeric(freq), length(idx), 3,
                                   byrow = TRUE)
}

model_comparison <- rbind(
  cbind(Model = "XGBoost + Skellam",
        xgb_metrics),
  cbind(Model = "Historical Frequency Baseline",
        outcome_metrics(baseline_matrix, match_data$Actual_Outcome))
)

cat("\n--- outcome measures, 246 matches ---\n")
cat("lower is better for Log Loss, Brier, RPS; higher for Accuracy\n")
print(model_comparison, row.names = FALSE, digits = 4)

cat("\nmean predicted probability vs actual frequency:\n")
print(round(rbind(
  predicted = colMeans(prob_matrix),
  actual    = as.numeric(prop.table(table(match_data$Actual_Outcome)))
), 3))


# -------------------------------------------------------------
# 14. Plots
# -------------------------------------------------------------

p1 <- match_data %>%
  select(P_Win, P_Draw, P_Loss) %>%
  pivot_longer(everything(), names_to = "Outcome",
               values_to = "Probability") %>%
  mutate(Outcome = factor(recode(Outcome, P_Win = "Win",
                                 P_Draw = "Draw", P_Loss = "Loss"),
                          levels = outcome_levels)) %>%
  ggplot(aes(x = Probability, fill = Outcome)) +
  geom_histogram(bins = 25, alpha = 0.75, color = "white") +
  facet_wrap(~Outcome, ncol = 1) +
  labs(title = "XGBoost: Predicted Match Outcome Probabilities",
       x = "Predicted Probability", y = "Number of Matches") +
  theme_minimal() + theme(legend.position = "none")
print(p1)

p2 <- data.frame(
  Outcome   = factor(outcome_levels, levels = outcome_levels),
  Predicted = colMeans(prob_matrix),
  Actual    = as.numeric(prop.table(table(match_data$Actual_Outcome)))
) %>%
  pivot_longer(c(Predicted, Actual), names_to = "Measure",
               values_to = "Probability") %>%
  ggplot(aes(x = Outcome, y = Probability, fill = Measure)) +
  geom_col(position = "dodge", width = 0.7) +
  labs(title = "XGBoost: Mean Predicted Probability vs Actual Frequency",
       x = "Match Outcome", y = "Probability") +
  theme_minimal()
print(p2)


# -------------------------------------------------------------
# Save
# -------------------------------------------------------------

write.csv(match_data, paste0(path, "xgb_match_probabilities.csv"),
          row.names = FALSE)

saveRDS(list(match_data = match_data, metrics = model_comparison,
             per_tournament = per_tournament, pred_oof = pred_oof),
        paste0(path, "xgb_skellam.rds"))


