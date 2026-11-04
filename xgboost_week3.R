# =============================================================
# Fallstudien 2 - UEFA Euro - XGBoost with nested cross-validation
#
# Outer loop (6 seasons) = evaluation   -> honest performance estimate
# Inner loop (5 seasons) = tuning       -> hyperparameter selection
# Final pass (all 6)     = tuning + fit -> the model for section 3
#
# LOSO = leave-one-season-out: one fold per tournament.
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

# sanity checks - these caught a duplicated-file problem once already
stopifnot(nrow(euro) == 492, sum(is.na(euro)) == 0)
print(table(euro$Year))                                   # 62 62 62 102 102 102
print(c(mean = mean(euro$Goals), var = var(euro$Goals)))  # ~1.20 / ~1.25

# match key: both rows of a match get the same ID.
# pmin/pmax sort the team names so the order of Team/Opponent does not matter;
# GroupStage is included because three team pairs met twice in one tournament
# (once in the group stage, once in the knockout round).
euro$MatchID <- paste(euro$Year,
                      pmin(euro$Team, euro$Opponent),
                      pmax(euro$Team, euro$Opponent),
                      euro$GroupStage,
                      sep = "_")
stopifnot(all(table(euro$MatchID) == 2))              # 246 matches, each twice

# design matrix (Team/Opponent excluded: the predictors are already
# differences team - opponent and therefore encode team strength)
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

# Poisson negative log-likelihood, matching xgboost's "poisson-nloglik":
#   lambda - k*log(lambda) + log(k!)
# The lfactorial term is constant in the model but xgboost includes it,
# so it must be here too or the two numbers are not comparable.
poisson_nll <- function(actual, lambda) {
  mean(lambda - actual * log(lambda) + lfactorial(actual))
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
#
# Tuning criterion: "poisson-nloglik" is native to the objective, so
# hyperparameters are selected by the loss actually being minimised.
# Switch to "mae" to check whether the criterion changes the result.
# -------------------------------------------------------------

tune_metric <- "poisson-nloglik"     # or "mae"

# the evaluation_log column name xgb.cv produces for that metric
metric_col <- paste0("test_", gsub("-", "_", tune_metric), "_mean")


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
    cv <- xgb.cv(
      params = list(
        objective        = "count:poisson",   # goals are counts
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
    stopifnot(metric_col %in% names(log))   # catches a metric/column mismatch
    
    best <- which.min(log[[metric_col]])    # mean over the inner folds
    grid$cv_score[i] <- log[[metric_col]][best]
    grid$nrounds[i]  <- best
  }
  
  grid[which.min(grid$cv_score), ]
}


# REFIT: train with a chosen combination on the given rows.
# eval_metric is irrelevant here (no evaluation set is passed), but kept
# consistent with the tuning criterion to avoid confusion.
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
# 4. Nested CV  -> honest performance estimate (SECTION 1)
#
# Each outer fold re-tunes from scratch, so the held-out season never
# influences the hyperparameters used to predict it. The six chosen
# combinations are a byproduct; the six errors are the result.
# -------------------------------------------------------------

results    <- data.frame()
pred_outer <- rep(NA, nrow(euro))    # out-of-fold predictions, kept for
# the match-outcome measures later

for (s in seasons) {
  cat("Outer fold:", s, "\n")
  
  test_rows <- fold_idx[[as.character(s)]]
  pool_rows <- setdiff(seq_len(nrow(euro)), test_rows)
  
  set.seed(42)
  combo <- tune_xgb(pool_rows, grid_base)   # inner loop: 54 combos x 5 seasons
  fit   <- fit_xgb(pool_rows, combo)        # refit on all 5 pool seasons
  
  p <- predict(fit, xgb.DMatrix(X[test_rows, ]))
  pred_outer[test_rows] <- p
  
  results <- rbind(results, data.frame(
    season           = s,
    n                = length(test_rows),
    mae              = mean(abs(p - y[test_rows])),
    rmse             = sqrt(mean((p - y[test_rows])^2)),
    nll              = poisson_nll(y[test_rows], p),
    floor_mae        = mean(abs(mean(y[pool_rows]) - y[test_rows])),
    max_depth        = combo$max_depth,
    eta              = combo$eta,
    subsample        = combo$subsample,
    min_child_weight = combo$min_child_weight,
    nrounds          = combo$nrounds
  ))
}

print(results)

cat("\n--- nested CV performance (mean +/- sd over 6 seasons) ---\n")
cat("tuned by:", tune_metric, "\n")
print(c(mean_mae  = mean(results$mae),  sd_mae  = sd(results$mae),
        mean_rmse = mean(results$rmse), sd_rmse = sd(results$rmse),
        mean_nll  = mean(results$nll),  sd_nll  = sd(results$nll)))

cat("\n'do nothing' floor (predict the pool mean):",
    mean(results$floor_mae), "\n")

# robustness: seasons differ in size (62 vs 102 rows)
cat("\nweighted by season size:\n")
print(c(mae  = weighted.mean(results$mae,  results$n),
        rmse = weighted.mean(results$rmse, results$n),
        nll  = weighted.mean(results$nll,  results$n)))


# -------------------------------------------------------------
# 5. Constant baselines
#
# MAE is minimised by the MEDIAN, RMSE by the MEAN. A Poisson model
# predicts the mean, so MAE structurally handicaps it: the constant
# median (1 goal) wins on MAE but loses on RMSE. Clean illustration
# that the winner depends on the evaluation measure - with no
# modelling involved at all.
# -------------------------------------------------------------

baselines <- data.frame()
for (s in seasons) {
  test_rows <- fold_idx[[as.character(s)]]
  pool_rows <- setdiff(seq_len(nrow(euro)), test_rows)
  
  const_median <- median(y[pool_rows])
  const_mean   <- mean(y[pool_rows])
  
  baselines <- rbind(baselines, data.frame(
    season      = s,
    mae_median  = mean(abs(const_median - y[test_rows])),
    rmse_median = sqrt(mean((const_median - y[test_rows])^2)),
    mae_mean    = mean(abs(const_mean   - y[test_rows])),
    rmse_mean   = sqrt(mean((const_mean - y[test_rows])^2))
  ))
}

cat("\n--- constant baselines ---\n")
print(baselines)
print(colMeans(baselines[, -1]))

cat("\n--- model vs constants, season by season ---\n")
print(data.frame(
  season          = results$season,
  mae_xgb         = round(results$mae, 3),
  mae_const_med   = round(baselines$mae_median, 3),
  rmse_xgb        = round(results$rmse, 3),
  rmse_const_mean = round(baselines$rmse_mean, 3)
))


# -------------------------------------------------------------
# 6. Final pass (SECTION 3, not needed for section 1)
#
# Step 1 - TUNING: LOSO over all six seasons. Seasons ARE held out here,
#          one at a time, so this produces a real cv score. What it lacks
#          is an outer layer: every season took part in choosing the winner.
# Step 2 - FIT: one model on all 492 rows. No held-out data, no score.
# -------------------------------------------------------------

set.seed(42)
final_combo <- tune_xgb(seq_len(nrow(euro)), grid_base)   # step 1
final_fit   <- fit_xgb(seq_len(nrow(euro)), final_combo)  # step 2

cat("\n--- final hyperparameters (step 1) ---\n")
print(final_combo)

cat("\n--- variable importance (report Gain) ---\n")
print(xgb.importance(model = final_fit))

# which predictor each stump split on
trees  <- xgb.model.dt.tree(model = final_fit)
splits <- trees[Feature != "Leaf", .(Tree, Feature, Split)]
cat("\n--- stumps per predictor ---\n")
print(table(splits$Feature))


# -------------------------------------------------------------
# 7. How optimistic is the tuning score?
#
# Both rows train on 5 seasons and predict the 6th, on the same folds.
# They differ only in WHEN the hyperparameters were chosen:
#   nested CV   - chosen without ever seeing the held-out season
#   tuning pass - chosen using all six seasons
# The gap is the selection bias, isolated.
#
# Note: this re-runs the winner at a FIXED nrounds rather than early
# stopping per fold, so it reconstructs the tuning score closely, not
# exactly.
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
  data    = xgb.DMatrix(X, label = y),
  nrounds = final_combo$nrounds,
  folds   = all_folds,
  verbose = 0
)

b   <- nrow(cv_final$evaluation_log)
tun <- cv_final$evaluation_log[b]

cat("\n--- nested CV vs tuning pass ---\n")
comparison <- data.frame(
  metric      = c("MAE", "RMSE", "Poisson NLL"),
  nested_cv   = c(mean(results$mae), mean(results$rmse), mean(results$nll)),
  nested_sd   = c(sd(results$mae),   sd(results$rmse),   sd(results$nll)),
  tuning_pass = c(tun$test_mae_mean, tun$test_rmse_mean,
                  tun$test_poisson_nloglik_mean)
)
comparison$gap        <- comparison$nested_cv - comparison$tuning_pass
comparison$gap_in_sds <- round(comparison$gap / comparison$nested_sd, 3)
print(comparison, row.names = FALSE, digits = 4)

# sanity: all three gaps should be small and of the same order.
# A gap of several sd on one metric only means a definition mismatch,
# not a real effect.


#saveRDS(list(fit = final_fit, combo = final_combo,
 #            results = results, baselines = baselines,
  #           comparison = comparison,
   #          pred_outer = pred_outer, tune_metric = tune_metric),
    #    paste0(path, "xgb_", gsub("-", "_", tune_metric), ".rds"))