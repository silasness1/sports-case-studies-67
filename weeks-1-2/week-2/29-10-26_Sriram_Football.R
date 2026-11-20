EuroTest0 <- EuroTest
EuroTrain0 <- EuroTraining

str(EuroTest0)
summary(EuroTrain0)
sum(is.na(EuroTest0))
sum(complete.cases(EuroTrain0))
hist(EuroTrain0$Goals)
ggplot(EuroTrain0, aes(Goals, FifaRank)) + geom_point()
ggplot(EuroTrain0, aes(Goals, CLPlayers)) + geom_point()
boxplot(EuroTrain0$MarketValue)
model1 <- glm(Goals~., data= EuroTrain0, family= poisson)
summary(model1)
visreg(model1, "MarketValue", scale = "linear")
plot_model(model1, type='pred')
dispersiontest(model1)
model2 <- glm.nb(Goals~., data= EuroTrain0)
summary(model2)
# 2. Get predictions on the response scale (e.g., probabilities)
preds <- predict(model1, type = "response")

# 3. Calculate Mean Absolute Error (MAE)
mae <- mean(abs(EuroTrain0$Goals - preds))
print(mae)


# 3. Calculate Mean Absolute Error (MAE)
mae.test <- mean(abs(EuroTest0$Goals - preds))
print(mae.test)

model3 <- glm(Goals~Year+GroupStage+GDP+MarketValue+FifaRank+UefaPoints+CLPlayers, data= EuroTrain0, family= poisson)
summary(model)
preds_new <- predict(model3, type = "response", newdata=EuroTest0)

# 3. Calculate Mean Absolute Error (MAE)
mae_new <- mean(abs(EuroTest0$Goals - preds_new))
print(mae_new)
preds_new_tr <- predict(model3, type = "response")

# 3. Calculate Mean Absolute Error (MAE)
mae_new_tr <- mean(abs(EuroTrain0$Goals - preds_new_tr))
print(mae_new_tr)

model4 <- glm(Goals~GDP+MarketValue, data= EuroTrain0, family= poisson)
summary(model4)
preds4 <- predict(model4, type = "response")

# 3. Calculate Mean Absolute Error (MAE)
mae4 <- mean(abs(EuroTrain0$Goals - preds4))
print(mae4)
preds4test <- predict(model4, type = "response", newdata = EuroTest0)

# 3. Calculate Mean Absolute Error (MAE)
mae4test <- mean(abs(EuroTest0$Goals - preds4test))
print(mae4test)

model5 <- glm(Goals~Year*GroupStage*GDP*MarketValue*FifaRank*UefaPoints*CLPlayers, data= EuroTrain0, family= poisson)
summary(model5)
preds5 <- predict(model5, type = "response")

# 3. Calculate Mean Absolute Error (MAE)
mae5 <- mean(abs(EuroTrain0$Goals - preds5))
print(mae5)
preds5test <- predict(model5, type = "response", newdata = EuroTest0)

# 3. Calculate Mean Absolute Error (MAE)
mae5test <- mean(abs(EuroTest0$Goals - preds5test))
print(mae5test)
plot(model5)
plot(model4)
plot(EuroTrain0$Goals,model5$fitted.values)
boxplot(model4$residuals)

y <- EuroTrain0$Goals
x <- data.matrix(EuroTrain0[,c("Year","GroupStage","GDP","MarketValue","FifaRank","UefaPoints","CLPlayers")])
set.seed(6769)
cv_model <- cv.glmnet(x,y,alpha=1)
best_lambda <- cv_model$lambda.min
lasso_model_6769 <- glmnet(x,y,alpha=1, lambda = best_lambda)
summary(lasso_model_6769)
coef(lasso_model_6769)
preds_lasso = predict(lasso_model_6769,newx=x,s=best_lambda)
summary(preds_lasso)
preds_lasso

mae4lasso <- mean(abs(EuroTrain0$Goals - preds_lasso))
print(mae4lasso)

# Prepare the test predictor matrix using the same predictors
x_test <- data.matrix(
  EuroTest0[, c(
    "Year", "GroupStage", "GDP", "MarketValue",
    "FifaRank", "UefaPoints", "CLPlayers"
  )]
)

# Predict goals for Euro 2024
preds_lasso_test <- predict(
  lasso_model_6769,
  newx = x_test,
  s = best_lambda
)

# Calculate test errors
mae_lasso_test <- mean(
  abs(EuroTest0$Goals - preds_lasso_test)
)

rmse_lasso_test <- sqrt(
  mean((EuroTest0$Goals - preds_lasso_test)^2)
)

cat("Lasso Test MAE:", mae_lasso_test, "\n")
cat("Lasso Test RMSE:", rmse_lasso_test, "\n")

x_test=data.matrix(EuroTest0[,c("Year","GroupStage","GDP","MarketValue","FifaRank","UefaPoints","CLPlayers")])
preds_lasso_test = predict(lasso_model_6769,newx=x_test,s=best_lambda)
mae4lassotest <- mean(abs(EuroTest0$Goals - preds_lasso_test))
print(mae4lassotest) 
preds1test <- predict(model1, type = "response", newdata = newt)
mae1test <- mean(abs(EuroTest0$Goals - preds1test))
print(mae1test)
  newt=EuroTest0[!(EuroTest0$Team %in% c("Serbia","Georgia","Slovenia")) ,]
newt=newt[!(newt$Opponent %in% c("Serbia","Georgia","Slovenia")) ,]

colnames(EuroTrain0)
# Check the structure
str(EuroTrain0)
str(EuroTest0)

# Check for missing values in the relevant variables
colSums(is.na(EuroTrain0))
colSums(is.na(EuroTest0))

# Make sure GroupStage is treated as categorical
EuroTrain0$GroupStage <- as.factor(EuroTrain0$GroupStage)
EuroTest0$GroupStage  <- as.factor(EuroTest0$GroupStage)

model_pois <- glm(
  Goals ~ Year + GroupStage + GDP + MarketValue +
    FifaRank + UefaPoints + CLPlayers,
  family = poisson(link = "log"),
  data = EuroTrain0
)

# Examine the model
summary(model_pois)

# Predict expected number of goals
predictions <- predict(
  model_pois,
  newdata = EuroTest0,
  type = "response"
)

# Inspect predictions
head(predictions)

# Mean Absolute Error
MAE <- mean(abs(EuroTest0$Goals - predictions), na.rm = TRUE)

# Root Mean Squared Error
RMSE <- sqrt(mean((EuroTest0$Goals - predictions)^2, na.rm = TRUE))

MAE
RMSE

dispersion <- sum(residuals(model_pois, type = "pearson")^2) /
  df.residual(model_pois)

dispersion

# Baseline: predict the training-set mean for every test observation
baseline <- rep(
  mean(EuroTrain0$Goals),
  nrow(EuroTest0)
)

# Compare models
comparison <- data.frame(
  Model = c("Baseline", "Poisson Regression"),
  MAE = c(
    mean(abs(EuroTest0$Goals - baseline)),
    mean(abs(EuroTest0$Goals - predictions))
  ),
  RMSE = c(
    sqrt(mean((EuroTest0$Goals - baseline)^2)),
    sqrt(mean((EuroTest0$Goals - predictions)^2))
  )
)

print(comparison)

# Inspect whether the paired observations reverse the predictors
EuroTest0[1:6, c(
  "Goals", "Team", "Opponent", "Year",
  "GDP", "MarketValue", "FifaRank",
  "UefaPoints", "CLPlayers"
)]

# Model 1: Your original model
model_full <- glm(
  Goals ~ Year + GroupStage + GDP + MarketValue +
    FifaRank + UefaPoints + CLPlayers,
  family = poisson(link = "log"),
  data = EuroTrain0
)

# Model 2: A simpler model
model_simple <- glm(
  Goals ~ Year + MarketValue + FifaRank,
  family = poisson(link = "log"),
  data = EuroTrain0
)

# Predictions on Euro 2024
pred_full <- predict(
  model_full, newdata = EuroTest0, type = "response"
)

pred_simple <- predict(
  model_simple, newdata = EuroTest0, type = "response"
)

# Compare predictive performance
evaluate <- function(actual, predicted) {
  c(
    MAE = mean(abs(actual - predicted)),
    RMSE = sqrt(mean((actual - predicted)^2))
  )
}

rbind(
  Baseline = evaluate(
    EuroTest0$Goals,
    rep(mean(EuroTrain0$Goals), nrow(EuroTest0))
  ),
  Full_Poisson = evaluate(EuroTest0$Goals, pred_full),
  Simple_Poisson = evaluate(EuroTest0$Goals, pred_simple)
)

model_no_year <- glm(
  Goals ~ MarketValue + FifaRank,
  family = poisson(link = "log"),
  data = EuroTrain0
)

pred_no_year <- predict(
  model_no_year,
  newdata = EuroTest0,
  type = "response"
)

evaluate(EuroTest0$Goals, pred_no_year)

# Attach full-model predictions to the test data
EuroTestResults <- EuroTest0
EuroTestResults$PredictedGoals <- pred_full

# Overall prediction bias
mean(EuroTestResults$PredictedGoals - EuroTestResults$Goals)

# Actual and predicted mean goals
c(
  ActualMean = mean(EuroTestResults$Goals),
  PredictedMean = mean(EuroTestResults$PredictedGoals)
)

# Compare actual and predicted goals by observed goal count
aggregate(
  cbind(Goals, PredictedGoals) ~ Goals,
  data = EuroTestResults,
  FUN = mean
)

# Inspect the largest prediction errors
EuroTestResults$Error <-
  EuroTestResults$PredictedGoals - EuroTestResults$Goals

EuroTestResults[
  order(abs(EuroTestResults$Error), decreasing = TRUE),
  c("Team", "Opponent", "Goals", "PredictedGoals", "Error")
][1:10, ]

# Four standard GLM diagnostic plots
par(mfrow = c(2, 2))
plot(model_full)
par(mfrow = c(1, 1))

install.packages("statmod")  # Only if not already installed
library(statmod)

set.seed(123)
rq <- qresiduals(model_full)

qqnorm(
  rq,
  main = "Q-Q Plot of Randomized Quantile Residuals",
  pch = 19,
  cex = 0.7
)

qqline(rq, col = "red", lwd = 2)


ggplot(EuroTrain0, aes(x = Goals)) +
  geom_bar(fill = "steelblue", color = "white") +
  scale_x_continuous(breaks = 0:max(EuroTrain0$Goals)) +
  labs(
    title = "Distribution of Goals Scored",
    subtitle = "Historical European Championship matches",
    x = "Goals scored by team",
    y = "Number of observations"
  ) +
  theme_minimal(base_size = 13)

ggplot(EuroTrain0, aes(x = factor(Year), y = Goals)) +
  geom_boxplot(
    fill = "steelblue",
    alpha = 0.75,
    outlier.color = "firebrick"
  ) +
  labs(
    title = "Distribution of Goals by Tournament Year",
    x = "Tournament year",
    y = "Goals scored by team"
  ) +
  theme_minimal(base_size = 13)

ggplot(EuroTrain0, aes(x = MarketValue, y = Goals)) +
  geom_jitter(
    width = 0,
    height = 0.08,
    alpha = 0.45,
    color = "steelblue"
  ) +
  geom_smooth(
    method = "loess",
    formula = y ~ x,
    color = "firebrick",
    se = TRUE
  ) +
  geom_hline(
    yintercept = mean(EuroTrain0$Goals),
    linetype = "dashed",
    color = "grey40"
  ) +
  labs(
    title = "Relative Market Value and Goals Scored",
    subtitle = "Each observation represents one team's performance in a match",
    x = "Market value difference (team minus opponent)",
    y = "Goals scored"
  ) +
  theme_minimal(base_size = 13)

ggplot(EuroTrain0, aes(x = CLPlayers, y = Goals)) +
  geom_jitter(
    width = 0,
    height = 0.08,
    alpha = 0.45,
    color = "steelblue"
  ) +
  geom_smooth(
    method = "loess",
    formula = y ~ x,
    color = "firebrick",
    se = TRUE
  ) +
  geom_hline(
    yintercept = mean(EuroTrain0$Goals),
    linetype = "dashed",
    color = "grey40"
  ) +
  labs(
    title = "CLPlayers and Goals Scored",
    subtitle = "Each observation represents one team's performance in a match",
    x = "No. of CLPlayers",
    y = "Goals scored"
  ) +
  theme_minimal(base_size = 13)




prediction_data <- data.frame(
  Actual = EuroTest0$Goals,
  Predicted = pred_full
)

ggplot(prediction_data, aes(x = Predicted, y = Actual)) +
  geom_jitter(
    width = 0.03,
    height = 0.08,
    alpha = 0.6,
    color = "steelblue"
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = "firebrick"
  ) +
  labs(
    title = "Actual vs Predicted Goals",
    subtitle = "Poisson regression evaluated on Euro 2024",
    x = "Predicted expected goals",
    y = "Actual goals"
  ) +
  theme_minimal(base_size = 13)


ggplot(prediction_data, aes(x = Predicted, y = Actual)) +
  geom_jitter(
    width = 0.025,
    height = 0,
    alpha = 0.6,
    color = "steelblue"
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = "firebrick"
  ) +
  coord_equal(xlim = c(0, 5), ylim = c(0, 5)) +
  scale_x_continuous(breaks = 0:5) +
  scale_y_continuous(breaks = 0:5) +
  labs(
    title = "Actual vs Predicted Goals",
    subtitle = "Poisson regression evaluated on Euro 2024",
    x = "Predicted expected goals",
    y = "Actual goals"
  ) +
  theme_minimal(base_size = 13)

# Install packages if needed
install.packages("rpart")
install.packages("rpart.plot")

library(rpart)
library(rpart.plot)

# Ensure GroupStage has consistent factor levels
EuroTrain0$GroupStage <- factor(EuroTrain0$GroupStage)
EuroTest0$GroupStage <- factor(
  EuroTest0$GroupStage,
  levels = levels(EuroTrain0$GroupStage)
)

# Fit regression tree
tree_model <- rpart(
  Goals ~ Year + GroupStage + GDP + MarketValue +
    FifaRank + UefaPoints + CLPlayers,
  data = EuroTrain0,
  method = "anova",
  control = rpart.control(cp = 0.01)
)

# View model summary
print(tree_model)

rpart.plot(
  tree_model,
  type = 3,
  extra = 101,
  fallen.leaves = TRUE,
  box.palette = "Blues",
  main = "Regression Tree for Football Goals"
)

pred_tree <- predict(
  tree_model,
  newdata = EuroTest0
)

# Inspect predictions
head(data.frame(
  Team = EuroTest0$Team,
  Opponent = EuroTest0$Opponent,
  ActualGoals = EuroTest0$Goals,
  PredictedGoals = round(pred_tree, 3)
))

evaluate <- function(actual, predicted) {
  c(
    MAE = mean(abs(actual - predicted)),
    RMSE = sqrt(mean((actual - predicted)^2))
  )
}

results <- rbind(
  Baseline = evaluate(
    EuroTest0$Goals,
    rep(mean(EuroTrain0$Goals), nrow(EuroTest0))
  ),
  Regression_Tree = evaluate(
    EuroTest0$Goals,
    pred_tree
  )
)

round(results, 4)

library(ggplot2)

tree_results <- data.frame(
  Actual = EuroTest0$Goals,
  Predicted = pred_tree
)

ggplot(tree_results, aes(x = Predicted, y = Actual)) +
  geom_jitter(
    width = 0.025,
    height = 0,
    alpha = 0.6,
    color = "steelblue"
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = "firebrick"
  ) +
  labs(
    title = "Actual vs Predicted Goals",
    subtitle = "Regression tree evaluated on Euro 2024",
    x = "Predicted goals",
    y = "Actual goals"
  ) +
  theme_minimal(base_size = 13)

printcp(tree_model)

plotcp(tree_model)

tree_model <- rpart(
  Goals ~ GroupStage + GDP + MarketValue +
    FifaRank + UefaPoints + CLPlayers,
  data = EuroTrain0,
  method = "anova",
  control = rpart.control(cp = 0.01)
)

# Display tree
rpart.plot(
  tree_model,
  type = 3,
  extra = 101,
  fallen.leaves = TRUE,
  main = "Regression Tree for Euro Football Goals"
)

pred_tree <- predict(
  tree_model,
  newdata = EuroTest0
)

evaluate <- function(actual, predicted) {
  c(
    MAE = mean(abs(actual - predicted)),
    RMSE = sqrt(mean((actual - predicted)^2))
  )
}

evaluate(EuroTest0$Goals, pred_tree)

tree_errors <- data.frame(
  Team = EuroTest0$Team,
  Opponent = EuroTest0$Opponent,
  Goals = EuroTest0$Goals,
  PredictedGoals = pred_tree
)

tree_errors$Error <- tree_errors$PredictedGoals -
  tree_errors$Goals

# Largest overestimations
head(
  tree_errors[order(-tree_errors$Error), ],
  10
)

# Largest underestimations
head(
  tree_errors[order(tree_errors$Error), ],
  10
)


tree_results <- data.frame(
  Actual = EuroTest0$Goals,
  Predicted = pred_tree
)

ggplot(tree_results, aes(x = Predicted, y = Actual)) +
  geom_jitter(
    width = 0.025,
    height = 0,
    alpha = 0.6,
    color = "steelblue"
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = "firebrick"
  ) +
  labs(
    title = "Actual vs Predicted Goals",
    subtitle = "Regression tree evaluated on Euro 2024",
    x = "Predicted goals",
    y = "Actual goals"
  ) +
  theme_minimal(base_size = 13)

comparison <- rbind(
  Baseline = evaluate(
    EuroTest0$Goals,
    rep(mean(EuroTrain0$Goals), nrow(EuroTest0))
  ),
  Regression_Tree = evaluate(
    EuroTest0$Goals,
    pred_tree
  )
)

round(comparison, 4)

# Display the complexity parameter table
printcp(tree_model)

# Plot cross-validated error against complexity
plotcp(tree_model)

# Prune to the subtree with 3 splits
tree_pruned <- prune(tree_model, cp = 0.021205)

# Plot the pruned tree
rpart.plot(
  tree_pruned,
  type = 3,
  extra = 101,
  fallen.leaves = TRUE,
  main = "Pruned Regression Tree for Euro Football Goals"
)

# Predict on Euro 2024
pred_pruned <- predict(tree_pruned, newdata = EuroTest0)

# Evaluate performance
evaluate(EuroTest0$Goals, pred_pruned)

rbind(
  Baseline = evaluate(
    EuroTest0$Goals,
    rep(mean(EuroTrain0$Goals), nrow(EuroTest0))
  ),
  Original_Tree = evaluate(EuroTest0$Goals, pred_tree),
  Pruned_Tree = evaluate(EuroTest0$Goals, pred_pruned)
)

library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)

tree_default <- rpart(
  Goals ~ GroupStage + GDP + MarketValue +
    FifaRank + UefaPoints + CLPlayers,
  data = EuroTrain0,
  method = "anova"
)

rpart.plot(
  tree_default,
  type = 3,
  extra = 101,
  fallen.leaves = TRUE,
  main = "Default Regression Tree: Euro Football"
)

# Get and sort the raw variable importance scores
importance <- sort(
  tree_default$variable.importance,
  decreasing = TRUE
)

# Convert scores to percentages
importance_pct <- 100 * importance / sum(importance)

# Round for display
importance_pct <- round(importance_pct, 1)

# Increase bottom and left margins; set y-axis to 0–100%
par(mar = c(8, 5, 4, 2) + 0.1)

barplot(
  importance_pct,
  las = 2,
  col = "steelblue",
  main = "Variable Importance in the Regression Tree",
  ylab = "Share of Total Importance (%)",
  ylim = c(0, 100),
  cex.names = 0.85
)

# Predictions
pred_train_default <- predict(tree_default, newdata = EuroTrain0)
pred_test_default  <- predict(tree_default, newdata = EuroTest0)

# Evaluation function
evaluate <- function(actual, predicted) {
  c(
    MAE = mean(abs(actual - predicted)),
    RMSE = sqrt(mean((actual - predicted)^2))
  )
}

default_results <- rbind(
  Training = evaluate(EuroTrain0$Goals, pred_train_default),
  Test = evaluate(EuroTest0$Goals, pred_test_default)
)

round(default_results, 4)

# Define the parameter grid
param_grid <- expand.grid(
  cp = c(0, 0.001, 0.005, 0.01, 0.02, 0.05),
  minsplit = c(2, 5, 10, 20, 40),
  maxdepth = c(1, 2, 3, 5, 7, 10, 15)
)

# Fit and evaluate every combination
grid_results <- lapply(seq_len(nrow(param_grid)), function(i) {
  
  params <- param_grid[i, ]
  
  model <- rpart(
    Goals ~ GroupStage + GDP + MarketValue +
      FifaRank + UefaPoints + CLPlayers,
    data = EuroTrain0,
    method = "anova",
    control = rpart.control(
      cp = params$cp,
      minsplit = params$minsplit,
      maxdepth = params$maxdepth
    )
  )
  
  pred_train <- predict(model, newdata = EuroTrain0)
  pred_test  <- predict(model, newdata = EuroTest0)
  
  data.frame(
    cp = params$cp,
    minsplit = params$minsplit,
    maxdepth = params$maxdepth,
    train_MAE = mean(abs(EuroTrain0$Goals - pred_train)),
    test_MAE = mean(abs(EuroTest0$Goals - pred_test)),
    train_RMSE = sqrt(mean((EuroTrain0$Goals - pred_train)^2)),
    test_RMSE = sqrt(mean((EuroTest0$Goals - pred_test)^2))
  )
})

grid_results <- bind_rows(grid_results)

# Sort by test MAE
grid_results <- grid_results %>%
  arrange(test_MAE)

# Best-performing combination
best_result <- grid_results[1, ]
print(best_result)

# Inspect the 10 best combinations
head(grid_results, 10)

best_tree <- rpart(
  Goals ~ GroupStage + GDP + MarketValue +
    FifaRank + UefaPoints + CLPlayers,
  data = EuroTrain0,
  method = "anova",
  control = rpart.control(
    cp = best_result$cp,
    minsplit = best_result$minsplit,
    maxdepth = best_result$maxdepth
  )
)

rpart.plot(
  best_tree,
  type = 3,
  extra = 101,
  fallen.leaves = TRUE,
  main = "Best Regression Tree: Euro Football"
)

grid_results <- grid_results %>%
  mutate(
    MAE_gap = test_MAE - train_MAE
  )

# Largest train-test MAE gap
most_overfit_result <- grid_results %>%
  slice_max(MAE_gap, n = 1, with_ties = FALSE)

print(most_overfit_result)

overfit_tree <- rpart(
  Goals ~ GroupStage + GDP + MarketValue +
    FifaRank + UefaPoints + CLPlayers,
  data = EuroTrain0,
  method = "anova",
  control = rpart.control(
    cp = most_overfit_result$cp,
    minsplit = most_overfit_result$minsplit,
    maxdepth = most_overfit_result$maxdepth
  )
)

rpart.plot(
  overfit_tree,
  type = 3,
  extra = 101,
  fallen.leaves = TRUE,
  main = "Most Overfitting Regression Tree"
)

depth_grid <- expand.grid(
  maxdepth = c(1, 2, 3, 5, 7, 10, 15)
)

depth_results <- lapply(seq_len(nrow(depth_grid)), function(i) {
  
  depth <- depth_grid$maxdepth[i]
  
  model <- rpart(
    Goals ~ GroupStage + GDP + MarketValue +
      FifaRank + UefaPoints + CLPlayers,
    data = EuroTrain0,
    method = "anova",
    control = rpart.control(
      cp = 0,
      minsplit = 2,
      maxdepth = depth
    )
  )
  
  pred_train <- predict(model, newdata = EuroTrain0)
  pred_test  <- predict(model, newdata = EuroTest0)
  
  data.frame(
    maxdepth = depth,
    Training = mean(abs(EuroTrain0$Goals - pred_train)),
    Test = mean(abs(EuroTest0$Goals - pred_test))
  )
})

depth_results <- bind_rows(depth_results)

# Convert to long format for plotting
depth_long <- bind_rows(
  data.frame(
    maxdepth = depth_results$maxdepth,
    Dataset = "Training",
    MAE = depth_results$Training
  ),
  data.frame(
    maxdepth = depth_results$maxdepth,
    Dataset = "Test",
    MAE = depth_results$Test
  )
)

ggplot(depth_long, aes(x = maxdepth, y = MAE, color = Dataset)) +
  geom_line() +
  geom_point(size = 2) +
  scale_x_continuous(breaks = depth_grid$maxdepth) +
  labs(
    title = "Training and Test MAE vs Tree Depth",
    subtitle = "Regression tree with cp = 0 and minsplit = 2",
    x = "Maximum tree depth",
    y = "Mean Absolute Error",
    color = "Dataset"
  ) +
  theme_minimal(base_size = 13)


set.seed(6769)

# ------------------------------------------------------------
# 1. Define the formula
# ------------------------------------------------------------

tree_formula <- Goals ~ GroupStage + GDP + MarketValue +
  FifaRank + UefaPoints + CLPlayers


# ------------------------------------------------------------
# 2. Fit the current best tree
#    cp = 0, minsplit = 5, maxdepth = 3
# ------------------------------------------------------------

tree_current <- rpart(
  formula = tree_formula,
  data = EuroTrain0,
  method = "anova",
  control = rpart.control(
    cp = 0,
    minsplit = 5,
    maxdepth = 3
  )
)


# ------------------------------------------------------------
# 3. Fit a more stable tree
#    Minimum 10 observations per terminal leaf
# ------------------------------------------------------------

tree_stable <- rpart(
  formula = tree_formula,
  data = EuroTrain0,
  method = "anova",
  control = rpart.control(
    cp = 0,
    minsplit = 20,
    minbucket = 10,
    maxdepth = 3
  )
)


# ------------------------------------------------------------
# 4. Select tree complexity using 10-fold cross-validation
#    Training data only; Euro 2024 test data is not used here
# ------------------------------------------------------------

set.seed(6769)

tree_cv <- rpart(
  formula = tree_formula,
  data = EuroTrain0,
  method = "anova",
  control = rpart.control(
    cp = 0.001,
    minsplit = 20,
    minbucket = 10,
    maxdepth = 3,
    xval = 10
  )
)

# Inspect cross-validation results
printcp(tree_cv)

# Choose cp with the lowest cross-validated error
cp_table <- tree_cv$cptable

best_cp <- cp_table[
  which.min(cp_table[, "xerror"]),
  "CP"
]

cat("\nSelected cp:", best_cp, "\n")

# Refit using the selected cp
tree_cv_best <- rpart(
  formula = tree_formula,
  data = EuroTrain0,
  method = "anova",
  control = rpart.control(
    cp = best_cp,
    minsplit = 20,
    minbucket = 10,
    maxdepth = 3,
    xval = 10
  )
)


# ------------------------------------------------------------
# 5. Calculate leaf sizes for all three trees
# ------------------------------------------------------------

get_leaf_sizes <- function(tree) {
  leaf_rows <- tree$frame$var == "<leaf>"
  
  data.frame(
    Leaf = rownames(tree$frame)[leaf_rows],
    Observations = tree$frame$n[leaf_rows],
    PredictedGoals = round(tree$frame$yval[leaf_rows], 3),
    row.names = NULL
  )
}

cat("\n--- Current tree leaf sizes ---\n")
print(get_leaf_sizes(tree_current))

cat("\n--- Stable tree leaf sizes ---\n")
print(get_leaf_sizes(tree_stable))

cat("\n--- Cross-validated tree leaf sizes ---\n")
print(get_leaf_sizes(tree_cv_best))


# ------------------------------------------------------------
# 6. Define MAE and RMSE
# ------------------------------------------------------------

calculate_metrics <- function(actual, predicted) {
  c(
    MAE = mean(abs(actual - predicted)),
    RMSE = sqrt(mean((actual - predicted)^2))
  )
}


# ------------------------------------------------------------
# 7. Compare training and test performance
# ------------------------------------------------------------

trees <- list(
  "Current tree" = tree_current,
  "Stable tree (minbucket = 10)" = tree_stable,
  "Cross-validated tree" = tree_cv_best
)

results <- do.call(
  rbind,
  lapply(names(trees), function(model_name) {
    
    tree <- trees[[model_name]]
    
    # Predictions on training data
    pred_train <- predict(tree, newdata = EuroTrain0)
    
    # Predictions on Euro 2024 test data
    pred_test <- predict(tree, newdata = EuroTest0)
    
    train_metrics <- calculate_metrics(
      EuroTrain0$Goals,
      pred_train
    )
    
    test_metrics <- calculate_metrics(
      EuroTest0$Goals,
      pred_test
    )
    
    data.frame(
      Model = model_name,
      Train_MAE = unname(train_metrics["MAE"]),
      Train_RMSE = unname(train_metrics["RMSE"]),
      Test_MAE = unname(test_metrics["MAE"]),
      Test_RMSE = unname(test_metrics["RMSE"])
    )
  })
)

# Round results for presentation
results[, -1] <- round(results[, -1], 4)

cat("\n--- Model performance comparison ---\n")
print(results, row.names = FALSE)


# ------------------------------------------------------------
# 8. Plot the trees
# ------------------------------------------------------------

par(mfrow = c(1, 3))

rpart.plot::rpart.plot(
  tree_current,
  main = "Current Tree"
)

rpart.plot::rpart.plot(
  tree_stable,
  main = "Stable Tree"
)

rpart.plot::rpart.plot(
  tree_cv_best,
  main = "Cross-Validated Tree"
)

par(mfrow = c(1, 1))


# ------------------------------------------------------------
# 9. Identify the model with the lowest test error
#    (evaluation only, not model selection)
# ------------------------------------------------------------

cat("\nModel with lowest test MAE:\n")
print(results[which.min(results$Test_MAE), ])

cat("\nModel with lowest test RMSE:\n")
print(results[which.min(results$Test_RMSE), ])