library(rpart)

tennis <- read.csv("D:/Course/Case Studies/SlamsData.csv", stringsAsFactors = TRUE)
tennis$win <- as.factor(tennis$win)     

tree.tennis <- rpart(win ~ rank + points + elo + age.30,
                     data = tennis, method = "class")
print(tree.tennis)
library(rpart.plot)
rpart.plot(tree.tennis)
rpart.plot(tree.tennis, type = 2, extra = 104, box.palette = "GnBu")
tennis$pred.prob <- predict(tree.tennis, type = "prob")[, "1"]
head(tennis$pred.prob)
table(tennis$pred.prob)
tennis$pred.class <- predict(tree.tennis, type = "class")   

tab <- table(Predicted = tennis$pred.class, Actual = tennis$win)
tab
sum(diag(tab)) / sum(tab)

#BOX PLOT
boxplot(pred.prob ~ win, data = tennis,
        names = c("lost (win=0)", "won (win=1)"),
        xlab  = "Actual outcome",
        ylab  = "Predicted winning probability",
        col   = "#A7C5B5", ylim = c(0,1))
abline(h = 0.5, lty = 3)

boxplot(pred.prob ~ win, data = tennis,
        names   = c("lost (win=0)", "won (win=1)"),
        col     = NA, border = "#16324F",
        outline = FALSE, ylim = c(0, 1),
        xlab    = "Actual outcome",
        ylab    = "Predicted winning probability")
abline(h = 0.5, lty = 3, col = "grey60")

# spread the points so overlapping matches become visible
correct <- (tennis$pred.prob > 0.5) == (tennis$win == "1")

points(jitter(as.numeric(tennis$win), amount = 0.32),
       jitter(tennis$pred.prob,       amount = 0.035),
       pch = 16, cex = 0.22,
       col = ifelse(correct, "#2E7D5B55", "#C0392B55"))

legend("center", legend = c("correct", "wrong"),
       pch = 16, col = c("#2E7D5B", "#C0392B"),
       bty = "n", pt.cex = 1.1)


#Q7)
length(unique(predict(tree.tennis, type = "prob")[, "1"]))   # 2

#refit with co=0.001

tree.deep <- rpart(win ~ rank + points + elo + age.30, data = tennis,
                   method = "class", control = rpart.control(cp = 0.001))
print(tree.deep)
rpart.plot(tree.deep, type = 2, extra = 104, tweak = 0.7,
           box.palette = "GnBu", main = "cp = 0.001")

# --- predictions ---
tennis$prob.deep <- predict(tree.deep, type = "prob")[, "1"]
length(unique(tennis$prob.deep))                 # 19
tapply(tennis$prob.deep, tennis$win, summary)    # quartiles per group

# --- plain boxplot ---
boxplot(prob.deep ~ win, data = tennis,
        names = c("lost (win=0)", "won (win=1)"),
        col = "#A7C5B5", border = "#16324F", ylim = c(0, 1),
        xlab = "Actual outcome", ylab = "Predicted winning probability")
abline(h = 0.5, lty = 3, col = "grey60")

# --- jittered version (green/yellow) ---
set.seed(1)
boxplot(prob.deep ~ win, data = tennis,
        names = c("lost (win=0)", "won (win=1)"),
        col = NA, border = "#16324F", outline = FALSE, ylim = c(0, 1),
        xlab = "Actual outcome", ylab = "Predicted winning probability")
abline(h = 0.5, lty = 3, col = "grey60")
correct <- (tennis$prob.deep > 0.5) == (tennis$win == "1")
points(jitter(as.numeric(tennis$win), amount = 0.32),
       jitter(tennis$prob.deep,       amount = 0.012),
       pch = 16, cex = 0.22,
       col = ifelse(correct, "#2E7D5B55", "#E6A80099"))
legend("center", legend = c("correct", "wrong"), pch = 16,
       col = c("#2E7D5B", "#E6A800"), bty = "n", pt.cex = 1.1)


#Q2 Test and training

train <- subset(tennis, year >= 2011 & year <= 2017)   # 3327
test  <- subset(tennis, year >= 2018 & year <= 2024)   # 3293

# "ratio of wrongly predicted matches" = misclassification rate
err <- function(fit, d) mean(predict(fit, newdata = d, type = "class") != d$win)

grid <- expand.grid(cp       = c(0, 0.001, 0.005, 0.01, 0.02, 0.05),
                    minsplit = c(2, 5, 10, 20, 40),
                    maxdepth = c(1, 2, 3, 5, 7, 10, 15))
grid$train.err <- NA; grid$test.err <- NA; grid$leaves <- NA

for (i in seq_len(nrow(grid))) {
  fit <- rpart(win ~ rank + points + elo + age.30, data = train, method = "class",
               control = rpart.control(cp       = grid$cp[i],
                                       minsplit = grid$minsplit[i],
                                       maxdepth = grid$maxdepth[i],
                                       xval     = 0))
  grid$train.err[i] <- err(fit, train)
  grid$test.err[i]  <- err(fit, test)
  grid$leaves[i]    <- sum(fit$frame$var == "<leaf>")
}
grid$gap <- grid$test.err - grid$train.err
grid[which.min(grid$test.err), ]
grid[grid$test.err == min(grid$test.err), ]   # 171 rows
grid[which.max(grid$gap), ]

d8 <- grid[grid$cp == 0 & grid$minsplit == 2, ]
d8 <- d8[order(d8$maxdepth), ]

plot(d8$maxdepth, d8$test.err, type = "b", pch = 16, col = "#C0392B",
     lwd = 2, ylim = c(0, 0.40), xlab = "maxdepth (cp = 0, minsplit = 2)",
     ylab = "Ratio of wrongly predicted matches")
lines(d8$maxdepth, d8$train.err, type = "b", pch = 16, col = "#2E7D5B", lwd = 2)
legend("topleft", c("test", "training"), col = c("#C0392B", "#2E7D5B"), lwd = 2, bty = "n")
