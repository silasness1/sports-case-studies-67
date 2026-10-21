tennis <- read.csv("D:/Course/Case Studies/SlamsData.csv", stringsAsFactors = TRUE)
str(tennis)
#Understanding the data
#player1 and player2 are arbitarily placed (no home advantage). 
#year= 2011-2024
#tournament=5
#Surface=3
#win= 1 (player1 win) & 0 (player2 win)
#sets.1, sets.2 = sets won by each player. best of 5
#rank (diff in atp ranking) 1= best so negative means player.1 is better
#points( diff in atp points) more points=stronger. positive favours player.1
#elo= diff in elo rating higher elo = stronger
#age.30 = dist from 30 (optimal age)
#prob= avgprob1 - avgprob2
#Avg.prob.1,2= winning probabilities for each match
#Bet365= decimal odds (5.5% margin fee included, avg prob is probably derived from here)
summary(tennis)
colSums(is.na(tennis))  

#Descriptive Statistics

mean(tennis$win)        
table(tennis$win)       

summary(tennis[, c("rank","points","elo","age.30")])
sapply(tennis[, c("rank","points","elo","age.30")], sd)

barplot(table(tennis$win),
        names.arg = c("player.2 wins (0)", "player.1 wins (1)"),
        col = "#3B7CB3", ylab = "Count", main = "Match outcomes")

tapply(tennis$win, tennis$surface,    mean)
tapply(tennis$win, tennis$tournament, mean)

par(mfrow = c(2,2))
boxplot(elo    ~ win, data = tennis, main = "Elo difference by outcome")
boxplot(points ~ win, data = tennis, main = "ATP points difference by outcome")
boxplot(rank   ~ win, data = tennis, main = "ATP rank difference by outcome")
boxplot(age.30 ~ win, data = tennis, main = "Age-from-30 difference by outcome")
par(mfrow = c(1,1))

sapply(tennis[, c("rank","points","elo","age.30")],
       function(x) cor(x, tennis$win))


#Logistic Regression

train <- subset(tennis, year >= 2011 & year <= 2017)
test  <- subset(tennis, year >= 2018 & year <= 2024)
nrow(train)   
nrow(test)    

logit.tennis <- glm(win ~ rank + points + elo + age.30 - 1,
                    data = train, family = "binomial")
summary(logit.tennis)
exp(coef(logit.tennis))

#testdata

# 1. predicted win probability for player.1 in each test match
test$pred.prob <- predict(logit.tennis, newdata = test, type = "response")

# 2. convert probability -> 0/1 prediction at the 0.5 threshold
test$y.pred <- ifelse(test$pred.prob > 0.5, 1, 0)

# 3. confusion matrix: predicted (rows) vs actual (columns)
tab <- table(test$y.pred, test$win)
tab

# 4. accuracy = correct predictions / total
sum(diag(tab)) / sum(tab)