y=football_train$Goals
x=as.matrix(football_train[,-which(names(football_train)=="Goals")])
smtg=glmnet(x,y,alpha = 0.5)
smtg
summary(smtg)
?glmnet

print(smtg)
smtg$beta

smtg$a0
coef(smtg)
x0=as.matrix(football_test[,-which(names(football_test)=="Goals")])
preds=predict(smtg,x0)
preds
residuals=mean(abs(preds- football_test$Goals))
?cv.glmnet
thing=cv.glmnet(x,y,alpha=0.2)
?
  thing
summary(thing)
for (i in c(0,0.2,0.5,0.7,1)){
  fg=cv.glmnet(x,y,alpha=i)
  print(fg)
}

plot(thing)
best_coefs <- coef(thing, s="lambda.min")
best_coefs_matrix=as.matrix(best_coefs)
print(best_coefs_matrix)
estimates=predict(thing,x0)
estimates
plot(estimates,football_test$Goals)
coef(thing)
mean(estimates-1.2)
range(estimates)
table(football_test$Goals)
mean(football_train$Goals)


install.packages("rpart")
library(rpart.plot)
tree=rpart(Goals~GroupStage+GDP+ MarketValue+FifaRank+UefaPoints+CLPlayers,data=football_train)
rpart.plot(tree)
treepreds=predict(tree,football_test[,-which(names(football_test)=="Goals")])
res=treepreds- football_test$Goals
mean(abs(treepreds- football_test$Goals))
mean((treepreds- football_test$Goals)**2)
boxplot(res)
plot(res)
boxplot(treepreds)
?rpart
