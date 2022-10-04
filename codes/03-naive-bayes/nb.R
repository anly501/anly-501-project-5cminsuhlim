library(ggplot2)
library(e1071)
library(caret)
library(cvms)

df <- read.csv('../../data/01-modified-data/big_five_final.csv')
df <- df %>% select(-c(case_id, country))
df$sex <- as.factor(df$sex)

s <-  sort(sample(nrow(df), nrow(df)*.8))
train <- df[s,]
test <- df[-s,]

nb_model <- naiveBayes(sex ~., data=train)

modelPred <- predict(nb_model, test)

confusion <- table(modelPred, test$sex)

plot(confusion)

p1 <- predict(nb_model, train)
tab1 <- table(p, train$admit)
