library(ggplot2)
library(tidyverse)
library(e1071)
library(caTools)
library(caret)

df <- read.csv('../../data/01-modified-data/big_five_final.csv')
df <- df %>% dplyr::select(-c(age, case_id, country)) %>%
        mutate(sex = factor(ifelse(sex == 1, 'Male', 'Female')))
df$sex <- as.factor(df$sex)

set.seed(123)

split <- sample.split(df, SplitRatio=0.8)
train <- subset(df, split=='TRUE')
test <- subset(df, split=='FALSE')

classifier <- naiveBayes(sex ~ ., data=train)
modelPred <- predict(classifier, test)
confusion <- table(modelPred, test$sex)
plot(confusion)
confusionMatrix(confusion)

library(naivebayes)
library(klaR)

klar <- NaiveBayes(sex ~ ., data = train)
plot(klar, vars = c("agreeable_score", "extraversion_score", "openness_score", "conscientiousness_score", "neuroticism_score"))

