library(ggplot2)
library(tidyverse)
library(e1071)
library(caTools)
library(caret)
library(naivebayes)
library(klaR)

df <- read.csv('../../data/01-modified-data/big_five_final.csv')
df <- df %>% dplyr::select(-c(age, case_id, country)) %>%
        mutate(sex = factor(ifelse(sex == 1, 'Male', 'Female')))
df$sex <- as.factor(df$sex)

set.seed(123)

## ref: https://www.learnbymarketing.com/tutorials/naive-bayes-in-r/
split <- sample.split(df, SplitRatio=0.8)
train <- subset(df, split=='TRUE')
test <- subset(df, split=='FALSE')

classifier <- naiveBayes(sex ~ ., data=train)
modelPred <- predict(classifier, test)
confusion <- table(modelPred, test$sex)
confusionMatrix(confusion)

## ref: https://stackoverflow.com/questions/37897252/plot-confusion-matrix-in-r-using-ggplot
Target <- factor(c('Male', 'Male', 'Female', 'Female'))
Prediction <- factor(c(0, 1, 0, 1))
Y      <- c(26241, 14201, 54263, 7733)
df <- data.frame(Target, Prediction, Y)
ggplot(df, aes(x = Target, y = Prediction)) +
        geom_tile(aes(fill = Y), color = "white") +
        geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
        scale_fill_gradient(low = "blue", high = "red") +
        scale_x_discrete(labels=c("Female","Male")) +
        scale_y_discrete(labels=c("Female","Male")) +
        theme_bw() + 
        theme(legend.position = "none")

klar <- NaiveBayes(sex ~ ., data = train)
plot(klar, vars = c("agreeable_score", "extraversion_score", "openness_score", "conscientiousness_score", "neuroticism_score"))
