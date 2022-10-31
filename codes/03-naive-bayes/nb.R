library(ggplot2)
library(tidyverse)
library(e1071)
library(caTools)
library(caret)
library(naivebayes)

df <- read.csv('../../data/01-modified-data/big_five_final.csv')
df <- df %>% select(-c(age, case_id, country)) %>%
        mutate(sex = factor(ifelse(sex == 1, 'Male', 'Female')))
df$sex <- as.factor(df$sex)
df_m <- df %>% filter(sex=='Male')
df_f <- df %>% filter(sex=='Female')

set.seed(123)

## ref: https://www.learnbymarketing.com/tutorials/naive-bayes-in-r/
split <- sample.split(df, SplitRatio=0.8)
train_m <- subset(df_m, split=='TRUE')
test_m <- subset(df_m, split=='FALSE')
train_f <- subset(df_f, split=='TRUE')
test_f <- subset(df_f, split=='FALSE')
train <- rbind(train_m, train_f)
test <- rbind(test_m, test_f)

classifier <- naiveBayes(sex ~ ., data=train)
modelPred <- predict(classifier, test)
confusion <- table(modelPred, test$sex)
confusionMatrix(confusion)

## ref: https://stackoverflow.com/questions/37897252/plot-confusion-matrix-in-r-using-ggplot
Target <- factor(c('Male', 'Male', 'Female', 'Female'))
Prediction <- factor(c(0, 1, 0, 1))
Y      <- c(18287, 8375, 39694, 4519)
df <- data.frame(Target, Prediction, Y)
ggplot(df, aes(x = Target, y = Prediction)) +
        geom_tile(aes(fill = Y), color = "white") +
        geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
        scale_fill_gradient(low = "lightblue", high = "salmon") +
        scale_x_discrete(labels=c("Female","Male")) +
        scale_y_discrete(labels=c("Female","Male")) +
        labs(title = 'Confusion Matrix') + 
        theme_bw() + 
        theme(legend.position = "none", plot.title = element_text(hjust = 0.5))

big5 <- naive_bayes(as.factor(sex) ~., data=train)
plot(big5, vars = c("agreeable_score","extraversion_score","openness_score",
                    "conscientiousness_score","neuroticism_score"))
