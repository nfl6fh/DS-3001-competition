library(dplyr)
library(tidyverse)
library(MLmetrics)
library(mltools)
library(caret)
library(rpart.plot)

df <- read.csv("SpotifyFeatures.csv")

df1 <- unique(df[-c(1,2,3)])
df2 <- df1[-c(1)]

# factorizing character columns
factor_cols <- c(7,10,13)
df2[,factor_cols] <- lapply(df2[,factor_cols], as_factor)

# normalize numeric vectors
# cols_to_normalize = c(4,9,12)

train_index <- createDataPartition(df2$popularity, p=.8,
                                   list=F,
                                   times=1)

train <- df2[train_index,]
test_tune <- df2[-train_index,]

tune_index <- createDataPartition(test_tune$popularity, p=.5,
                                  list=F,
                                  time=1)

tune <- test_tune[tune_index,]
test <- test_tune[-tune_index,]

features <- train[,-1]
target <- train$popularity

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5) 

set.seed(1999)
myDT <- train(x=features,
              y=target,
              method="rpart2",
              trControl=fitControl,
              metric="RMSE")
myDT
plot(myDT)
varImp(myDT)
rpart.plot(myDT$finalModel, type=5, extra=101)

df_pred <- predict(myDT$finalModel, tune)
df_pred
postResample(df_pred, tune$popularity)

linear <- lm(popularity~., df2)
library(MASS)
linear_aic = stepAIC(linear,direction='both',trace=F)
detach('package:MASS')
summary(linear_aic)
