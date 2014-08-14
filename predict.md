Classification of Exercises Quality
========================================================
# Introduction

```r
library(dplyr)
url.train <- 'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
url.test <- 'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
df.train <- read.csv(url.train)
df.test <- read.csv(url.test)
```

# Data Preprocessing
First we check the ratio of NAs in each column in the training set and test set.  

```r
table(colMeans(is.na(df.train)))
```

```
## 
##                 0 0.979308938946081 
##                93                67
```

```r
table(colMeans(is.na(df.test)))
```

```
## 
##   0   1 
##  60 100
```
There are 67 columns in training set and 100 columns in test set with very high ratios of NAs.  We further extract the names of test columns and check the difference.


```r
cols.na.train <- names(df.train)[colMeans(is.na(df.train)) > 0.9]
cols.na.test <-  names(df.test)[colMeans(is.na(df.test)) > 0.9]
setdiff(cols.na.train, cols.na.test)
```

```
## character(0)
```

```r
setdiff(cols.na.test, cols.na.train)
```

```
##  [1] "kurtosis_roll_belt"      "kurtosis_picth_belt"    
##  [3] "kurtosis_yaw_belt"       "skewness_roll_belt"     
##  [5] "skewness_roll_belt.1"    "skewness_yaw_belt"      
##  [7] "max_yaw_belt"            "min_yaw_belt"           
##  [9] "amplitude_yaw_belt"      "kurtosis_roll_arm"      
## [11] "kurtosis_picth_arm"      "kurtosis_yaw_arm"       
## [13] "skewness_roll_arm"       "skewness_pitch_arm"     
## [15] "skewness_yaw_arm"        "kurtosis_roll_dumbbell" 
## [17] "kurtosis_picth_dumbbell" "kurtosis_yaw_dumbbell"  
## [19] "skewness_roll_dumbbell"  "skewness_pitch_dumbbell"
## [21] "skewness_yaw_dumbbell"   "max_yaw_dumbbell"       
## [23] "min_yaw_dumbbell"        "amplitude_yaw_dumbbell" 
## [25] "kurtosis_roll_forearm"   "kurtosis_picth_forearm" 
## [27] "kurtosis_yaw_forearm"    "skewness_roll_forearm"  
## [29] "skewness_pitch_forearm"  "skewness_yaw_forearm"   
## [31] "max_yaw_forearm"         "min_yaw_forearm"        
## [33] "amplitude_yaw_forearm"
```
The set cols.na.test is a larger set and contains the set cols.na.train.  Thus we remove columns in cols.na.test.

```r
df.train2 <- df.train[,setdiff(names(df.train),cols.na.test)]
df.test2 <- df.test[,setdiff(names(df.test),cols.na.test)]
dim(df.train2)
```

```
## [1] 19622    60
```

```r
dim(df.test2)
```

```
## [1] 20 60
```

```r
names(df.train2)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```
The first few features include user and time related information.  Based on the experiment, a user did one class of exercise for a period of time, then he moved on to the next class, so on and so forth.  The user and time related information could have strong predictive power but won't be generalized to new data recorded by other users or in the future.  These features should not be used in model training and thus are removed.

```r
names(df.train2)[1:7]
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

```r
df.train2 <- select(df.train2, -(X:num_window))
df.test2 <- select(df.test2, -(X:num_window))
dim(df.train2)
```

```
## [1] 19622    53
```

```r
dim(df.test2)
```

```
## [1] 20 53
```
To evaluate out-of-sample performance, we reserve 40% of data for validation.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
seed = 123
set.seed(seed)
trainIndex <- createDataPartition(df.train2$classe, p = .6,
                                  list = FALSE)
df.train.split <- df.train2[trainIndex,]
df.validate.split <- df.train2[-trainIndex,]
```
For model selection, we choose the random forest model.  There is no need for cross-validation as this model could provide out-of-bag (oob) validation [2].


```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
fit.rf <- randomForest(classe~., df.train.split)
fit.rf
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = df.train.split) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.65%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3343    2    0    1    2    0.001493
## B   11 2263    5    0    0    0.007021
## C    0   19 2032    3    0    0.010711
## D    1    0   23 1905    1    0.012953
## E    0    0    2    7 2156    0.004157
```
Then we apply the model to the validation set to estimate the out-of-sample error accuracy.

```r
pred <- predict(fit.rf, newdata=df.validate.split)
conf <- confusionMatrix(pred, df.validate.split$classe)
conf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229    9    0    0    0
##          B    3 1506   11    0    0
##          C    0    3 1355   11    3
##          D    0    0    2 1275    2
##          E    0    0    0    0 1437
## 
## Overall Statistics
##                                         
##                Accuracy : 0.994         
##                  95% CI : (0.992, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.993         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.992    0.990    0.991    0.997
## Specificity             0.998    0.998    0.997    0.999    1.000
## Pos Pred Value          0.996    0.991    0.988    0.997    1.000
## Neg Pred Value          0.999    0.998    0.998    0.998    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.173    0.163    0.183
## Detection Prevalence    0.285    0.194    0.175    0.163    0.183
## Balanced Accuracy       0.999    0.995    0.994    0.995    0.998
```



