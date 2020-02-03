## MAZY G. CARNEIRO
## 27th January 2020

#This classification problem was treated as a Multiclass classification as there were three classes Draw(D), Lost(L), Won(W)
#The classification performance was measured using the Kappa of each model
#The Gradient Boosting Model gave us a Kappa in the Validation set that was better than the Random Forest model.
#The results for the Test Set (Kappa = 1) seems too high (out of the ordinary) but I couldn't find any errors or bias that could have caused  the models to get such results.

library(data.table) 
library(doParallel)
library(CORElearn)

#Are there any missing values ?
na = c('?','NA', 'null','Null','NULL')

#Set working directory - where the csv files and scripts can be accessed
setwd("/Users/Mazy/Desktop/Black_Box_test/")

#library(data.table)
df = read.csv(file.path('/Users/Mazy/Desktop/Black_Box_test', 'Rugby_Stats-Table_1.csv'),
                              header=TRUE,
                              sep=',', 
                              stringsAsFactors=TRUE,
                              na.strings=na)
str(df)
dim(df) #8986  199
names(df)

#Input features I think are necessary
df  = subset(df, select=c("id", "Fixture...Date","Competition.ID","Home.Team...ID","Home.Team...Result",
                          "Away.Team...ID","Total_Match_Points","Winning_Margin",
                          "comp_home_win_perc20","comp_home_win_perc40","comp_home_win_perc80",
                          "home_team_win_perc_all_10", "home_team_win_perc_comp_10"             
                                  , "home_team_win_perc_home_10"             
                                  , "away_team_win_perc_all_10"              
                                  , "away_team_win_perc_comp_10"             
                                  , "away_team_win_perc_away_10"             
                                  , "home_team_win_perc_all_20"              
                                  , "home_team_win_perc_comp_20"             
                                  , "home_team_win_perc_home_20"             
                                  , "away_team_win_perc_all_20"              
                                  , "away_team_win_perc_comp_20"             
                                  , "away_team_win_perc_away_20"             
                                  , "home_team_win_perc_all_30"              
                                  , "home_team_win_perc_comp_30"             
                                  , "home_team_win_perc_home_30"             
                                  , "away_team_win_perc_all_30"              
                                  , "away_team_win_perc_comp_30"             
                                  , "away_team_win_perc_away_30" ))

dim(df) #8986   29
str(df)

#Levels in our target column
unique(df$Home.Team...Result)
df$Home.Team...Result = as.factor(df$Home.Team...Result)

levels(df$Home.Team...Result)
table(df$Home.Team...Result)

df = data.frame(df$Home.Team...Result, df)
df$Home.Team...Result = NULL
names(df)[1] = "Home_TEAM_RESULT";names(df)

dim(df) #8986 observations  29 features
rowSums(is.na(df))
sum(is.na(df))

#These columns are shown as numericals but are being changed to factors since each have levels in them
names = c("id", "Fixture...Date","Competition.ID","Home.Team...ID","Away.Team...ID")

df[,names] <- lapply(df[,names] , factor)
str(df)

#Parallel Process - library(doParallel)
cl = makeCluster(detectCores()); cl
registerDoParallel(cl)

# Are there any features that have Zero Variance ?
set.seed(3291)
library(caret)
nearZeroVar(df, names = TRUE) 
#df = df[,-nearZeroVar(df)]

dim(df) #8986   29

#------------------ Dummification -----------------------------
set.seed(3291)

dummifing_ = dummyVars("Home_TEAM_RESULT~. - Fixture...Date -id -Competition.ID -Home.Team...ID -Away.Team...ID", data = df)
dummified_ = predict(dummifing_, newdata = df)
summary(dummified_)
dim(dummified_) #8986 31
# Besides the factors that are mentioned not to be dummified by the dummyVars function there are no other factors 
dummified_df = data.frame(df$Home_TEAM_RESULT,
                          df$Fixture...Date,
                          df$id,
                          df$Competition.ID,
                          df$Home.Team...ID,
                          df$Away.Team...ID, dummified_)
str(dummified_df)
dim(dummified_df) #8986  31

#Renaming features
names(dummified_df)[1] = "Home_TEAM_RESULT"
#Home_TEAM_RESULT is moved to the first column from the last column
names(dummified_df)[2] = "Fixture...Date"
names(dummified_df)[3] = "id"
names(dummified_df)[4] = "Competition.ID"
names(dummified_df)[5] = "Home.Team...ID"
names(dummified_df)[6] = "Away.Team...ID"
str(dummified_df)

#................................CENTERING, SCALING & IMPUTATION ALL THE COLUMNS.....................

#Centering and scaling the features and knnImpute for Imputation .
sum(is.na(dummified_df))

set.seed(3291)

df_cent_sca_preprocessing = preProcess(dummified_df, method = c("center","scale","knnImpute"))
df_cent_sca = predict(df_cent_sca_preprocessing, dummified_df)
anyNA(df_cent_sca)
sum(is.na(df_cent_sca))
summary(df_cent_sca)
str(df_cent_sca)
dim(df_cent_sca) #8986 29

#.......................TRAINING AND TESTING DATASETS..............................

#Splitting the data into training and testing
set.seed(1791)
df_TRAIN = df_cent_sca[1:6774,];dim(df_TRAIN) #6774 observations 29 columns
TEST_dataset = df_cent_sca[6775:8986,];dim(TEST_dataset) #2212 observations 29 columns
#Train set has observations before 01/09/18 & the Test set has observations after 01/09/18

#.......................SPLITTING TRAINING SET INTO TRAINING AND VALIDATING SETS .......................
# Shuffle the data
split_data = df_TRAIN[sample(nrow(df_TRAIN)),]

#Splitting the data into training and validation
split = createDataPartition(df_TRAIN$Home_TEAM_RESULT, p = 0.70)[[1]]
df_TRAIN = split_data[split,]; dim(df_TRAIN) #4743  29
VALIDATION_dataset = split_data[-split,]; dim(VALIDATION_dataset) #2031  29

#.......................BALANCING THE TRAINING SET ONLY................................
# Balance the classes in the TRAIN dataset only 

prop.table(table(df_TRAIN$Home_TEAM_RESULT))  # returns %propotion 0f each class

#ploting the class proportion for Draw, Lost and Won
barplot(prop.table(table(df_TRAIN$Home_TEAM_RESULT)), col = rainbow(2), ylim = c(0,0.7), main = 'Class Distribution')  #plots the imbalanced classes
length(df_TRAIN$Home_TEAM_RESULT)

#using UPSAMPLING
#https://topepo.github.io/caret/subsampling-for-class-imbalances.html
#?upSample
df_trainBalanced = upSample(df_TRAIN, df_TRAIN$Home_TEAM_RESULT, list = FALSE) #upsamples the minority classes
head(df_trainBalanced)
dim(df_trainBalanced) #8712  32
summary(df_trainBalanced)

prop.table(table(df_trainBalanced$Class))  # returns %propotion 0f each balanced class    
barplot(prop.table(table(df_trainBalanced$Class)), col = rainbow(2), ylim = c(0,0.7), main = 'Class Distribution') #plots the balanced labels
head(df_trainBalanced$Class)

#Since a duplicate column is created called CLASS as more samples were added to the minority classes
#one will be deleted, they have the sames values
table(df_trainBalanced$Class) #New column created due to upSampling
table(df_trainBalanced$Home_TEAM_RESULT) #Original 

#Removing Class
TRAIN_dataset = df_trainBalanced
TRAIN_dataset$Class = NULL;names(TRAIN_dataset)
dim(TRAIN_dataset)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
set.seed(3291)
require(compiler)

multiClassSummary = cmpfun(function (data, lev = NULL, model = NULL){
  
  #Load Libraries
  require(Metrics)
  require(caret)
  
  #Check data
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
    stop("levels of observed and predicted data do not match")
  
  #Calculate custom one-vs-all stats for each class
  prob_stats <- lapply(levels(data[, "pred"]), function(class){
    
    #Grab one-vs-all data for the class
    pred <- ifelse(data[, "pred"] == class, 1, 0)
    obs  <- ifelse(data[,  "obs"] == class, 1, 0)
    prob <- data[,class]
    
    #Calculate one-vs-all AUC and logLoss and return
    cap_prob <- pmin(pmax(prob, .000001), .999999)
    prob_stats <- c(auc(obs, prob), logLoss(obs, cap_prob))
    names(prob_stats) <- c('Kappa', 'logLoss')
    return(prob_stats) 
  })
  prob_stats <- do.call(rbind, prob_stats)
  rownames(prob_stats) <- paste('Class:', levels(data[, "pred"]))
  
  #Calculate confusion matrix-based statistics
  CM <- confusionMatrix(data[, "pred"], data[, "obs"])
  
  #Aggregate and average class-wise stats
  #Todo: add weights
  class_stats <- cbind(CM$byClass, prob_stats)
  class_stats <- colMeans(class_stats)
  
  #Aggregate overall stats
  overall_stats <- c(CM$overall)
  
  #Combine overall with class-wise stats and remove some stats we don't want 
  stats <- c(overall_stats, class_stats)
  stats <- stats[! names(stats) %in% c('AccuracyNull', 
                                       'Prevalence', 'Detection Prevalence')]
  
  #Clean names and return
  names(stats) <- gsub('[[:blank:]]+', '_', names(stats))
  return(stats)
  
})

#https://www.r-bloggers.com/error-metrics-for-multi-class-problems-in-r-beyond-accuracy-and-kappa/

anyNA(TRAIN_dataset)
anyNA(VALIDATION_dataset)
#--------

ctrl = trainControl(method='cv',                 #  Cross Validation
                    classProbs=TRUE,             # compute class probabilities
                    summaryFunction = multiClassSummary,
                    number = 5,                  #5 folds
                    verboseIter = TRUE,
                    savePredictions = TRUE)

table(TRAIN_dataset$Home_TEAM_RESULT)
table (VALIDATION_dataset$Home_TEAM_RESULT)

#--------------------------------------------------------GRADIENT BOOSTING------------------------------------------------
set.seed(3291)

gbmGrid = expand.grid(interaction.depth = 6,
                      n.trees = 100,
                      shrinkage = 0.01,
                       n.minobsinnode = 15)

Gbm.model = train(Home_TEAM_RESULT ~.,
                  TRAIN_dataset,
                   method='gbm',
                   trControl = ctrl,
                   verbose = FALSE,
                   tuneGrid = gbmGrid,
                   metric = 'Kappa')

print(Gbm.model) 

head(Gbm.model$pred)
#Confusion matrix
confusionMatrix(data = Gbm.model$pred$pred, 
                reference = Gbm.model$pred$obs)
#Kappa = 1      Accuracy = 1

set.seed(3291)
Gbm.model_Prediction_Classes = predict(Gbm.model, newdata = VALIDATION_dataset)
table (VALIDATION_dataset$Home_TEAM_RESULT)
table(Gbm.model_Prediction_Classes)
confusionMatrix(Gbm.model_Prediction_Classes,VALIDATION_dataset$Home_TEAM_RESULT)
#Kappa = 1     Accuracy : 1

Gbm.model_Prediction_Classes_prob = predict(Gbm.model, newdata = VALIDATION_dataset, type="prob")

#--------------------------------------------------------Random Forest------------------------------------------------
set.seed(3291)

rf_grid = expand.grid(.mtry=c(5,10,25,75,100))

rf.model = train(Home_TEAM_RESULT ~.,
                 TRAIN_dataset,
                 method='rf',
                 trControl = ctrl,
                 verbose = FALSE,
                 ntree = 100,
                 metric = 'Kappa',
                 keep.inbag = TRUE,
                 tuneGrid = rf_grid,
                 allowParallel = TRUE)

print(rf.model)
ggplot(rf.model) + theme_bw()

head(rf.model$pred)
#Confusion matrix
confusionMatrix(data = rf.model$pred$pred,
                reference = rf.model$pred$obs)
#Kappa =  0.5994         Accuracy :0.7329 

set.seed(3291)
rf.model_Prediction_Classes = predict(rf.model, newdata = VALIDATION_dataset)
table (VALIDATION_dataset$Home_TEAM_RESULT)
table(rf.model_Prediction_Classes)
confusionMatrix(rf.model_Prediction_Classes,VALIDATION_dataset$Home_TEAM_RESULT)
#Kappa = 0.6665      Accuracy : 0.8385 

rf.model_Prediction_Classes_prob = predict(rf.model, newdata = VALIDATION_dataset, type="prob")

#******* SINCE GBM GAVE US BETTER RESULTS, THE GBM MODEL WILL BE TESTED ON THE TEST DATASET

# ----------- PREDICTION ON TEST SET

set.seed(3291)
Test_Gbm.model_Prediction_Classes = predict(Gbm.model, newdata = TEST_dataset)
table (TEST_dataset$Home_TEAM_RESULT)
table(Test_Gbm.model_Prediction_Classes)
confusionMatrix(Test_Gbm.model_Prediction_Classes,TEST_dataset$Home_TEAM_RESULT)    
#Kappa : 1   Accuracy : 1

Test_Gbm.model_Prediction_Classes_prob = predict(Gbm.model, newdata = TEST_dataset, type="prob")

#Probability of the Home team winning
winning_probability  = Test_Gbm.model_Prediction_Classes_prob$W

#Creating a csv filr with the required columns along with the Probability Of Home Win.
write.table(cbind(TEST_dataset$Fixture...Date, TEST_dataset$id, TEST_dataset$Competition.ID,TEST_dataset$Home.Team...ID,TEST_dataset$Away.Team...ID,winning_probability), 
            file="Rugby_predictions.csv",row.names=F,col.names=c("Fixture Date", "Fixture ID", "Competition ID","Home Team ID", "Away Team ID", "Probability Of Home Win"),sep = '\t')

stopCluster(cl)


