
## Red And White project (HavardX)
# Normaly 10 mn to execute this program.
# The dataset was downloaded from the UCI Machine Learning Repository :
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/.  

#Loading and cleaning Dataset  

# Loading white wine dataset
white_url <- 
"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
white <- read.csv(white_url, header = TRUE, sep = ";")
# Loading red wine dataset
red_url <- 
"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
red <- read.csv(red_url, header = TRUE, sep = ";")
#-------------------------------------------------

# Verify required packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lattice)) install.packages("lattice", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(klaR)) install.packages("klaR", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")

# Loading library  

# --------------------------------------------------------------------------------
library(tidyverse)
library(caret)
library(lattice)
library(GGally)
library(e1071)
library(MASS)
library(klaR)
library(kernlab)
# --------------------------------------------------------------------------------


# Checking for any missing (NA) values.  


white <- white[!duplicated(unique(white)), ]
red <- red[!duplicated(unique(red)), ]

# We add categorical variable **type** to both sets.   
red['type'] <- 'red'
white['type'] <- 'white'

# --------------------------------------------------------------------------------
### From regression to binary classification  

# We create another categorical variable, classifying the wines as **bad** (rating 0 to 5), and **good** (rating 6 to 10).  

# red wine
good <- red$quality >= 6
bad <- red$quality < 6
red[good, 'quality'] <- 'good'
red[bad, 'quality'] <- 'bad'  
red$quality <- as.factor(red$quality)


# white win  
good <- white$quality >= 6
bad <- white$quality < 6
white[good, 'quality'] <- 'good'
white[bad, 'quality'] <- 'bad'  
white$quality <- as.factor(white$quality)



# The two datasets **red** and  **white** are joined into one larger dataset  **wine**.  

wine <- rbind(red, white)

# We delete type variable in dataset red and white.  
red$type <- NULL
white$type <- NULL
# --------------------------------------------------------------------------------

# Dimension of total wine  
dim(wine)

# Dimension of red wine
wine %>% filter(type=="red")  %>% dim

# Dimension of white wine 
wine %>% filter(type=="white")  %>% dim

# Table preview
glimpse(wine)

# Distribution Graph of white wine  
oldpar = par(mfrow = c(2,3))
for ( i in 1:11 ) {
  truehist(white[[i]], xlab = names(white)[i], col = 'chocolate1', 
           main = paste("Average =", signif(mean(white[[i]]),3)))
}
# --------------------------------------------------------------------------------
#Distribution Graph of red wine  
oldpar = par(mfrow = c(2,3))
for ( i in 1:11 ) {
  truehist(red[[i]], xlab = names(red)[i], col = '#ff0000', 
main = paste("Average =", signif(mean(red[[i]]),3)))
}

# white wine boxplot
oldpar = par(mfrow = c(1,6))
for ( i in 1:11 ) {
  boxplot(white[[i]])
  mtext(names(white)[i], cex = 1, side = 3, line = 0)
}

# red wine boxplot
oldpar = par(mfrow = c(1,6))
for ( i in 1:11 ) {
  boxplot(red[[i]])
  mtext(names(red)[i], cex = 1, side = 3, line = 0)
}
# --------------------------------------------------------------------------------
# White wine and quality  

white[1:12] %>% 
  gather(key = "Variable", value = "Value", `fixed.acidity`:alcohol) %>% 
  ggplot(aes(x = quality, y = Value)) +
  geom_jitter(alpha = 0.005)  +
  geom_boxplot(aes(group = cut_width(quality, 1)), fill = "blue") +
  facet_wrap(~ Variable, scales = "free") +
  guides(fill = TRUE)

# correlation between quality and alcohol for white wine
cor.test(as.numeric(white$quality), white$alcohol, 
         method = 'pearson')

# Red wine  and quality  
red[1:12] %>% 
  gather(key = "Variable", value = "Value", `fixed.acidity`:alcohol) %>% 
  ggplot(aes(x = quality, y = Value)) +
  geom_jitter(alpha = 0.005)  +
  geom_boxplot(aes(group = cut_width(quality, 1)), fill = "red") +
  facet_wrap(~ Variable, scales = "free") +
  guides(fill = TRUE)

# correlation between quality and alcohol for red wine
cor.test(as.numeric(red$quality), red$alcohol, method = 'pearson')

# --------------------------------------------------------------------------------
# Here correlation for red wine  
ggcorr(red, label = TRUE, label_round = 2,label_size = 3,size = 2)

# Here correlation for white wine
ggcorr(white, label = TRUE, label_round = 2,label_size = 3,size = 2)

# --------------------------------------------------------------------------------
# cross validation we use 5-fold
ctrl <- trainControl(method = 'cv', number = 5)

# initialize seed to 1
set.seed(1, sample.kind="Rounding")
# --------------------------------------------------------------------------------
# red wine split
redSplit <- createDataPartition(red$quality,p = 0.75,list = FALSE)
redTrain <- red[redSplit,]
redTest  <- red[-redSplit,]

# white wine split
whiteSplit <- createDataPartition(white$quality,p = 0.75,list = FALSE)
whiteTrain <- white[whiteSplit,]
whiteTest  <- white[-whiteSplit,]

# --------------------------------------------------------------------------------
# Feature elimination for red with Backward selection  
fitCtrl_redrfe <- rfeControl(functions = rfFuncs, method = 'cv', number = 5) 
fit_redrfe <- rfe(quality ~., data = redTrain,
                  sizes = c(1:10), 
                  rfeControl = fitCtrl_redrfe)
features_red <- predictors(fit_redrfe) 

# accuracy for features
max(fit_redrfe$results$Accuracy)

# Recursive Feature Elimination for red wine with 5-fold CV
plot(fit_redrfe, type = c('g', 'o'), main = 'Recursive Feature Elimination')

# The new features for red wine :   
features_red
# --------------------------------------------------------------------------------
# Recursive Feature elimination for white wine with 5-fold CV Graph 
# very slow 5 mn

fitCtrl_whiterfe <- rfeControl(functions = rfFuncs, method = 'cv', number = 5) 
fit_whiterfe <- rfe(quality ~., data = whiteTrain,
                    sizes = c(1:10),  
                    rfeControl = fitCtrl_whiterfe)
features_white <- predictors(fit_whiterfe) 

max(fit_whiterfe$results$Accuracy)
plot(fit_whiterfe, type = c('g', 'o'), main = 'Recursive Feature Elimination')

# The new features for white wine :  
  features_white  

# --------------------------------------------------------------------------------
  
# We generate a Naive Bayes model, using 5-fold cross-validation for red wine  

fit_rednaive <- train(x = redTrain[, features_red], y = redTrain$quality,
                method ="nb",
                trControl = ctrl)
predict_rednaive <- predict(fit_rednaive, newdata = redTest[, features_red])
confMat_rednaive <- confusionMatrix(predict_rednaive, redTest$quality, positive = 'good')
importance_rednaive <- varImp(fit_rednaive, scale = TRUE)
plot(importance_rednaive, main = 'Feature importance for Naive Bayes')
accuracy_bn_red <-max(fit_rednaive$results$Accuracy)

# accuracy Naive Bayes for red wine
accuracy_results_red <- data.frame(Method = "Naive Bayes red variables", 
 Accuracy = accuracy_bn_red)
accuracy_results_red 

# We generate a Naive Bayes model, using 5-fold cross-validation for white wine 
fit_whitenaive <- train(x = whiteTrain[, features_white], y = whiteTrain$quality,
                method ="nb",
                trControl = ctrl)
predict_whitenaive <- predict(fit_whitenaive, newdata = whiteTest[, features_white])
confMat_whitenaive <- confusionMatrix(predict_whitenaive, whiteTest$quality, positive = 'good')
importance_whitenaive <- varImp(fit_whitenaive, scale = TRUE)
plot(importance_whitenaive, main = 'Feature importance for Naive Bayes')
accuracy_bn_white <- max(fit_whitenaive$results$Accuracy)

# accuracy Naive Bayes for white wine
accuracy_results_white <- data.frame(Method = "Naive Bayes white variables", 
Accuracy = accuracy_bn_white)
accuracy_results_white   
# --------------------------------------------------------------------------------
# Logistic Regression red wine

fit_redglm <- train(x = redTrain[, features_red], y = redTrain$quality,
                 method = 'glm',
                 preProcess = 'range', 
                 trControl = ctrl) 
predict_redglm <- predict(fit_redglm, newdata = redTest[, features_red])
confMat_redglm <- confusionMatrix(predict_redglm, redTest$quality, positive = 'good')
importance_redglm <- varImp(fit_redglm, scale = TRUE)

plot(importance_redglm, main = 'Feature importance for Logistic Regression Red wine')
accuracy_glm_red <- max(fit_redglm$results$Accuracy)  

# Accuracy Logistic Regression red wine
accuracy_results_red <- bind_rows(accuracy_results_red,
data_frame(Method=" Logistic Regression Model Red wine",Accuracy = accuracy_glm_red )) 
accuracy_results_red  

# Logistic Regression white wine

fit_whiteglm <- train(x = whiteTrain[, features_white], y = whiteTrain$quality,
                 method = 'glm',
                 preProcess = 'range',
                 trControl = ctrl) 
predict_whiteglm <- predict(fit_whiteglm, newdata = whiteTest[, features_white])
confMat_whiteglm <- confusionMatrix(predict_whiteglm, whiteTest$quality, positive = 'good')
importance_whiteglm <- varImp(fit_whiteglm, scale = TRUE)

plot(importance_whiteglm, main = 'Feature importance for Logistic Regression White wine')
accuracy_glm_white <- max(fit_whiteglm$results$Accuracy)  

# Accuracy Logistic Regression white wine
accuracy_results_white <- bind_rows(accuracy_results_white,
data_frame(Method=" Logistic Regression Model White wine",Accuracy = accuracy_glm_white )) 
accuracy_results_white    

# --------------------------------------------------------------------------------

# K Nearest Neighbors (KNN) red wine
fit_redknn <- train(x = redTrain[, features_red], y = redTrain$quality,
        method = 'knn',
        preProcess = 'range', 
        trControl = ctrl, 
        tuneGrid = expand.grid(.k = c(3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51, 75, 101)))  
predict_redknn <- predict(fit_redknn, newdata = redTest[, features_red])
confMat_redknn <- confusionMatrix(predict_redknn, redTest$quality, positive = 'good')
importance_redknn <- varImp(fit_redknn, scale = TRUE)

plot(importance_redknn, main = 'Feature importance for K Nearest Neighbors Red wine')
accuracy_knn_red <- max(fit_redknn$results$Accuracy)

# best k nearest neighbors for red wine
fit_redknn$bestTune

# Accuracy knn for red wine
accuracy_results_red <- bind_rows(accuracy_results_red,
            data_frame(Method=" KNN Model Red wine",Accuracy = accuracy_knn_red )) 
accuracy_results_red  


# KNN white wine
fit_whiteknn <- train(x = whiteTrain[, features_white], y = whiteTrain$quality,
                      method = 'knn',
                      preProcess = 'range', 
                      trControl = ctrl, 
                      tuneGrid = expand.grid(.k = c(3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51, 75, 101)))  
predict_whiteknn <- predict(fit_whiteknn, newdata = whiteTest[, features_white])
confMat_whiteknn <- confusionMatrix(predict_whiteknn, whiteTest$quality, positive = 'good')
importance_whiteknn <- varImp(fit_whiteknn, scale = TRUE)
# Feature importance for K Nearest Neighbors white wine
plot(importance_whiteknn, main = 'Feature importance for K Nearest Neighbors white wine')
accuracy_knn_white <- max(fit_whiteknn$results$Accuracy)

# best k nearest neighbors for white wine
fit_whiteknn$bestTune

# Accuracy knn for white wine
accuracy_results_white <- bind_rows(accuracy_results_white,
data_frame(Method=" KNN Model white wine",Accuracy = accuracy_knn_white )) 
accuracy_results_white  
# --------------------------------------------------------------------------------

# Random Forest (RF) for red wine 
fit_redrf <- train(x = redTrain[, features_red], y = redTrain$quality,
                   method = 'rf',
                   trControl = ctrl,
                   tuneGrid = expand.grid(.mtry = c(2:6)),
                   n.tree = 1000) 
predict_redrf <- predict(fit_redrf, newdata = redTest[, features_red])
confMat_redrf <- confusionMatrix(predict_redrf, redTest$quality, positive = 'good')
importance_redrf <- varImp(fit_redrf, scale = TRUE)
# Feature importance for Random Forest Red wine
plot(importance_redrf, main = 'Feature importance for Random Forest Red wine')
accuracy_rf_red <- max(fit_redrf$results$Accuracy)

# Accuracy Random Forest (RF) for red wine
accuracy_results_red <- bind_rows(accuracy_results_red,
                                  data_frame(Method=" Random Forest Model Red wine",Accuracy = accuracy_rf_red )) 
accuracy_results_red  

# Random Forest white wine
fit_whiterf <- train(x = whiteTrain[, features_white], y = whiteTrain$quality,
                     method = 'rf',
                     trControl = ctrl,
                     tuneGrid = expand.grid(.mtry = c(2:6)),
                     n.tree = 1000) 
predict_whiterf <- predict(fit_whiterf, newdata = whiteTest[, features_white])
confMat_whiterf <- confusionMatrix(predict_whiterf, whiteTest$quality, positive = 'good')
importance_whiterf <- varImp(fit_whiterf, scale = TRUE)

# Feature importance for Random Forest white wine
plot(importance_whiterf, main = 'Feature importance for Random Forest white wine')
accuracy_rf_white <- max(fit_whiterf$results$Accuracy)

# Accuracy Random Forest (RF) for white wine
accuracy_results_white <- bind_rows(accuracy_results_white,
                                    data_frame(Method=" Random Forest Model white wine",Accuracy = accuracy_rf_white )) 
accuracy_results_white  

# --------------------------------------------------------------------------------

# Support Vector Machines with linear kernel (svmLinear)  for red wine
#  very slow 
fit_redsvm <- train(x = redTrain[, features_red], y = redTrain$quality,
                    method = 'svmLinear',
                    preProcess = 'range',
                    trControl = ctrl,
                    tuneGrid = expand.grid(.C = c(0.001, 0.01, 0.1, 1, 10, 100)))
predict_redsvm <- predict(fit_redsvm, newdata = redTest[, features_red])
confMat_redsvm <- confusionMatrix(predict_redsvm, redTest$quality, positive = 'good')
importance_redsvm <- varImp(fit_redsvm, scale = TRUE)

plot(importance_redsvm, main = 'Feature importance for SVM-Linear red wine')
accuracy_svm_red <- max(fit_redsvm$results$Accuracy)

# Accuracy svmLinear  for red wine
accuracy_results_red <- bind_rows(accuracy_results_red,
                  data_frame(Method=" SVM-Linear Model Red wine",Accuracy = accuracy_svm_red )) 
accuracy_results_red  

# SVM Linear white wine
#  very slow 3 to 5 mn
fit_whitesvm <- train(x = whiteTrain[, features_white], y = whiteTrain$quality,
                      method = 'svmLinear',
                      preProcess = 'range',
                      trControl = ctrl,
                      tuneGrid = expand.grid(.C = c(0.001, 0.01, 0.1, 1, 10, 100)))
predict_whitesvm <- predict(fit_whitesvm, newdata = whiteTest[, features_white])
confMat_whitesvm <- confusionMatrix(predict_whitesvm, whiteTest$quality, positive = 'good')
importance_whitesvm <- varImp(fit_whitesvm, scale = TRUE)

plot(importance_whitesvm, main = 'Feature importance for SVM-Linear white wine')
accuracy_svm_white <- max(fit_whitesvm$results$Accuracy)

# Accuracy SVM Linear white wine
accuracy_results_white <- bind_rows(accuracy_results_white,
                                    data_frame(Method=" SVM-Linear Model white wine",Accuracy = accuracy_svm_white )) 
accuracy_results_white  

# --------------------------------------------------------------------------------


# Support Vector Machines with Radial Basis Function (svmRBF) for red wine  
# very slow                                         
fit_redsvmRBF <- train(x = redTrain[, features_red], y = redTrain$quality,
                       method = 'svmRadial',
                       preProcess = 'range',
                       trControl = ctrl,
                       tuneGrid = expand.grid(.C = c(0.001, 0.01, 0.1, 1, 10, 100),
                                              .sigma = c(0.001, 0.01, 0.1)))
predict_redsvmRBF <- predict(fit_redsvmRBF, newdata = redTest[, features_red])
confMat_redsvmRBF <- confusionMatrix(predict_redsvmRBF, redTest$quality, positive = 'good')
importance_redsvmRBF <- varImp(fit_redsvmRBF, scale = TRUE)

plot(importance_redsvmRBF, main = 'Feature importance for SVM-RBF red wine')
accuracy_svmRBF_red <- max(fit_redsvmRBF$results$Accuracy)

# Accuracy svmRBF for red wine
accuracy_results_red <- bind_rows(accuracy_results_red,
                                  data_frame(Method=" SVM-RBF Model Red wine",Accuracy = accuracy_svmRBF_red )) 
accuracy_results_red  

# Support Vector Machines with Radial Basis Function (svmRBF) for white wine
# very slow  3 to 5 mn                                        
fit_whitesvmRBF <- train(x = whiteTrain[, features_white], y = whiteTrain$quality,
                         method = 'svmRadial',
                         preProcess = 'range',
                         trControl = ctrl,
                         tuneGrid = expand.grid(.C = c(0.001, 0.01, 0.1, 1, 10, 100),
                                                .sigma = c(0.001, 0.01, 0.1)))
predict_whitesvmRBF <- predict(fit_whitesvmRBF, newdata = whiteTest[, features_white])
confMat_whitesvmRBF <- confusionMatrix(predict_whitesvmRBF, whiteTest$quality, positive = 'good')
importance_whitesvmRBF <- varImp(fit_whitesvmRBF, scale = TRUE)

plot(importance_whitesvmRBF, main = 'Feature importance for SVM-RBF white wine')
accuracy_svmRBF_white <- max(fit_whitesvmRBF$results$Accuracy)

# Accuracy svmRBF for white wine

accuracy_results_white <- bind_rows(accuracy_results_white,
                                    data_frame(Method=" SVM-RBF Model white wine",Accuracy = accuracy_svmRBF_white )) 
accuracy_results_white  

# --------------------------------------------------------------------------------
# Results  

# Comparring all models for red wine  
models_red <- resamples(list(NB = fit_rednaive, KNN = fit_redknn, GLM = fit_redglm,
                             SVM = fit_redsvm,
                             SVM_RBF = fit_redsvmRBF,
                             RF = fit_redrf))

scales <- list(x=list(relation="free"), y=list(relation="free"))
dotplot(models_red, scales=scales)

# Results Accuracy for red wine :  
accuracy_results_red %>% arrange(desc(Accuracy))

# The best model for red wine is the **Random Forest Model**   
  
#  Comparring all models for white wine  
models_white <- resamples(list(NB = fit_whitenaive, KNN = fit_whiteknn, GLM = fit_whiteglm,
                               SVM = fit_whitesvm,
                               SVM_RBF = fit_whitesvmRBF,
                               RF = fit_whiterf))

scales <- list(x=list(relation="free"), y=list(relation="free"))
dotplot(models_white, scales=scales)

# Results Accuracy for white wine :  

accuracy_results_white %>% arrange(desc(Accuracy))

# The best model for white wine is the **Random Forest Model**  
# --------------------------------------------------------------------------------
# Conclusion  
  
# Random forest is the best model in red and white wine
