### Theme 3 


#import the data

data <- read.csv("C:\\Users\\xrusa\\OneDrive\\Υπολογιστής\\IMDBDataset.csv",sep=",",header=T)
str(data)

#libraries needed for naive bayes

library(e1071)

library(tm)

library(SnowballC)



library(stringr)

library(gmodels)

library(caret)

data$sentiment<-str_replace_all(data$sentiment, "[^[:alnum:] ]", "")
data$review<-str_replace_all(data$review, "[^[:alnum:] ]", "")


#This way, we input the review and determine whether it is a positive or negative comment.

data$sentiment <- as.factor(data$sentiment) 
data$review<-as.factor(data$review) 

#How many positive or negative comments are there?
table(data$sentiment)




data_corpus <- VCorpus(VectorSource(data$review))


data_dtm <- DocumentTermMatrix(data_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

dim(data_dtm)

#document term matrix
data_dtm<-removeSparseTerms(data_dtm, sparse = 0.9989)
data_dtm <-as.matrix(data_dtm)


#train-test data separation
data_dtm_train <- data_dtm[1:40000, ]
data_dtm_test  <- data_dtm[40001:50000, ]



#labels for train-test data

#Positive or negative comment for each review. 
data_train_labels <- data[1:40000, ]$sentiment
data_test_labels  <- data[40001:50000, ]$sentiment

#How much % I have positive and negative in the training data as well as in the test data
prop.table(table(data_train_labels))
prop.table(table(data_test_labels))




convert_counts <- function(x) {
  result <- ifelse(x > 0, "Yes", "No")
  return(result)
}
train_dtm_binary <- apply(data_dtm_train, MARGIN = 2, convert_counts)
test_dtm_binary  <- apply(data_dtm_test, MARGIN = 2, convert_counts)
data_classifier <- naiveBayes(as.matrix(train_dtm_binary), data_train_labels)
data_test_pred <- predict(data_classifier, as.matrix(test_dtm_binary))


#accuracy

data_test_labels<-as.factor(data_test_labels)
accuracy <- confusionMatrix(data_test_pred,data_test_labels)
accuracy