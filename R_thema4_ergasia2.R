

#### Theme 4


#install.packages("tree")
#install.packages("rpart")
#install.packages("rattle")
#install.packages("rpart.plot")
#install.packages("RColorBrewer")
library(tree)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
data<-read.csv("C:\\Users\\xrusa\\OneDrive\\Υπολογιστής\\agaricus-lepiota.data")
head(data)



set.seed(2) 
# Create training set.
train = sample( 1:nrow(data), nrow(data)*0.8)
#Create testing testing set
test = -train 
training_data = data[train,]
testing_data = data[test,]
# Create the decision tree using the training data. 
tree_model = rpart(p ~ ., data=training_data, method="class")
fancyRpartPlot(tree_model, caption = "Decision Tree plot")






head(testing_data)
tree_predict = predict(tree_model,testing_data,type="class")
testingDataConfusionTable= table(tree_predict, testing_data$p)
print(testingDataConfusionTable)
accuracy <- (sum(diag(testingDataConfusionTable))/sum(testingDataConfusionTable))
error_rate = 1-accuracy
sprintf("The Decision tree accuracy rate is %f and the error rate is %f",accuracy*100,error_rate*100)



### Entrropy Gain###

data2<-data[1:30,]
head(data2)

entropy_entire= - (22/30)*log2(22/30) - (8/30)*log2(8/30)
entropy_entire

unique(data2$u)

Entropy_1 = - (7/11)*log2(7/11) - (4/11)*log2(4/11)
Entropy_1
Entropy_2 = - (12/12)*log2(12/12) - 0
Entropy_2
Entropy_3 = - (2/6)*log2(2/6) - (4/6)*log2(4/6)
Entropy_3
Entropy_4 = - (1/1)*log2(1/1) - 0
Entropy_4
Entropy_split<- 11/30*Entropy_1 + 12/30*Entropy_2 + 6/30*Entropy_3 + 1/30*Entropy_4
Entropy_split
Entropy_gain = entropy_entire - Entropy_split
Entropy_gain


