setwd('F:/Miaozhi/Academic/Data_Science/Bootcamp/Project_Capstone/nycdsa-capstone')
data = read.csv('./data/cs-training-outlier-f10.csv', header =T)
names(data)


DepdRisk = (data$NumberOfDependents) / as.numeric(data$MonthlyIncome)
data = as.data.frame(cbind(data, DepdRisk))

data = data[,-c(7,12)]

write.csv(data,'cs-training-depdrisk-f09.csv')

setwd('F:/Miaozhi/Academic/Data_Science/Bootcamp/Project_Capstone/nycdsa-capstone')
test = read.csv('./data/cs-test-outlier-f10.csv', header = T)

DepdRisk = (test$NumberOfDependents) / as.numeric(test$MonthlyIncome)
test = as.data.frame(cbind(test, DepdRisk))

test = test[,-c(7,12)]

write.csv(data,'cs-test-depdrisk-f09.csv')
