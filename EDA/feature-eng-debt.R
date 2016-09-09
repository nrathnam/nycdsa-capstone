######In this file, we do the feature engineering by multiply DebtRatio with MonthlyIncome to get DebtAmt. Then drop MonthlyIncome and
#####DebtRatio.

setwd('F:/Miaozhi/Academic/Data_Science/Bootcamp/Project_Capstone/nycdsa-capstone')
data = read.csv('./data/cs-training-outlier-f10.csv', header =T)
names(data)
data = data[,-1]

DebtAmt = data$DebtRatio * as.numeric(data$MonthlyIncome)

data = as.data.frame(cbind(data, DebtAmt))
data = data[,-c(5,6)]

write.csv(data,'cs-training-outlier-debt-f09.csv')
