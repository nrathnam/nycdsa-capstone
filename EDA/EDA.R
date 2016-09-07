setwd('F:/Miaozhi/Academic/Data_Science/Bootcamp/Project_Capstone/nycdsa-capstone')
data = read.csv('./data/cs-training.csv', header =T)
data = data[ ,-1]
library(VIM)
library(mice)
library(adabag)
library(corrplot)
library(caret)
#library(doMC)
library(neuralnet)
library(nnet)
library(devtools)
library(pROC)
library(clusterGeneration)
library(NeuralNetTools)
library(mice)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(psych)
library(ggpubr)
library(reshape2)


####Plot missing value####
aggr(data)

####Correltion plot####
corrplot(cor(data[complete.cases(data),]), method="circle", type = "lower", order="hclust",
         col=colorRampPalette(brewer.pal(11,"Spectral"))(8))
cor(data)
####PCA#####
fa.parallel(data, #The data in question.
            fa = "pc", #Display the eigenvalues for PCA.
            n.iter = 100) #Number of simulated analyses to perform.
abline(h = 1) #Adding a horizontal line at 1.
pc_bodies = principal(data, #The data in question.
                      nfactors = 5, #The number of PCs to extract.
                      rotate = "none")
pc_bodies

heatmap <- qplot(x=Var1, y=Var2, data=melt(cor(data)), geom="tile",
                 fill=value)

pca <- prcomp(data, scale=T,na.action = na.omit())
melted <- cbind(variable.group, melt(pca$rotation[,1:5]))

barplot <- ggplot(data=data) +
  geom_bar(aes(x=Var1, y=value, fill=variable.group), stat="identity") +
  facet_wrap(~Var2)