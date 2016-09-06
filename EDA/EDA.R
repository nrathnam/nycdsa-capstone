setwd('F:/Miaozhi/Academic/Data_Science/Bootcamp/Project_Capstone/nycdsa-capstone')
data = read.csv('./data/cs-training.csv', header =T)
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

####Plot missing value####
aggr(data)

####Correltion plot####
corrplot(cor(data), method="circle", type = "upper", order="hclust",
         col=colorRampPalette(brewer.pal(11,"Spectral"))(8))

####PCA#####
fa.parallel(data, #The data in question.
            fa = "pc", #Display the eigenvalues for PCA.
            n.iter = 100) #Number of simulated analyses to perform.
abline(h = 1) #Adding a horizontal line at 1.
pc_bodies = principal(data, #The data in question.
                      nfactors = 5, #The number of PCs to extract.
                      rotate = "none")
pc_bodies