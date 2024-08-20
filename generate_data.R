#Generate data Game of Life
library(mmand)

#wd
setwd("~/Dropbox/Diego/Profissional/Code/Experiments/Boolean Networks/Workspace")

set.seed(11)
X <- list()
Y <- list()
for(d in seq(0.3,0.5,0.05)){
  print(d)
  for(i in 1:100){
    xi <- gameOfLife(size = c(16,16),density = d,steps = 0)
    for(t in 1:10){
      X <- append(X,list(xi))
      xi <- gameOfLife(init = xi,steps = 1,viz = F)
      Y <- append(Y,list(xi))
    }
  }
}

for(i in 1:length(X)){
  write.csv(x = X[[i]],file = paste("x_GoL_final_",i - 1,".csv",sep = ""),row.names = F)
  write.csv(x = Y[[i]],file = paste("y_GoL_final_",i - 1,".csv",sep = ""),row.names = F)
}