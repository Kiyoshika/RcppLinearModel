# RcppLinearModel
My first Rcpp package. Building a generalized linear model library to get familiar with Rcpp.
## Example Usage
```R
library(Rcpp)

data = read.csv("dummy.csv") # data is included in Github repo
names(data) = c("x1","x2","x3","y")

library(dplyr)
x = data %>% select(!y)
y = data %>% select(y)

sourceCpp("RcppLinearModel.cpp")

model = linear_model(x, y, max_iter = 200000)
# NOTE: predict_point is zero-indexed, meaning index starts at 0 rather than 1 like typical R
predict_point(model, x, 1) # predict a single point (at index 1) in a data frame
predict(model, x) # run predictions on entire data frame

r_model = lm(y ~ ., data = data)
stats::predict(r_model)

mean(unlist(abs(y - predict(model, x))))
mean(unlist(abs(y - stats::predict(r_model))))
```
