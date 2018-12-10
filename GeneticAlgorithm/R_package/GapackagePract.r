library(GA)
library(foreach)
library(iterators)
f <- function(x)
{
  return(abs(x) + cos(x))
}
min = -20
max = 20
curve(f,min,max)
fitness <- function(x) f(x)
GA <- ga(type = "real-valued",fitness = fitness,lower = min,upper = max)
plot(GA)
summary(GA)
opt_sol = optimize(f,lower = min,upper = max,maximum = F)