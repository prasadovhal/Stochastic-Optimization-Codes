library(GA)

#hypothetical function with 4 no.of variables
f6 <- function(x,y,z,s)
{
	return((1 - x*x)^2 + 100 * (y - x*x)^2 + 0.5*z - 0.001*s)
}

#GA Settings
fitness = function(x)  -f6(x[1],x[2],x[3],x[4])	# - sign for minimization
type = "real-valued"
UpperLimit = c(10,10,10,10)
LowerLimit = c(-10,-10,-10,-10)
popSize = 100
elite = base::max(1, round(popSize*0.05))
maxiter = 100
bitSize = 32

#GA function
ga6 = ga(type = type,fitness = fitness,popSize = popSize,lower = LowerLimit,upper = UpperLimit,maxiter = maxiter,optim = F,elitism = elite ,keepBest = T,nBits = bitSize)

print(summary(ga6))
plot(ga6)

