library(GA)

#Ackley function
f1 <- function(x,y)
{
	return(-20*exp(-0.2*sqrt(0.5*(x*x+y*y)))- exp(0.5*(cos(2*pi*x)+cos(2*pi*y)))+20)
}

ga1 = ga(type="real-valued",fitness=function(x) -f1(x[1],x[2]),lower=c(-5,-5),upper=c(5,5),maxiter=500,optim=T)


#Rosenbrock function
f2 <- function(x,y)
{
	a = 1
	b = 100
	return((a - x*x)^2 + b * (y - x*x)^2)
}

ga2 = ga(type="real-valued",fitness = function(x) -f2(x[1],x[2]),lower=c(-5,-5),upper=c(5,5),maxiter=100,optim=T)

#Beale function
f3 <- function(x,y)
{
	return((1.5 - x + x*y)^2 + (2.25 - x + x*y*y)^2 + (2.625 - x + x*y^3)^2)
}

ga3 = ga(type="real-valued",fitness = function(x) -f3(x[1],x[2]),lower=c(-4.5,-4.5),upper=c(4.5,4.5),maxiter=100,optim=T)

#Booth function
f4 <- function(x,y)
{
	(x + 2 * y - 7)^2 + (2 * x + y - 5)^2
}

ga4 = ga(type="real-valued",fitness=function(x) -f4(x[1],x[2]),lower=c(-10,-10),upper=c(10,10),maxiter=100,optim=T)#,nBits=8)


#timepass mulivariate function
f5 <- function(x,y,z)
{
	(x + 2 * y - 7 + z)^2 + (2 * x + y - 5 -2*z)^2
}

funcval = 
ga5 = ga(type="real-valued",fitness=function(x) -f5(x[1],x[2],x[3]),lower=c(-10,-10,-10),upper=c(10,10,10),maxiter=100,optim=T)#,nBits=8)




#f7 <- function(x){return(x^2)}
#gatp <- ga(type="real-valued",fitness=function(x) f7(x),lower=-10,upper=10,optim=T)


f8 <- function(x,y)
{
	(x - 10)^2 + (y - 20)^2
}

ga8 = ga(type="real-valued",fitness=function(x) -f8(x[1],x[2]),lower=c(13,0),upper=c(100,100),maxiter=100,optim=T)#,nBits=8)

