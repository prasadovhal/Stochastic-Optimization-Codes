V = c(0,10)
gm = 100
t = 1:100
pop = list(c(5.3,4.5),c(6.1,4.9))

#print(pop)
#Non Uniform mutation
gene = sample(1:2,1)

b = 1

delta <-  function(t,y)
{
	Del = y * (1-(runif(1))^(1-(t/gm)))^b
	return(Del)
}

for(i in t)
{
	tau = runif(1)
	selectEx = sample(1:2,1)
	randomChrom = pop[1][[1]]

	if(tau >= 0.5){
		Mutpop = randomChrom[gene] + delta(t,V[1] - randomChrom[gene])
	}else{
		Mutpop = randomChrom[gene] + delta(t,randomChrom[gene] - V[2])
	}

	randomChrom[gene] = Mutpop
print(randomChrom)
#	pop[[selectEx]] = randomChrom
}
