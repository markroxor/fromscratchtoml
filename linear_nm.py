import numpy as np
dataset=[[1,1],[2,3],[4,3],[3,2],[5,5]]
print (dataset)
x=[row[0] for row in dataset]
y=[row[1] for row in dataset]

mean_x=np.mean(x)
print("X Mean: ",mean_x)

mean_y=np.mean(y)
print("Y Mean: ",mean_y)


var_x=np.var(x)
#print(var_x)
print("Variance x: ",var_x)

var_y=np.var(y)

co_var=np.sum(np.cov(x,y))
print("Covariance: ",co_var)

b1=co_var/var_x
#print(b1)
b0=mean_y - b1*mean_x
#print(b0)

print('Coefficients: B0=%.3f, B1=%.3f' % (b0,b1))
#print('Coefficients: B1=%.3f',b1)
