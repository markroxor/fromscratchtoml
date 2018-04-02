# Supervised_Learning
## Linear Regression
Let's say we have a dataset of type [ ```(x1, y1), (x2, y2), (x3, y3) ... (xn, yn)```] where xi = size of house and yi = price of house in a particular area.

The task is train your system on the given dataset and predict the price for on an unseen size. 

We can train a hypothesis function using linear regression which can predict the price for a given size. The hypothesis function for linear regression can be can be derived in two ways. The first way is trivial but interesting which is known as the analytical approach. The following section will describe about the analytical approach.

##### Analytical Approach
y = target variable
x = predictor variable
take a hypothesis function ```f(x) = b1x+b0``` ( which is the equation of a straight line, 12th standard math).
b0 and b1 are the parameters where b1 = slope of the line and b0 = intercept. If you are able to find these two parameters then you have found the hypothesis function. The analytical approach can compute the parameter b0 and b1 directly from the training data.

#### Let's see the couple of mathematical equations.
In machine learning literature, we find the hypothesis function by computing the loss function. We try to optimize the loss function in a way that the given the input the loss should be the minimum. This loss function is also known as cost function or objective function.

There exist varieties of loss function. However, we will use the simplest loss function (L2 loss). The equations below describes the Loss function and the steps to compute the first parameter b0.

 ! [first equation] (https://raw.githubusercontent.com/somnat/Supervised_Learning/master/img/eq1.gif)
 
 ! [second equation] (https://raw.githubusercontent.com/somnat/Supervised_Learning/master/img/eq2.gif)
 
 ! [third equation] (https://raw.githubusercontent.com/somnat/Supervised_Learning/master/img/eq3.gif)
  
 ! [fourth equation] (https://raw.githubusercontent.com/somnat/Supervised_Learning/master/img/eq4.gif)

 ! [fifth equation] (https://raw.githubusercontent.com/somnat/Supervised_Learning/master/img/eq5.gif)
 
 As we can see in the equation(5) that the value of ```b0 = mean(y) - b1*mean(x)```. This implies that the first step in analaytical approach is to compute the mean for target and predictor. The equations below shows how to compute the another parameter b1 (note in b1 is wrongly written as b). 
 
  ! [sixth equation] (https://raw.githubusercontent.com/somnat/Supervised_Learning/master/img/eq6.gif)
  
  ! [seventh equation] (https://raw.githubusercontent.com/somnat/Supervised_Learning/master/img/eq7.gif)
   
  ! [eighth equation] (https://raw.githubusercontent.com/somnat/Supervised_Learning/master/img/eq8.gif)


As shown above in equation (8) that the parameter b1 is the ratio of covariance of x,y to variance of x. Therefore, the second step is to calculate the cov(x,y) and var(x). Use these two values to compute the parameter b1.


 
 
 


 



