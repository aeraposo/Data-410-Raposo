## Project 2: Lowess vs. Random Forest Regression
### Math behind the methods
**Lowess:**<br/>
Based on visibly discernible trends in data, one may come to quick conclusions about approximate parametric (linear) models to represent a
supposed relationship. Although this method may be an appropriate starting place in estimating a model, it is ultimately riddled with uncertainty
and may poorly approximate more specific, localized trends. A more suitable option for such analysis is locally weighted regression, or “loess”.
Loess approximates non-parametric trends by fitting linear models to sequential subsets of data. Subdivisions are split on intervals in one
dimension of the data based on a computer determined, or user specified, number of points per interval by a weight function. Let ![Math](https://render.githubusercontent.com/render/math?math=X) represent
multidimensional inputdata (a UxP matrix thats transpose is ![Math](https://render.githubusercontent.com/render/math?math=X^T)) and ![Math](https://render.githubusercontent.com/render/math?math=y) be the
dependent variable. Linear regression determines these linear models, which give a predicted ![Math](https://render.githubusercontent.com/render/math?math=y), known as ![Math](https://render.githubusercontent.com/render/math?math=\hat{y}) for each input observation ![Math](https://render.githubusercontent.com/render/math?math=x) in ![Math](https://render.githubusercontent.com/render/math?math=X). Linear regression can be seen as a linear combination of the observed outputs (values of the dependent variable). To understand why this is so, we must further investigate the math behind the assumed form of the models: ![Math](https://render.githubusercontent.com/render/math?math=y=X\beta%2B\sigma\epsilon) where ![Math](https://render.githubusercontent.com/render/math?math=\epsilon) is independently and identically distributed with mean 0 and standard deviation 1.<br/>
![Math](https://render.githubusercontent.com/render/math?math=X^Ty=X^TX\beta%2B\sigma{X^T\epsilon})<br/>

We solve for ![Math](https://render.githubusercontent.com/render/math?math=\beta) by assuming ![Math](https://render.githubusercontent.com/render/math?math=X) is not rank deficient (assme ![Math](https://render.githubusercontent.com/render/math?math=X^TX) is invertible, same as OLS assumption). So,<br/>
![Math](https://render.githubusercontent.com/render/math?math=\beta=(X^TX)^{-1}(X^Ty)-\sigma(X^TX)^{-1}X^T\epsilon).<br/>
We take the expected vlaue of this equation and obtain (where ![Math](https://render.githubusercontent.com/render/math?math=\bar{\beta}) is ![Math](https://render.githubusercontent.com/render/math?math=E(\beta)))<br/>
![Math](https://render.githubusercontent.com/render/math?math=\bar{\beta}=(X^TX)^{-1}(X^Ty))<br/>
Therefore, the predictions (predicted values) we make are <br/>
![Math](https://render.githubusercontent.com/render/math?math=\bar{\beta}=\hat{y}=X\cdot\beta=X\cdot(X^TX)^{-1}(X^Ty))<br/>
Now, we can see that predictions ![Math](https://render.githubusercontent.com/render/math?math=\hat{y}) are linear combinations of ![Math](https://render.githubusercontent.com/render/math?math=y) (ie, they are a matrix multiplied by ![Math](https://render.githubusercontent.com/render/math?math=y))<br/>
#------------------- left off here
For the locally weighted regression, we have<br/>
$$\hat{y}=X\cdot\beta = X\cdot(X^TWX)^{-1}(X^TWy)$$
The big idea:<br/>
Now we can see that the predictions we make ($\hat{y}$) are a linear combination of the actual observed variable $y$ (a matrix times $y$).<br/>
For loaclly weighted regression, $\hat{y}$ is obtained as a different linear combination of the value of $y$ (still a linear combination but we use the computed weights as seen above)

Parametric models assume the form ![Math](https://render.githubusercontent.com/render/math?math=y=X\beta%2Br\epsilon) where
![Math](https://render.githubusercontent.com/render/math?math=\epsilon) is independently and identically distributed with mean 0 and standard
deviation 1. To determine the ![Math](https://render.githubusercontent.com/render/math?math=\beta) parameters, or weights, of the model, we write ![Math](https://render.githubusercontent.com/render/math?math=X^Ty=X^TX{\beta}X^T)
where ![Math](https://render.githubusercontent.com/render/math?math=X^T) is model noise and ![Math](https://render.githubusercontent.com/render/math?math=E(X^T)=0).
Thus, ![Math](https://render.githubusercontent.com/render/math?math=E(X^Ty){\cdot}E(X^TX\beta)=X^TX{\cdot}E(\beta)(X^TX)^{-1}{\cdot}E(X^Ty)=E(\beta)),
which allows the parametric model’s ![Math](https://render.githubusercontent.com/render/math?math=\beta) parameters associated with each vector contained in
![Math](https://render.githubusercontent.com/render/math?math=X) (representing the different dimensions of the data) to be calculated by
![Math](https://render.githubusercontent.com/render/math?math=(X^TX)^{-1}). Loess uses this same process to estimate parametric equations on
data subintervals as described above and since weights are calculated for individual datapoints, we can ensure that although the model may not be smooth, it will be coninuous (no breaks or jumps). It is important to note that a widely varied distribution of datapoints can lead to overfitting
if too few or many points are selected for each interval, however, modifications to the kernal (or "bump") fuction can help reduce this.<br/>
**Important concept:** 




**Why?** We have 
We solve for $\beta$ by assuming X is not rank deficient (assme $X^TX$ is invertible, OLS assumption). So,<br/>
$$\beta = (X^TX)^{-1}(X^Ty)-\sigma(X^TX)^{-1}X^T\epsilon$$
We take the expected vlaue of this equation and obtain (where $\bar{\beta}$ is $E(\beta)$)
$$\bar{\beta} = (X^TX)^{-1}(X^Ty)$$
Therefore, the predictions (predicted values) we make are 
$$\hat{y}=X\cdot\beta = X\cdot(X^TX)^{-1}(X^Ty)$$
For the locally weighted regression, we have
$$\hat{y}=X\cdot\beta = X\cdot(X^TWX)^{-1}(X^TWy)$$
The big idea:<br/>
Now we can see that the predictions we make ($\hat{y}$) are a linear combination of the actual observed variable $y$ (a matrix times $y$).<br/>
For loaclly weighted regression, $\hat{y}$ is obtained as a different linear combination of the value of $y$ (still a linear combination but we use the computed weights as seen above)
**Random forest:**<br/>

### Implementing Lowess
### Implementing Random Forest
### Comparison using MSE
