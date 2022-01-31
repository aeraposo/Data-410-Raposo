## Reading reflection - Locally Weighted Regression: An Approach to Regression Analysis by Local Fitting (Cleveland and Devlin)
Based on visibly discernible trends in data, one may come to quick conclusions about approximate parametric (linear) models to represent a
supposed relationship.Although this method may be an appropriate starting place in estimating a model, it is ultimately riddled with uncertainty
and may poorly approximate more specific, localized trends. A more suitable option for such analysis is locally weighted regression, or “loess”.
Loess approximates non-parametric trends by fitting linear models to sequential subsets of data. Subdivisions are split on intervals in one
dimension of the data based on a computer determined, or user specified, number of points per interval by a weight function. Let X represent
multidimensional inputdata (a UxP matrix thats transpose is ![Math](https://render.githubusercontent.com/render/math?math=X^T)) and y be the
dependent variable. Parametric models assume the form ![Math](https://render.githubusercontent.com/render/math?math=y=X∗\beta+r\epsilon) where
![Math](https://render.githubusercontent.com/render/math?math=\epsilon) is independently and identically distributed with mean 0 and standard
deviation 1. To determine the \beta parameters, or weights, of the model, we write ![Math](https://render.githubusercontent.com/render/math?math=X^T*y = X^T∗X∗\beta X^T)
where ![Math](https://render.githubusercontent.com/render/math?math=X^T) is model noise and ![Math](https://render.githubusercontent.com/render/math?math=E(X^T)=0).
Thus, ![Math](https://render.githubusercontent.com/render/math?math=E(X^T*y)E(X^T∗𝑋∗\beta)=X^T∗X∗E(\beta)(X^T∗X)−1∗E(X^T ∗y)=E(\beta)),
which allows the parametric model’s \beta parameters associated with each vector contained in
![Math](https://render.githubusercontent.com/render/math?math=X) (representing the different dimensions of the data) to be calculated by
![Math](https://render.githubusercontent.com/render/math?math=(X^T*X)^{-1}). Loess uses this same process to estimate parametric equations on
data subintervals as described above, however, it is important to note that a widely varied distribution of datapoints can lead to overfitting
if too few or many points are selected for each interval.
