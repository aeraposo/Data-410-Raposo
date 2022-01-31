## Reading reflection - Locally Weighted Regression: An Approach to Regression Analysis by Local Fitting (Cleveland and Devlin)
Based on visibly discernible trends in data, one may come to quick conclusions about approximate parametric (linear) models to represent a
supposed relationship.Although this method may be an appropriate starting place in estimating a model, it is ultimately riddled with uncertainty
and may poorly approximate more specific, localized trends. A more suitable option for such analysis is locally weighted regression, or “loess”.
Loess approximates non-parametric trends by fitting linear models to sequential subsets of data. Subdivisions are split on intervals in one
dimension of the data based on a computer determined, or user specified, number of points per interval by a weight function. Let X represent
multidimensional inputdata (a UxP matrix thats transpose is ![Math](https://render.githubusercontent.com/render/math?math=X^T)) and y be the
dependent variable. Parametric models assume the form ![Math](https://render.githubusercontent.com/render/math?math=y=X\beta+r\epsilon) where
![Math](https://render.githubusercontent.com/render/math?math=\epsilon) is independently and identically distributed with mean 0 and standard
deviation 1. To determine the ![Math](https://render.githubusercontent.com/render/math?math=\beta) parameters, or weights, of the model, we write ![Math](https://render.githubusercontent.com/render/math?math=X^Ty=X^TX{\beta}X^T)
where ![Math](https://render.githubusercontent.com/render/math?math=X^T) is model noise and ![Math](https://render.githubusercontent.com/render/math?math=E(X^T)=0).
Thus, ![Math](https://render.githubusercontent.com/render/math?math=E(X^Ty){\cdot}E(X^TX\beta)=X^TX{\cdot}E(\beta)(X^TX)^{-1}{\cdot}E(X^Ty)=E(\beta)),
which allows the parametric model’s ![Math](https://render.githubusercontent.com/render/math?math=\beta) parameters associated with each vector contained in
![Math](https://render.githubusercontent.com/render/math?math=X) (representing the different dimensions of the data) to be calculated by
![Math](https://render.githubusercontent.com/render/math?math=(X^TX)^{-1}). Loess uses this same process to estimate parametric equations on
data subintervals as described above and since weights are calculated for individual datapoints, we can ensure that although the model may not be smooth, it will be coninuous (no breaks or jumps). It is important to note that a widely varied distribution of datapoints can lead to overfitting
if too few or many points are selected for each interval, however, modifications to the kernal (or "bump") fuction can help reduce this.
