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
Applying the above to locally weighted regression, we have<br/>
![Math](https://render.githubusercontent.com/render/math?math=\hat{y}=X\cdot\beta=X\cdot(X^TWX)^{-1}(X^TWy))<br/>
So in the case of Lowess, predictions ![Math](https://render.githubusercontent.com/render/math?math=\hat{y}) are linear combinations of ![Math](https://render.githubusercontent.com/render/math?math=y) and the computed weights ![Math](https://render.githubusercontent.com/render/math?math=W)
* ![Math](https://render.githubusercontent.com/render/math?math=W(i)) is the vecor of weights for observation ![Math](https://render.githubusercontent.com/render/math?math=i)
* The indpendent observations are the rows of the matrix ![Math](https://render.githubusercontent.com/render/math?math=X$). Each row has a number of columns (this is the number of features) - we can denote this number of features by ![Math](https://render.githubusercontent.com/render/math?math=p). As such, every row is a vector in ![Math](https://render.githubusercontent.com/render/math?math=\mathbb{R}^p).
* The Euclidean distance between 2 independent observations is the Euclidean distance (L2 norm) between the two represented ![Math](https://render.githubusercontent.com/render/math?math=p)-dimensional vectors. The equation is:
![Math](https://render.githubusercontent.com/render/math?math=dist(\vec{v},\vec{w})=%5Csqrt%7B%5C(v_1-w_1)^2%2B...%2B(v_p-w_p)^2%7D) where ![Math](https://render.githubusercontent.com/render/math?math=v_i) and ![Math](https://render.githubusercontent.com/render/math?math=w_i) represent features of observations ![Math](https://render.githubusercontent.com/render/math?math=v) and ![Math](https://render.githubusercontent.com/render/math?math=w). We will have ![Math](https://render.githubusercontent.com/render/math?math=n) different weight vectors where ![Math](https://render.githubusercontent.com/render/math?math=n) is the number of observations.



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
For loaclly weighted regression, $\hat{y}$ is obtained as a different linear combination of the value of $y$ (still a linear combination but we use the computed weights as seen above)<br/><br/><br/>
**Random forest regression:**<br/>
A random forest composed of n_estimators many decision trees, each with a maximum depth defined by max_depth. Decision trees are a form of supervised learning. That is, when the binary trees within the forest are constructed, the final classification or value of the dependent variable is known and is used to inform the independent construction of each decision tree (the trees do not interact or inform each other). Each tree draws a random sample from the given training data on which it trains and is constructed. For this reason, small quantities of high-dimensional data is more likely to result in overfitting so, in this case, it may be advantageous to experiment with dimensionality reduction techniques on the dataset. To make predictions, features from a selected observation in the testing dataset are used to draw similarities to the data used to construct, or train, the binary trees. Beginning with the root node of the tree, the algorithm traverses to either the left or right child of the current node. This decision of which direction to recur is made based on the probability (*p*) that the datapoint’s features are significantly different from the training data with *p*=1 indicating a perfect match between a feature observation in training data and the selected testing datapoint. On each node of a tree, a calculated hyper parameter determines which direction to traverse. The hyper parameter is a number falling within the distribution of a dataset feature. For example, if a weather dataset contains wind speed with a range of 0-50 (feature 1, *x_1*), temperature with a range of 20-80 (feature 2, *x_2*), and ice cream sales (*y*), a node’s hyper parameter might determine that data with wind speed *w* < 20 falls to the left child and ![Math](https://render.githubusercontent.com/render/math?math=w\geq20) falls to the right child. A node can split on several features (the number of features split on at each node is limited to prevent over-reliance on any particular feature for predictions). Subsequent nodes may further split the features represented in the hyper parameters of parent node or may split on different parameters. We also define a parameter called min_samples_split when calling the random forest regressor. This value defines how many training datapoints are required to recur onto a terminal node to split it (ie, this node will remain a leaf node unless enough datapoints end up classified as the same thing so further distinction can be made within this group of points deemed similar in some way by the previous node). Since splitting occurs recursively (since it’s supervised learning), meaning we start with the dependent variable y (in the leaves) and construct upwards to the root, splitting as we recurrently split the terminal nodes, populating the next higher depth with a new parent node whose children are the two nodes created by splitting. Once this value has been exceeded, it must next be determined if it is advantageous to split the node using “information gain” or Gain(y, x_i). For each feature within the tree’s subset of training data, ![Math](https://render.githubusercontent.com/render/math?math=Gain(y,x_i)=R(y)-R(y,x_i)) is calculated. With the goal of maximizing information gain, this value is used to select which feature (![Math](https://render.githubusercontent.com/render/math?math=x_i)) to split the node on. ![Math](https://render.githubusercontent.com/render/math?math=R) is a function known as the “impurity criterion”. This function is selected depending on the datatype of the dependent variable *y*. For regression (predicting a continuous value), mean squared error (MSE) or mean absolute error (MAE) can be used<br/>
MSE: ![Math](https://render.githubusercontent.com/render/math?math=\frac{1}{N}\sum_{i=1}^N(y_i-\mu)^2)<br/>
MAE: ![Math](https://render.githubusercontent.com/render/math?math=\frac{1}{N}\sum_{i=1}^N|y_i-\mu|)<br/>
* where *N* represents the number of observations in the data subset<br/>

    - https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e
- The leaves of a decision tree represent different classifications or predictions. This is why it’s important that we do not extrapolate - use the model to make predictions beyond the range or training data because our model does not encompass data behavior in these extended regions. Once the data has matriculated through all trees within the forest, each tree returns a prediction for the dependent variable y. These predictions are averaged to provide a final prediction for the y value of the test datapoint.
### Implementing Lowess
### Implementing Random Forest
### Comparison using MSE
