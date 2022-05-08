# Predicting Saturated fat Consumption to Gauge Diet Quality in Child WIC Participants
### Background Research
Diet quality in the United States, and globally, has seen a dramatic decrease over the last century. With the almost uniform availability of convenient, cheap, and ultra-processed foods at retail outlets, swift intervention is crucial to safeguarding the health of the population. Although food although food may be available in abundance for many, poverty and food insecurity have a greater role in food choice than is often attributed. When analyzing this affected population, we must define and draw connections between food insecurity and poverty.<br/><br/>
Poverty is a “state in which a person lacks financial and material resources needed to achieve a minimum standard of living” (Conrad, 2022). This metric is computed by the definition set by the federal government: 3 times the cost of a minimum food diet in 1963 (with cost updated for current % inflation). Although the relative number of people in poverty has increased due to population growth, the percentage of households considered below the federal poverty guideline has decreased. As of present, approximately 11% of American households are at or below the poverty guideline, a historical low for the United States.<br/><br/>
Food security is the degree to which the nutritional needs of individuals are impacted by their access to resources (particularly financial resources). There are 3 recognized levels of food security used in public health and nutrition science research – food secure, low food security, and very low food security. Individuals who are food secure are able to acquire sufficient food throughout the year. Low food security, of which ~6.8% of American households are classified, is characterized by uncertainty in obtaining or inability to obtain enough food to meet the nutritional needs of all household members because of insufficient monetary or other resources. Further, very low food security is recognized when normal eating patterns of some or all household members are disrupted and food intake falls below recommended levels, accounting for ~4.3% of Americans households. As suggested in the above descriptions, poverty and food security are closely intertwined. Notably, low food security is categorically considered a type of poverty.<br/><br/>
With the goals of reducing food insecurity and improving diet quality, several government programs were established that provide financial assistance and nutrition education to these affected groups. The Supplemental Nutrition Assistance Program (SNAP) and the Special Supplemental for Women, Infants, and Children (WIC) are the largest and most widely recognized of these programs. SNAP, formerly known as the “food stamp” program, aids over 40 million Americans each year (~11% of the US population at any given time), half of which are children. SNAP provides monthly benefit allocations through an inconspicuous Electronic Benefits Transfer (EBT) card in addition to access to nutrition and food preparation education through the SNAP-Ed program. Benefits, that average $240 per month, are distributed after eligibility is reassessed at the household level of enrolled and prospective participants on a monthly basis. Eligibility is determined based on an assessment of household members’ income, employment, and immigration status. Included in the omnibus Farm Bill, SNAP is funded on a rolling basis, meaning it has no budget cap and therefore serves as many participants as needed. In other words, increased participation in SNAP does not negatively impact the funds allocated to other participants. Although SNAP and WIC conclusively reduce food insecurity, there is no evidence to support that the programs are at all affective at increasing the diet quality of participants. Data collected at the national level shows that people near, at, and below the poverty line and those recognized as food insecure (low or very food secure) have lower diet quality than their higher income counterparts. Moreover, SNAP participants have historically had lower diet quality than eligible non-participants (Zhang et al., 2018). This trend, however, is not observed in WIC participants. WIC provides financial assistance participants to purchase food but is district from SNAP in that it limits purchases with the benefits to a specific list of approved foods for at-home consumption. This group of foods is called the “WIC food package” and includes breakfast cereal, whole grain bread, baby food/formula, milk, cheese, yogurt, tofu, canned fish, soy drinks, eggs, juice, and peanut butter. WIC also offers optional services nutrition counseling, breastfeeding support, and healthcare referrals. Eligibility for this program is determined based on 3 criteria domains: categorial, income, and nutrition. Applicants are deemed categorially eligible if they are a woman who is pregnant, less than 6 months postpartum, breastfeeding a child who is less than 1 year old, or has a child of less than 5 years of age; are income eligible if they are below 1.85 times the poverty line; and are deemed nutritionally eligible based on their risk assessed through a non-standardized clinical assessment. WIC currently supports between 7-9 million women, infants, and children at any given time, meaning around half of all infants in the US are actively enrolled. With rising rates of childhood obesity and diet-related illness, we must consider the program’s effectiveness in meeting the goal of increasing diet quality. One way by assessing the risk of obesity and other health complications. Because metrics such as the weights, heights, and other standard clinical measures of child participants are not available due to privacy concerns, we must use other metrics to infer the health impacts of their diet.<br/><br/>
One nutrient that is of significant concern in assessing diet quality is the quantity of saturated fat consumed. The structure of saturated fats contains no double bonds, meaning they are fully saturated with H atoms. This structure allows molecules to easily stack, resulting in a solid composition at room temperature. When consumed and metabolized, saturated fats cause the liver to synthesize large volumes of low-density lipoprotein (LDL) “bad” cholesterol and a small quantity of high-density lipoprotein (HDL) “good” cholesterol. Circulating LDL cholesterol can bind to the endothelial lining of the vascular system, resulting in arterial microtears, inflammation, reduced endothelial elasticity, plaque formation, and arterial narrowing. Chronically, these internal implications compound, which leads conditions such as obesity, hypertension, atherosclerosis (heart disease), stroke, and more. The medical implications of food choices are both underestimated and sever so it is of paramount importance to address harmful diet patterns at a young age to reduce or prevent life-threatening conditions that become more apparent during adult years. <br/><br/>

### Methods
#### Variable Selection
The goal of this research is to predict the level of saturated fat consumption in child WIC participants based on age and other dietary metrics such calorie, total fat, sugar, and fiber intake. Although this metric is just one small portion of what someone’s overall diet may look like, saturated fats are found in their highest concentrations in highly processed food pastries, processed or fatty meats, and solids fats such as butter, which may indicate poor diet quality when consumed in excess. The data used will come from the ‘WIC Infant and Toddler Feeding Practices Study-2 (WIC ITFPS-2): Prenatal, Infant Year, Second Year, Third Year, and Fourth Year’ dataset collected by USDA. The dataset contains 32,750 observations of 105 features related to the diets of 0-4 year old participants so a regularization and variable selection procedure will be necessary.<br/><br/>


In project 5, we identified 5 useful variable selection algorithms: Square Root Lasso, Lasso, Ridge, Elastic Net, and SCAD.

As in project 5, GridSearchCV was used to identify the optimal choice of alpha for each algorithm. In the cases of Lasso and Elastic Net, the GridSearchCV method did not converge so I wrote my own hyperparameter-tuning function ("try_alphas").

```
def try_alphas(rid = False, lass = False, EN = False, min_val = 0.01, max_val = 1.01, step = 0.1):
  best_alpha = 0
  best_mse = max(y)**2 # note that max(y) > 1
  print(best_mse)
  alphas = np.arange(min_val,max_val,step)
  def test_alpha(a,best_mse,best_alpha):
    print(best_mse)
    model.fit(x,y)
    mae = mean_absolute_error(y,model.predict(x)) # The MAE is low
    print(mae)
    if mae<best_mse:
      best_mse = mae
      best_alpha = a
    return [best_alpha, best_mse]
  for a in alphas:
    if rid == True:
      model = Ridge(alpha=a,fit_intercept=False,max_iter=10000)
      test = test_alpha(a, best_mse,best_alpha)
      best_mse = test[1]
      best_alpha = test[0]
    elif lass == True:
      print(a)
      model = Lasso(alpha=a,fit_intercept=False,max_iter=10000)
      test = test_alpha(a, best_mse,best_alpha)
      best_mse = test[1]
      best_alpha = test[0]
    else: #EN == True:
      model = ElasticNet(alpha=a,fit_intercept=False,max_iter=10000)
      test = test_alpha(a, best_mse,best_alpha)
      best_mse = test[1]
      best_alpha = test[0]
  return [best_alpha, best_mse]
```

Although Ridge was found to be the most accurate and consistent choice on data simulations, Lasso proved most useful on this dataset with a cross-validated MAE of 0.138. Since we are working with high-dimensional data, it was necessary to eliminate a substantial proportion of features to both maximize model performance and to eliminate sources of multicolinearity.
After identifying the variables eliminated by Lasso, I subset the data to isolate the features deemed important for analysis. Among these retained features, I noticed that several of the variables which contributed most strongly were highly correlated with saturated fat consumption. Namely, variables such as total fat and palmitic acid (a type of saturated fat) consumption were logically correlated to total saturated fat consumption so I opted to manually remove these features before proceeding to model testing and selection.<br/>


#### Model Selection
Next, I modeled the relationship between the independent variables (selected using aforementioned algorithm) and the dependent variable (quantity of saturated fat consumed) using numerous modeling techniques. The model types I will compare are Locally Weighted Regression (Lowess), Random Forest Regression, and Generalized Additive Modeling

##### Lowess
Because the reduced dataset still contains over 50 features and over 32,000 observatins, Lowess is unable to regress due to maximum runtime constraints. After reducing the model to just 10 features - 



– model selection will be based on the MSE and MAE through a k-fold cross validation process on reserved testing data.

If we boost Lowess and RFR, can they out-preform GAM??

Additionally, I will test boosting methods such as XGBoost, LightGBM, and a homemade booster that I developed for project 3, which could be incorporated to improve model accuracy. Using this model, I will make comparisons between the predicted saturated fat consumption of WIC participants, the recommended limit of saturated fat consumption for their age, and national averages of saturated fat consumption for the child’s age bracket to gauge if WIC may be accomplishing its goal improving diet quality. This conclusion could help inform changes to larger nutrition assistance programs such as SNAP that have been unsuccessful with this objective.


Lastly, I eliminated additional variables that were correlated with saturated fat consumption and other variables that were not easily measurable (so this model could be more accessible for WIC participants)



[Dataset Link](https://data.nal.usda.gov/dataset/wic-infant-and-toddler-feeding-practices-study-2-wic-itfps-2-prenatal-infant-year-second-year-third-year-and-fourth-year-datasets-0![image](https://user-images.githubusercontent.com/67920301/163575230-f57eae51-4c4e-4084-ad37-6f396af8c22a.png))

### Methods

### Results
