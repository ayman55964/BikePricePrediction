#!/usr/bin/env python
# coding: utf-8

# In[258]:


# ## Import Libraries
# All libraries are used for specific tasks including data preprocessing, visualization, transformation and evaluation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 
import warnings
warnings.filterwarnings("ignore")


# In[259]:


# ## Import Data
# ### Read Training Data
# The training set is read locally and the **head** function is used to display the data for intial understanding

# "======Data understanding======"



import pandas as pd

dataTrain = pd.read_excel(r'C:\Users\Downloads/Bike_Price_Train.xlsx')
dataTrain = dataTrain.rename(columns = {'CC(Cubic capacity)' : 'Cubic_Capacity'}) 
dataTrain.head()


# In[260]:


type(dataTrain)  #data type


# In[261]:


# The **shape** function displays the number of rows and columns in the training set



dataTrain.shape # check dimension


# In[262]:


dataTest = pd.read_excel(r'C:\Users\Downloads/Bike_Price_Test.xlsx')
dataTest = dataTest.rename(columns = {'CC(Cubic capacity)' : 'Cubic_Capacity'}) 

dataTest.head()


# In[263]:


# The **shape** function displays the number of rows and columns in the testing set



dataTest.shape


# In[264]:


# Checking for null values in each column and displaying the sum of all null values in each column (Training Set)



dataTrain.isnull().sum()


# In[265]:


# Checking for null values in each column and displaying the sum of all null values in each column (Testing Set)



dataTest.isnull().sum()


# In[266]:





# In[267]:


# Checking if null values are eliminated (Training set)



dataTrain.isnull().sum()


# In[268]:


dataTrain.shape 


# In[269]:


# Checking if null values are eliminated (Testing set)



dataTest.isnull().sum()  


# In[270]:


dataTest.shape  # 5 rows removed


# In[271]:


# Checking the data types to see if all the data is in correct format. All the data seems to be in their required format.



dataTrain.dtypes  # checking the data type of every column


# In[272]:


# ## EDA (Exploratory Data Analysis)
# Visualizations are used to understand the relationship between the target variable and the features, in addition to correlation coefficient and p-value. 
# The visuals include heatmap, scatterplot,boxplot etc.
# 

# # Heat map



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
corr = dataTrain.corr()  
##This is a pandas DataFrame method that is used to calculate the correlation between variables in the DataFrame.
sns.heatmap(corr,annot=True)
plt.show()


# In[273]:


# From the heatmap, it is observed that 'year_produced' is the best feature among all the features with numerical data


dataTrain.describe()  #generate various summary statistics of a DataFrame 
#Note: Only features with numeric data are considered


# In[274]:


# A descriptive analysis to check incorrect entries and anormalies. This is also used to give an overview of the numerical data. It is observed that most of the data has no incorrect entries.

# 1. Count: The number of values in the dataframe.
# 2. Mean: The arithmetic mean or average of the values.
# 3. Standard Deviation (std): A measure of the dispersion or spread of the values.
# 4. Minimum: The minimum (smallest) value in each column.
# 5. 25th Percentile (25%): The value below which 25% of the data falls (1st quartile). Means 25% of the entire data falls under the value 158000 for odometer_value
# 6. 50th Percentile (50%): The median or value below which 50% of the data falls (2nd quartile).
# 7. 75th Percentile (75%): The value below which 75% of the data falls (3rd quartile).
# 8. Maximum: The maximum (largest) value in the Series.

# **************************************************************

# #Looking at the "minimum price", 1 USD is found.
# #This could be a wrong entry (or an outlier)



#Search for price = 1 , if so, change the price to 500
dataTrain.loc[dataTrain['Price'] == 1, 'Price'] = 500 


# In[275]:


dataTrain.describe()


# In[276]:


#Search for price < 500 , if so, change the price to 500
dataTrain.loc[dataTrain['Price'] < 500, 'Price'] = 500


# In[277]:


dataTrain.describe()  


# In[278]:


# Find the distribution of the price in the entire dataset
# using "bins"  -- Technique applied is called data binning



import matplotlib.pyplot as plt

dataTrain['Price'].plot(kind = 'hist', bins = 5, edgecolor='black')   # 5 bins are used
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()


# In[279]:


# From the histogram, it is understood that majority of the car samples are of lower prices


dataTrain.describe(include = 'object') #summary statistics for categorical values


# In[280]:


# ### Regression/scatter Plot
# This regression plot show the relation between **Bike_model** and **price**. A slight negative correlation is observed
# whaich shows that price is being affected by the change in odometer value.



import seaborn as sns
plt.figure(figsize=(10,6))
sns.regplot(x="Bike_model", y="Price", data=dataTrain)
plt.show()


# In[281]:


# As observed in the plot, a **negative correlation** is observed

label_encoder = LabelEncoder()
dataTrain['Bike_model'] = label_encoder.fit_transform(dataTrain['Bike_model'])


from scipy import stats
pearson_coef, p_value = stats.pearsonr(dataTrain['Bike_model'], dataTrain['Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[282]:


plt.figure(figsize=(10,6))
sns.regplot(x="Manufactured_year", y="Price", data=dataTrain)


# In[283]:


# As observed above, a high positive correlation of 0.7 is calculated along with the p-value of 0. This indicates that the correlation between the variables is significant hence year produced feature can be used for prediction.



pearson_coef, p_value = stats.pearsonr(dataTrain['Manufactured_year'], dataTrain['Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[284]:


# check for correlation between 'engine_capacity' and 'price'



plt.figure(figsize=(10,6))
sns.regplot(x="Engine_type", y="Price", data=dataTrain)


# In[285]:


# A 0.3 correlation is calculated which is very small with a p value of 0. This indicates that even though the correlation is small but its 30% of 100 which is significant hence this feature can be used for predicition.

label_encoder = LabelEncoder()
dataTrain['Engine_type'] = label_encoder.fit_transform(dataTrain['Engine_type'])


pearson_coef, p_value = stats.pearsonr(dataTrain['Engine_type'], dataTrain['Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# In[286]:


# This regression plot shows an minor positive correlation observed with the help of the best fit line. The calculation will confirm the actual value.

# -----check for correlation between 'number of photos' and 'price'------------



plt.figure(figsize=(10,6))
sns.regplot(x="Fuel_Capacity", y="Price", data=dataTrain)


# In[287]:


# The correlation is 0.31 based on the calculation while the p-value calculated is zero. This is similar to the last feature hence the significant 31% of 100 correlation makes this feature eligble for prediction.

label_encoder = LabelEncoder()
dataTrain['Fuel_Capacity'] = label_encoder.fit_transform(dataTrain['Fuel_Capacity'])



pearson_coef, p_value = stats.pearsonr(dataTrain['Fuel_Capacity'], dataTrain['Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[288]:


# This plot shows correlation with points all over the graph like the previous feature varibale.

# -------check correlation b/w number of mantenance and price-------------



plt.figure(figsize=(10,6))
sns.regplot(x="Cubic_Capacity", y="Price", data=dataTrain)


# In[289]:


# The calculation proves that a correlation is lesser than 0.1 percent and indicates no correlation and the p-value lesser than 0.05 confirms it. This feature is not a critical feature for predicition
# 


label_encoder = LabelEncoder()
dataTrain['Cubic_Capacity'] = label_encoder.fit_transform(dataTrain['Cubic_Capacity'])



pearson_coef, p_value = stats.pearsonr(dataTrain['Cubic_Capacity'], dataTrain['Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[290]:


# ---- this plot shows no correlation with points all over the graph ----



plt.figure(figsize=(10,6))
sns.regplot(x="Fuel_type", y="Price", data=dataTrain)


# In[291]:


label_encoder = LabelEncoder()
dataTrain['Fuel_type'] = label_encoder.fit_transform(dataTrain['Fuel_type'])



pearson_coef, p_value = stats.pearsonr(dataTrain['Fuel_type'], dataTrain['Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[292]:


# ### Box Plot
# These plots are used for categorical data to determine the importance of features for prediction. 

# In the given plot below, it is observed that the price range vary for automatic and manual transmisson. This indicates the categories can vary with price hence feature can be used for prediction



sns.boxplot(x="Bike_company", y="Price", data=dataTrain)


# In[293]:


# The box plot shows how prices vary based on different colors. This shows that color can be used as a feature for price prediction.



plt.figure(figsize=(10,6))
sns.boxplot(x="Bike_model", y="Price", data=dataTrain)


# In[294]:


# The engine type (based on fuel type) shows that both categories have almost the same price range which will not bring differences in price when prediction is made. Hence this feature is not suitable for price prediction



sns.boxplot(x="Engine_type", y="Price", data=dataTrain)



# In[295]:


# Thee box plot shows body type categories with varying prices per category hence this feature can be used for price prediction, not so signficant though



plt.figure(figsize=(10,6))
sns.boxplot(x="Manufactured_year", y="Price", data=dataTrain)


# In[296]:


# Has warranty feature shows a huge difference in price ranges between cars with warrant and vice versa. This feature is very important for price prediction as the bigger the difference in range the better the feature.



sns.boxplot(x="Engine_warranty", y="Price", data=dataTrain)


# In[297]:


# This feature is similar to the feature above, all three categories have wider price ranges between one another. This feature is also crucial for price prediction.



sns.boxplot(x="Fuel_type", y="Price", data=dataTrain)


# In[298]:


# Front and rear drive have **minimal price difference** while all drive shows a **greater difference** hence the feature can be used for prediction.



sns.boxplot(x="Fuel_Capacity", y="Price", data=dataTrain)


# In[299]:


# With not same price range between categories this feature is  suitable for prediction.



sns.boxplot(x="Cubic_Capacity", y="Price", data=dataTrain)


# In[300]:


# Using Exploratory data analysis, few features can be dropped because they had no impact on the price prediction. Those features are removed with the function below.(Training set)



dataTrain.drop(['Fuel_Capacity', 'Fuel_type', 'Engine_type','Engine_warranty'], axis = 1, inplace = True)



# In[301]:


# Same features are removed for testing set since the data will be used to train the model



dataTest.drop(['Fuel_Capacity', 'Fuel_type', 'Engine_type','Engine_warranty'], axis = 1, inplace = True)


# In[302]:


dataTrain.shape


# In[303]:


dataTest.shape


# In[304]:


# ### Data Transformation
# Label encoding of categorical features in the training set. Label encoding is converting categorical data into numerical data since the model cant understand textual data.

# ----Data Preparation--------

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

dataTrain.Cubic_Capacity = labelencoder.fit_transform(dataTrain.Cubic_Capacity)
dataTrain.Bike_company = labelencoder.fit_transform(dataTrain.Bike_company)
dataTrain.Bike_model = labelencoder.fit_transform(dataTrain.Bike_model)
dataTrain.Manufactured_year = labelencoder.fit_transform(dataTrain.Manufactured_year)


# In[305]:


# Label encoding of all categorical data in the testing set.


labelencoder = LabelEncoder()

dataTest.Cubic_Capacity = labelencoder.fit_transform(dataTest.Cubic_Capacity)
dataTest.Bike_company = labelencoder.fit_transform(dataTest.Bike_company)
dataTest.Bike_model = labelencoder.fit_transform(dataTest.Bike_model)
dataTest.Manufactured_year = labelencoder.fit_transform(dataTest.Manufactured_year)


# In[306]:


# Checking on the remaining features and if label encoding is applied to all categorical features (Training set).



dataTrain.head(10)


# In[307]:


# Check on the remaining features and application of label encoding to all categorical features (Testing set).



dataTest.head(10)


# In[311]:


# Dividing the data for training and testing accordingly. X takes the all features while Y takes the target variable
# 
# We have 13 actual columns [0-12 index]; 12 are predictor variables and 1 is the target variable



x_train=dataTrain.iloc[:,0:4]
y_train=dataTrain.iloc[:,5]
x_test=dataTest.iloc[:,0:4]
y_test=dataTest.iloc[:,5]


# In[312]:


x_train.head()


# In[313]:


y_train.head()


# In[314]:


# ## Fit Model
# ### Multiple Linear Regression
# Calling multiple linear regression model and fitting the training set



from sklearn.linear_model import LinearRegression

model = LinearRegression()
model_mlr = model.fit(x_train,y_train)



# In[315]:


# Making price prediction using the testing set (Fit to MLR)

y_pred1 = model_mlr.predict(x_test)


# In[316]:


#randomly checking the y-test values 
y_test[0]


# In[317]:


#randomly checking the y-pred values 
y_pred1[0]


# In[318]:


# y_test[0]   and   y_pred1[0]   have different values.. In other words, there is error

# ### MLR Evaluation
# 

# Calculating the Mean Square Error for MLR model



mse1 = mean_squared_error(y_test, y_pred1)
print('The mean square error for Multiple Linear Regression: ', mse1)


# In[319]:


# Calculating the Mean Absolute Error for MLR model


mae1= mean_absolute_error(y_test, y_pred1)
print('The mean absolute error for Multiple Linear Regression: ', mae1)



# In[320]:


# ### Random Forest Regressor (checking other Models)
# Calling the random forest model and fitting the training data



rf = RandomForestRegressor()
model_rf = rf.fit(x_train,y_train)


# In[321]:


# Prediction of car prices using the testing data



y_pred2 = model_rf.predict(x_test)



# In[322]:


# ### Random Forest Evaluation
# 

# Calculating the Mean Square Error for Random Forest Model (Lowest MSE value)



mse2 = mean_squared_error(y_test, y_pred2)
print('The mean square error of price and predicted value is: ', mse2)



# In[323]:


# Calculating the Mean Absolute Error for Random Forest Model (Lowest Mean Absolute Error)



mae2= mean_absolute_error(y_test, y_pred2)
print('The mean absolute error of price and predicted value is: ', mae2)


# In[324]:


# ### LASSO Model 
# Calling the model and fitting the training data



LassoModel = Lasso()
model_lm = LassoModel.fit(x_train,y_train)


# In[325]:


# Price prediction uisng testing data



y_pred3 = model_lm.predict(x_test)


# In[326]:


# ### LASSO Evaluation  (checking another model)
# 

# Mean Absolute Error for LASSO Model



mae3= mean_absolute_error(y_test, y_pred3)
print('The mean absolute error of price and predicted value is: ', mae3)


# In[327]:


# Mean Squared Error for the LASSO Model


mse3 = mean_squared_error(y_test, y_pred3)
print('The mean square error of price and predicted value is: ', mse3)


# In[328]:


scores = [('MLR', mae1),
          ('Random Forest', mae2),
          ('LASSO', mae3)
         ]         



# In[329]:


mae = pd.DataFrame(data = scores, columns=['Model', 'MAE Score'])
mae


# In[330]:


mae.sort_values(by=(['MAE Score']), ascending=False, inplace=True)

f, axe = plt.subplots(1,1, figsize=(10,7))
sns.barplot(x = mae['Model'], y=mae['MAE Score'], ax = axe)
axe.set_xlabel('Model', size=20)
axe.set_ylabel('Mean Absolute Error', size=20)

plt.show()


# In[ ]:




