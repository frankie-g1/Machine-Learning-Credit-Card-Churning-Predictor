#A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction

#I got this dataset from a website with the URL as https://leaps.analyttica.com/home. I have been using this for a while to get datasets and accordingly work on them to produce fruitful results. The site explains how to solve a particular business problem.

#Now, this dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

#We have only 16.07% of customers who have churned. Thus, it's a bit difficult to train our model to predict churning customers.

# -Sakshi Goyal (Credit Card customers) https://www.kaggle.com/sakshigoyal7/credit-card-customers

import pandas as pd
import matplotlib.pyplot as plt
import numpy



data = pd.read_csv('/home/guarinof/BankChurners.csv')
data.shape
data.head(3)


data.shape
data.head(3)
data['Attrition_Flag'].head(2)

data['Customer_Age'].unique()

#.aggregate({'Customer_Age':np.max}).sort_values(by=['Customer_Age'], ascending=False)
eduMask = data['Education_Level'] == 'High School'
churners = data['Attrition_Flag'] == 'Attrited Customer'


#Average age of all atrited customers and all existing customers
data.groupby('Attrition_Flag')['Customer_Age'].aggregate({'Customer_Age':np.mean})
#


#Minimum age of all atrited customers and all existing customers
data.groupby('Attrition_Flag')['Customer_Age'].aggregate({'Customer_Age':np.min})
#


#Maximum age of all atrited customers and all existing customers
data.groupby('Attrition_Flag')['Customer_Age'].aggregate({'Customer_Age':np.max})
#


#Side note, cannot aggregate discrete values, but you can group by them~
#Show median credit limit of Attrited Customers - (mask), for each card
attrited = data['Attrition_Flag'] == 'Attrited Customer'
data[attrited].groupby('Card_Category')['Credit_Limit'].aggregate({'Credit_Limit':np.median})
#Cannot aggregate when chose column is 'Income_Category'.


#Show median months on book of Attrited Customers - (mask), for each card
data[attrited].groupby('Card_Category')['Months_on_book'].aggregate({'Months_on_book':np.median})
#### Interesting. Most attrited customers, even almost regardless of card, all
#### stay for an average of ~36