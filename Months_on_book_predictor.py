import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy
import seaborn as sns

bank = pd.read_csv('BankChurners.csv')

'''
We tried one-hot encoding, which transforms a discrete value column
into x (number of unique discrete values) amount of columns
Ex
idx CTRY
11  Spain
22  France
33  Antarctica

into 
idx Spain France Antarctica
11  1       0       0
22  0       1       0
33  0       0       1
'''
from sklearn.preprocessing import LabelBinarizer
y = LabelBinarizer().fit_transform(bank.Education_Level)
w = LabelBinarizer().fit_transform(bank.Income_Category)
z = LabelBinarizer().fit_transform(bank.Card_Category)

'''
Gender and Marital are converted to 1s and 0s also, but there are only 2 discrete values 
this time around

Also dropping Attrition_Flag and Months_on_book since those are both
target columns
The other two columns the Kaggle guy said to drop
'''
x = bank.drop(['Attrition_Flag','Months_on_book','CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)
x['Gender'] = np.where(x['Gender'] == 'M', 1, 0)
x['Marital_Status'] = np.where(x['Marital_Status'] == 'Married', 1, 0)
x = x.drop(['Education_Level', 'Income_Category', 'Card_Category'], axis=1)


'''
Assigning data DF column names from the earlier one-hots
Column names came from ex.) pd.get_dummies(bank.Income_Category, prefix='Income_Category')
'''
x['Education_Level_College'] = y[:,0]
x['Education_Level_Doctorate'] = y[:, 1]
x['Education_Level_Graduate'] = y[:, 2]
x['Education_Level_High School'] = y[:, 3]
x['Education_Level_Post-Graduate'] = y[:, 4]
x['Education_Level_Uneducated'] = y[:, 5]
x['Education_Level_Unknown'] = y[:, 6]

x['Income_Category_$120K +'] = w[:, 0]
x['Income_Category_$40K - $60K'] = w[:, 1]
x['Income_Category_$60K - $80K'] = w[:, 2]
x['Income_Category_$80K - $120K'] = w[:, 3]
x['Income_Category_Less than $40K'] = w[:, 4]
x['Income_Category_Unknown'] = w[:, 5]


x['Card_Category_Blue'] = z[:, 0]
x['Card_Category_Gold'] = z[:, 1]
x['Card_Category_Platinum'] = z[:, 2]
x['Card_Category_Silver'] = z[:, 3]

target = bank['Months_on_book']

'''
Train and Learn
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, target, test_size=0.1, random_state=1)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_model = model.predict(X_test)
y_model.shape

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_model)

'''
Swirlydirly for months on book
If this was 100% accurate, y would equal x
'''
plt.scatter(y_model, y_test, c=y_test, cmap=plt.cm.get_cmap('nipy_spectral',10))
plt.colorbar(label='month label', ticks=range(0,56, 4))
plt.clim(0, 56)
plt.xlabel('Predicted Values')
plt.ylabel('True values')
plt.savefig('Months_on_book scatter.png', dpi=300)

'''
Another visual
Months_on_book is hard to predict
Accuracy score ~17%
'''
#Heatmap for months on book
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_model)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(mat, square=True, annot=True, cbar=False, cmap='YlGnBu') #flag, YlGnBu, jet
plt.xlabel('predicted value')
plt.ylabel('true value');
plt.savefig('Months_on_book_Heatmap.png', dpi=300)


