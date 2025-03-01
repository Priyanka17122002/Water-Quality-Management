#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


# In[10]:


data = pd.read_csv('water_potability.csv') 
data.head()
         


# In[11]:


data.isna().sum()


# In[12]:


data.describe()


# In[34]:


def conti_var(x):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), tight_layout=True)
    
    axes[0].set_title('Distribution')
    sns.histplot(x, ax=axes[0])
    axes[0].grid()
    
    axes[1].set_title('Outliers')
    sns.boxplot(x, ax=axes[1])
    
    axes[2].set_title('Relationship with Output Variable')
    sns.boxplot(x=data['Potability'], y=x, ax=axes[2])
    axes[2].grid()
conti_var(data.ph)    
    





# In[35]:


conti_var(data.Hardness)


# In[36]:


conti_var(data.Solids)


# In[37]:


conti_var(data.Sulfate)


# In[38]:


conti_var(data.Conductivity)


# In[39]:


conti_var(data.Organic_carbon)


# In[40]:


conti_var(data.Trihalomethanes)


# In[41]:


conti_var(data.Turbidity)


# In[48]:


data.Potability.value_counts()



        


# In[62]:


data.ph.fillna(data.ph.median(), inplace=True)
data.Trihalomethanes.fillna(data.Trihalomethanes.median(), inplace=True
                           )

test_x = data[data.Sulfate.isna()].drop('Sulfate', axis=1)
train_x = data[data.Sulfate.notna()].drop('Sulfate', axis=1)
train_y = data.Sulfate[data.Sulfate.notna()]

print('train_x = {}, train_y = {}, test_x = {}'.format(train_x.shape, train_y.shape, test_x.shape))



# In[65]:


from sklearn.linear_model import LinearRegression


lin = LinearRegression()


lin.fit(train_x,train_y)


for i in data[data.Sulfate.isna()].index:
    data.Sulfate[i] = lin.predict([data.loc[i,data.columns != 'Sulfate']])
plt.figure(figsize=(12,6)) 
sns.heatmap(data.corr(),annot=True,cmap='RdBu')


# In[67]:


from sklearn.ensemble import RandomForestClassifier


ran = RandomForestClassifier()


ran.fit(data.drop('Potability',axis=1),data.Potability)

plt.figure(figsize=(10,5)) 
sns.barplot(x=ran.feature_importances_,y=data.drop('Potability',axis=1
).columns) 
plt.show()


# In[15]:


import pandas as pd


data = pd.read_csv('water_potability.csv')
x = data.drop(['Potability', 'Organic_carbon'], axis=1)
y = data['Potability']
print('Input shape={}, Output shape={}'.format(x.shape, y.shape))

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
import pandas as pd  # Adding this import assuming pd is used for DataFrame

scalar = StandardScaler()
x = pd.DataFrame(scalar.fit_transform(x), columns=x.columns)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print('Shape of Splitting:')
print('x_train={}, y_train={}, x_test={}, y_test={}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))




# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

# Simulated dataset (Replace this with your actual dataset)
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.randint(0, 2, size=100)  # Binary labels (0 or 1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get predictions
pred_logis = model.predict(X_test)  # Predicted class labels (0 or 1)
pred_probs = model.predict_proba(X_test)[:, 1]  # Probability scores for the positive class

# Compute confusion matrix
cm = confusion_matrix(y_test, pred_logis)

# Visualize the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="inferno", linewidths=0.5)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, pred_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Compute accuracy
acc_logis = accuracy_score(y_test, pred_logis)
print(f'Accuracy: {acc_logis:.4f}')





# In[ ]:




