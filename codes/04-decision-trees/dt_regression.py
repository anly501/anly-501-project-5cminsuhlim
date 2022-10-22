# %% [markdown]
# # Regression with Decision Trees

# %% [markdown]
# ### Imports

# %%
import sklearn
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ### Load Data

# %%

df = pd.read_csv('../../data/01-modified-data/hours_worked_(by_sex_and_by_occupation)_final.csv')

## drop unneeded rows
df = df[df['sex'] != 'B']

# %%
# convert from long to wide
df = pd.pivot(df, index=['Category', 'sex'], columns=['Measure'], values='Value').reset_index()

# %%
# create numerical representations for occupation categories
df['Category_num'] = 0
df.iloc[0:2, 7] = 1
df.iloc[2:4, 7] = 2
df.iloc[4:6, 7] = 3
df.iloc[6:8, 7] = 4
df.iloc[8:, 7] = 5

# %%
# convert sex to numerical (0 = m, 1 = f)
df['sex'] = df['sex'].replace('F', 1).replace('M', 0)

# %%
df.rename(columns={'Average hrs worked among all workers' : 'Target'}, inplace=True)

# %% [markdown]
# ### Separate Predictor and Response Variables

# %%
X = df.iloc[:, [1,3,4,5,6,7]]
Y = df['Target']

# %% [markdown]
# ### Normalization

# %%
X=0.1+(X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))
Y=0.1+(Y-np.min(Y,axis=0))/(np.max(Y,axis=0)-np.min(Y,axis=0))

# %% [markdown]
# ### Numerical EDA

# %%
df['Target'].value_counts(ascending=True)

# %% [markdown]
# ### Multivariable Pair Plot

# %%
sns.pairplot(df.iloc[:, 1:7], hue='Target')
plt.show()

# %% [markdown]
# ### Correlation

# %%
corr = X.corr()
print(corr)	

# %% [markdown]
# ### Correlation Matrix Heatmap

# %%
sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(11, 9))  # Set up the matplotlib figure
cmap = sns.diverging_palette(230, 20, as_cmap=True) 	# Generate a custom diverging colormap
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,  cmap=cmap, vmin=-1, vmax=1, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show();

# %% [markdown]
# ### Remove Correlated Features

# %%
X.columns

# %%
X.drop(X.columns[[1,2,4]], axis=1, inplace=True)

# %% [markdown]
# ### Split Data

# %%
from sklearn.model_selection import train_test_split
test_ratio=0.2
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio, random_state=0)

# %%
print(type(x_train), x_train.shape)
print(type(y_train), y_train.shape)
print(type(x_test), x_test.shape)
print(type(y_test), y_test.shape)

# %% [markdown]
# ### Hyperparameter Tuning

# %%
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


# HYPER PARAMETER SEARCH FOR OPTIMAL NUMBER OF NEIGHBORS 
hyper_param=[]
train_error=[]
test_error=[]

# LOOP OVER HYPER-PARAM
for i in range(1,11):
    # INITIALIZE MODEL 
    model = DecisionTreeRegressor(max_depth=i)

    # TRAIN MODEL 
    model.fit(x_train,y_train)

    # OUTPUT PREDICTIONS FOR TRAINING AND TEST SET 
    yp_train = model.predict(x_train)
    yp_test = model.predict(x_test)

    # shift=1+np.min(y_train) #add shift to remove division by zero 
    err1=mean_absolute_error(y_train, yp_train) 
    err2=mean_absolute_error(y_test, yp_test) 
    
    # err1=100.0*np.mean(np.absolute((yp_train-y_train)/y_train))
    # err2=100.0*np.mean(np.absolute((yp_test-y_test)/y_test))

    hyper_param.append(i)
    train_error.append(err1)
    test_error.append(err2)

    print("hyperparam =",i)
    print(" train error:",err1)
    print(" test error:" ,err2)
    print(" error diff:" ,abs(err2-err1))

# %% [markdown]
# ### Convergence Plot

# %%
plt.plot(hyper_param,train_error ,linewidth=2, color='k')
plt.plot(hyper_param,test_error ,linewidth=2, color='b')

plt.xlabel("Depth of tree (max depth)")
plt.ylabel("Training (black) and test (blue) MAE (error)")

plt.show();

# %% [markdown]
# ### Re-train w/ Optimal Parameters

# %%
# INITIALIZE MODEL 
model = DecisionTreeRegressor(max_depth=2)
model.fit(x_train,y_train)                     # TRAIN MODEL 


# OUTPUT PREDICTIONS FOR TRAINING AND TEST SET 
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

err1=mean_absolute_error(y_train, yp_train) 
err2=mean_absolute_error(y_test, yp_test) 
    
print(" train error:",err1)
print(" test error:" ,err2)

# %% [markdown]
# ### Parity Plot

# %%
plt.plot(y_train,yp_train ,"o", color='k')
plt.plot(y_test,yp_test ,"o", color='b')
plt.plot(y_train,y_train ,"-", color='r')

plt.xlabel("y_data")
plt.ylabel("y_pred (blue=test)(black=Train)")

plt.show();

# %% [markdown]
# ### Plot Tree

# %%
from sklearn import tree
def plot_tree(model):
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(model, 
                    feature_names=X.columns,  
                    class_names=Y.name,
                    filled=True)
    plt.show()
plot_tree(model)

# %%
# LINEAR REGRESSION 
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, Y)

# OUTPUT PREDICTIONS FOR TRAINING AND TEST SET 
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

plt.plot(y_train,yp_train ,"o", color='k')
plt.plot(y_test,yp_test ,"o", color='b')
plt.plot(y_train,y_train,"-", color='r')

plt.xlabel("y_data")
plt.ylabel("y_pred (blue=test)(black=Train)")

    
err1=100.0*np.mean(np.absolute((yp_train-y_train)/y_train))
err2=100.0*np.mean(np.absolute((yp_test-y_test)/y_test))

print(" train error:",err1)
print(" test error:" ,err2)

plt.show();


