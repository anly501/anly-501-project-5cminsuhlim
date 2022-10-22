# %% [markdown]
# # Classification with Decision Trees

# %% [markdown]
# ### Imports

# %%
import sklearn
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# %% [markdown]
# ### Load Data

# %%
df = pd.read_csv('../../data/01-modified-data/occupations_detailed_(employment_and_wage).csv')

## drop unneeded column created from read_csv
df = df.iloc[:, 1:]

# %% [markdown]
# ### Separate Predictor and Response Variables

# %%
# Y="Target" COLUMN and X="everything else"
X = df.iloc[:, 2:6]
Y = df.iloc[:, 7]

# %% [markdown]
# ### Normalization

# %%
X=0.1+(X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))

# %% [markdown]
# ### Numerical EDA

# %%
df['Target'].value_counts(ascending=True)

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
# ### Multivariable Pair Plot

# %%
sns.pairplot(df.iloc[:, 2:7], hue='Target')
plt.show()

# %% [markdown]
# ### Baseline: Random Classifier

# %%
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
def random_classifier(y_data):
    ypred=[];
    max_label=np.max(y_data); #print(max_label)
    for i in range(0,len(y_data)):
        ypred.append(int(np.floor((max_label+1)*np.random.uniform(0,1))))

    print("-----RANDOM CLASSIFIER-----")
    print("count of prediction:",Counter(ypred).values()) # counts the elements' frequency
    print("probability of prediction:",np.fromiter(Counter(ypred).values(), dtype=float)/len(y_data)) # counts the elements' frequency
    print("accuracy",accuracy_score(y_data, ypred))
    print("precision, recall, fscore,",precision_recall_fscore_support(y_data, ypred))
random_classifier(Y)


# %% [markdown]
# ### Split Data

# %%
X.drop(columns=['EMP_PRSE'], inplace=True)

# %%
# PARTITION THE DATASET INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
test_ratio=0.2
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio, random_state=0, stratify=Y)

# %%
print(type(x_train), x_train.shape)
print(type(y_train), y_train.shape)
print(type(x_test), x_test.shape)
print(type(y_test), y_test.shape)

# %% [markdown]
# ### Train the Model

# %%
# TRAIN A SKLEARN DECISION TREE MODEL ON x_train,y_train 
from sklearn import tree
model = tree.DecisionTreeClassifier()
model = model.fit(x_train, y_train)

# %% [markdown]
# ### Check the Results

# %%
# USE THE MODEL TO MAKE PREDICTIONS FOR THE TRAINING AND TEST SET 
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

# %%
# GENERATES A CONFUSION MATRIX PLOT AND PRINTS MODEL PERFORMANCE METRICS
def confusion_plot(y_data, y_pred):    
    cm = confusion_matrix(y_data, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    
    print('ACCURACY:', accuracy_score(y_data, y_pred))
    print('RECALL:', recall_score(y_data, y_pred, average='weighted'))
    print('PRECISION:', precision_score(y_data, y_pred, average='weighted'))
    
    plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# %%
print("------TRAINING------")
confusion_plot(y_train,yp_train)
print("------TEST------")
confusion_plot(y_test,yp_test)

# %% [markdown]
# ### Visualize the Tree

# %%
# VISUALIZE THE DECISION TREE
from sklearn.tree import DecisionTreeRegressor

regr = DecisionTreeRegressor(random_state=1234)
model = regr.fit(x_train, y_train)

def plot_tree(model, X, Y):
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(model, 
                    feature_names=X.columns,  
                    class_names=Y.name,
                    filled=True)


# %%
plot_tree(model, X, Y)

# %% [markdown]
# ### Hyperparameter Tuning

# %%
# LOOP OVER POSSIBLE HYPER-PARAMETERS VALUES
test_results=[]
train_results=[]

for num_layer in range(1,51):
    model = tree.DecisionTreeClassifier(max_depth=num_layer)
    model = model.fit(x_train, y_train)

    yp_train=model.predict(x_train)
    yp_test=model.predict(x_test)

    # print(y_pred.shape)
    test_results.append([num_layer,accuracy_score(y_test, yp_test),recall_score(y_test, yp_test, average='weighted')])
    train_results.append([num_layer,accuracy_score(y_train, yp_train),recall_score(y_train, yp_train, average='weighted')])

# %%
layers = [el[0] for el in test_results]
          
test_acc = [el[1] for el in test_results]
test_rec = [el[2] for el in test_results]

train_acc = [el[1] for el in train_results]
train_rec = [el[2] for el in train_results]

# %%
# GENERATE PLOTS TO IDENTIFY OPTIMAL HYPERPARAMETER
def gen_plots(x, train, test):
    plt.plot(x,train, c='b')
    plt.scatter(x,train,c='b')
    plt.plot(x,test,c='r')
    plt.scatter(x,test,c='r')
    plt.xlabel("Number of layers in decision tree (max_depth)")
    plt.show();

plt.ylabel("ACCURACY: Training (blue) and Test (red)")
gen_plots(layers, train_acc, test_acc)
plt.ylabel("RECALL: Training (blue) and Test (red)")
gen_plots(layers, train_rec, test_rec)

# %% [markdown]
# ### Find Optimal Hyperparameter

# %%
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
hyper_param=[]
train_error=[]
test_error=[]
for i in range(1,23):
    # INITIALIZE MODEL 
    model = DecisionTreeRegressor(max_depth=i)

    # TRAIN MODEL 
    model.fit(x_train,y_train)

    # OUTPUT PREDICTIONS FOR TRAINING AND TEST SET 
    yp_train = model.predict(x_train)
    yp_test = model.predict(x_test)

    # GET MAE
    err1=mean_absolute_error(y_train, yp_train) 
    err2=mean_absolute_error(y_test, yp_test) 

    hyper_param.append(i)
    train_error.append(err1)
    test_error.append(err2)

    print("hyperparam =",i)
    print(" train error:",err1)
    print(" test error:" ,err2)
    print(" error diff:" ,abs(err2-err1))

# %% [markdown]
# ### Train Optimal Model

# %%
#### TRAIN A SKLEARN DECISION TREE MODEL ON x_train,y_train 
model = tree.DecisionTreeClassifier(max_depth=4)
model = model.fit(x_train, y_train)

yp_train=model.predict(x_train)
yp_test=model.predict(x_test)

# %%
print("------TRAINING------")
confusion_plot(y_train,yp_train)
print("------TEST------")
confusion_plot(y_test,yp_test)

# %%
regr = DecisionTreeRegressor(random_state=1234, max_depth=4)
model = regr.fit(x_train, y_train)

plot_tree(model, X, Y)


