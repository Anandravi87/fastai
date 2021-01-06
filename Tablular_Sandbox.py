#!/usr/bin/env python
# coding: utf-8

# In[2]:


#hide
get_ipython().system('pip install -Uqq fastbook kaggle waterfallcharts treeinterpreter dtreeviz')
import fastbook
fastbook.setup_book()


# In[3]:


get_ipython().system('pip install kaggle')


# In[4]:


#hide
from fastbook import *
from kaggle import api
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from dtreeviz.trees import *
from IPython.display import Image, display_svg, SVG

pd.options.display.max_rows = 20
pd.options.display.max_columns = 8


# In[5]:


df = pd.read_csv('/storage/archive/bluebook/TrainAndValid.csv', low_memory=False)


# In[6]:


df.columns


# # Playground start

# In[84]:


df_trial = pd.read_csv('/storage/archive/bluebook/TrainAndValid.csv', low_memory=False)


# In[9]:


df_trial['ProductSize'].unique()


# In[30]:


df_trial['ProductSize'] = df_trial['ProductSize'].astype('category')


# In[31]:


df_trial['ProductSize']


# ### Key learning 1: astype('category') changes the data type of the column from Object to Category

# In[15]:


len(df_trial.columns)


# ### Goal: to understand how Categorify works

# In[57]:


df_trial1 = pd.read_csv('/storage/archive/bluebook/TrainAndValid.csv', low_memory=False)


# In[11]:


procs_try = [Categorify, FillMissing]


# In[20]:


data = { 
    'A':['A1', 'A2', 'A3', 'A4', 'A5'],  
    'B':['B1', 'B2', 'B3', 'B4', 'B5'],  
    'C':['C1', 'C2', 'C3', 'C4', 'C5'],  
    'D':['D1', 'D2', 'D3', 'D4', 'D5'],  
    'E':['E1', 'E2', 'E3', 'E4', 'E5'] } 


# In[21]:


df_play = pd.DataFrame(data)


# In[22]:


df_play


# ### to drop columns use drop and iloc | axis = 1 | inplace = True modifies the dataframe on which the operation is performed
# #### df_trial.drop(df_trial.iloc[:, 4:52], axis = 1, inplace = True)

# In[86]:


df_trial.drop(df_trial.iloc[:, 4:52], axis = 1, inplace = True)


# In[87]:


df_trial.shape


# In[51]:


df_trial = df_trial.iloc[0:10, ]


# In[54]:


df_trial_bkp = df_trial


# ### to drop rows use iloc and directly assign the rows that need to be retained to the same df variable
# #### df_trial = df_trial.iloc[0:10, ]

# In[90]:


df_trial['ProductGroup'] = df_trial1['ProductGroup'].head(10)


# In[60]:


df_trial1['ProductGroup'].head(10)


# In[88]:


df_trial = df_trial.iloc[0:10, ]


# In[108]:


df_trial.shape


# In[74]:


df_trial.loc[0]


# ### to copy a row from another dataframe, use the below 
# #### df_trial['Grouser_Type'] = df_trial1['Grouser_Type'].head(10)

# In[91]:


df_trial['Grouser_Type'] = df_trial1['Grouser_Type'].head(10)


# In[138]:


df_trial.shape


# In[132]:


to_trial.items.head(4)


# In[133]:


to_trial.show(4)


# In[137]:


to_trial.items.shape


# In[80]:


dep_var = 'SalePrice'


# In[110]:


cont_try,cat_try = cont_cat_split(df_trial, 1, dep_var=dep_var)


# In[111]:


cont_try


# In[118]:


df_trial['ProductGroup']


# In[116]:


cat_try


# In[109]:


procs_try = [Categorify, FillMissing]


# In[ ]:





# In[120]:


train_idx


# In[121]:


df_trial.shape


# In[122]:


array([0,1,2,3,4])


# In[126]:


train_idx_trial = array(range(5))


# In[127]:


valid_idx_trial = array ([5,6,7,8,9])


# In[128]:


splits_trial = (list(train_idx_trial),list(valid_idx_trial))


# In[129]:


to_trial = TabularPandas(df_trial, procs_try, cat_try, cont_try, y_names=dep_var, splits=splits_trial)


# In[130]:


to_trial.items.head(3)


# # Playground end

# In[10]:


df['ProductSize'].unique()


# In[7]:


sizes = 'Large','Large / Medium','Medium','Small','Mini','Compact'


# In[9]:


df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'].cat.set_categories(sizes, ordered=True, inplace=True)


# In[12]:


dep_var = 'SalePrice'


# In[13]:


#take the logarithm of the dependent variable to calculate root mean squared log error (RMSLE)
df[dep_var] = np.log(df[dep_var])


# In[14]:


# Convert date to ordinal and categorical variables
#helps the model to learn from DOW, Holidays etc
#saledate is a timestamp datatype with all times set to 00:00
df = add_datepart(df, 'saledate')


# In[20]:


#view the columns created as a result of using the add_datepart function
#note that the saledate column is now replaced with several other features/columns
' '.join(o for o in df.columns if o.startswith('sale'))


# In[16]:


#Repeating the above for the test set
df_test = pd.read_csv('/storage/archive/bluebook/Test.csv', low_memory=False)
df_test = add_datepart(df_test, 'saledate')


# In[21]:


#data cleaning
#Categorify replaces a column with a numerical categorical column
#FillMissing replaces the missing values with the mean of that column
procs = [Categorify, FillMissing]


# In[22]:


#split training and validation sets
#data after Nov 2011 is the validation set and the data prior to Nov 2011 is the training set
cond = (df.saleYear<2011) | (df.saleMonth<10)
train_idx = np.where( cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx),list(valid_idx))


# In[23]:


#fast ai function to return the continous and categorical variables
#this is required for TabularPandas
cont,cat = cont_cat_split(df, 1, dep_var=dep_var)


# In[24]:


cont


# In[17]:


cat


# In[27]:


df['Grouser_Type'].unique()


# In[28]:


#TabularPandas behaves like the fatai Datasets object
#it provides the train and valud attributes (see next cell)
to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)


# In[29]:


len(to.train),len(to.valid)


# In[31]:


#when using show(), the data is displayed as strings
to.show(5)


# In[32]:


#however, the underlying data is numeric
to.items.head(5)


# # Creating the Decision Tree -- with stopping criteria (max leaves = 4)

# In[36]:


xs,y = to.train.xs,to.train.y
valid_xs,valid_y = to.valid.xs,to.valid.y


# In[38]:


#Building the Decision Tree
m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(xs, y);


# ### to keep things simple, only 4 leaf nodes were used in this decision tree

# In[43]:


draw_tree(m, xs, size=20, leaves_parallel=True, precision=2)


# In[44]:


#Tree visualization using dtreeviz library
samp_idx = np.random.permutation(len(y))[:500]
dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
        fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
        orientation='LR')


# In[45]:


#Replace year made = 1000 with 1950 to make the visualization easier
xs.loc[xs['YearMade']<1900, 'YearMade'] = 1950
valid_xs.loc[valid_xs['YearMade']<1900, 'YearMade'] = 1950


# In[46]:


#Running the Tree visualization again to see if the visualization has improved
samp_idx = np.random.permutation(len(y))[:500]
dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
        fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
        orientation='LR')


# # Creating the Decision Tree -- without stopping criteria

# In[47]:


m = DecisionTreeRegressor()
m.fit(xs, y);


# In[ ]:


#Creating a function to check the root mean squared error of the model (m_rmse)


# In[48]:


def r_mse(pred,y): 
    return round(math.sqrt(((pred-y)**2).mean()), 2)

def m_rmse(m, xs, y): 
    return r_mse(m.predict(xs), y)


# In[49]:


#calculating the training error using the function created above
m_rmse(m, xs, y)


# In[50]:


#Calculating the validation error
m_rmse(m, valid_xs, valid_y)


# In[51]:


#Seems like the model is overfitting
#calculating the number of leaf nodes of the unconstrained model
m.get_n_leaves(), len(xs)


# # Creating the Decision Tree with a constrain that every leaf node contains ATLEAST 25 examples -- to assess if the the model still overfits

# In[52]:


m = DecisionTreeRegressor(min_samples_leaf=25)
m.fit(to.train.xs, to.train.y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


# In[53]:


#with the constraint in place, we got a lower validation error and non-zero training error
#calculating the number of leaves in the model with 
m.get_n_leaves()


# # Creating a Random Forest
# ### n_estimators --> number of trees
# ### max_samples --> number of rows
# ### max_features --> number of columns (0.5 means take half of the total number)
# ### min_samples_leaf --> min number of examples each leaf node should have (similar to previous section)

# In[60]:


def rf(xs, y, n_estimators=40, max_samples=200_000,
       max_features='sqrt', min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


# In[61]:


#Creating the Random Forest model
m = rf(xs, y);


# In[62]:


#Calculating the error of the Random Forest model
print ("training error", m_rmse(m, xs, y))


# In[63]:


print ("validation error", m_rmse(m, valid_xs, valid_y))


# In[ ]:





# In[104]:


#see the training and validation errors
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


# # Creating a Random Forest

# In[39]:


get_ipython().system('pip install -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn')


# In[40]:


#n_estimators = number of trees, max_samples=number of rows, max_features = number of features 
#(% of the total number of features)

def rf(xs, y, n_estimators=40, max_samples=200_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


# In[41]:


m = rf(xs, y);


# In[42]:


m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


# In[43]:


#n_estimators = number of trees, max_samples=number of rows, max_features = number of features 
#(% of the total number of features)

def rf(xs, y, n_estimators=150, max_samples=200_000,
       max_features='sqrt', min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


# In[ ]:


m = rf(xs, y);


# In[ ]:




