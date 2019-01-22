
# coding: utf-8

# # Introduction

# <img src="images/bio.jpg">

# We have a lot of topics to cover over the next two days, so the purpose of this course is to give you the full pipeline of development for a data scientist using  a practical use case scenario.

# Why Data Science now?

# <img src="images/why data science.png">

# <src image = 'images/why data science 2.jpg'>

# <img src="images/why data science2.jpg">

# Why Data Science with Python

# <img src="images/python-r-other-2016-2017.jpg">

# ## Customizing your Notebook Environment

# Get your notebook up and running with Anaconda

# https://www.anaconda.com/download/

# <img src="images/anaconda.jpg">

# https://notebooks.azure.com/

# <img src = 'images/azure.jpg' >

# start your jupyter-notebook

# Some functions with your Jupyter Notebook

# Create files, folders, and functions

# ### Jupyter Notebook Functions: There are plenty

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ## Day 1 -> Section 1 -> Demo 1

# ### on Windows

# C:\Users\Ahmed Sherif>conda create --name AhmedDataScience python=3.6.1

# activate AhmedDataScience

# <img src = 'images/virtual env mac.jpg'>

# ### on mac or linux

# ahmed sherif$ conda create --name AhmedDataScience python=3.6.1

# source activate AhmedDataScience

# <img src = 'images/virtual env windows.jpg'>

# ### End of Demo 1

# In[ ]:




# ## Day 1 -> Section 2 -> Demo 2

# pip install ipykernel

# ### on mac/linux/windows
python –m ipykernel install -- user --name <VirtualEnvironmentName>
# pip list

# pip freeze

# ### General Python Functionality readily available in Python

# In[1]:

print('hello world')


# In[2]:

print(5, type(5))


# In[3]:

print(True, type(True))


# In[4]:

instructor = 'Ahmed'


# In[5]:

print('Hello, my name is '+ instructor +'\nand I am looking forward to meeting you!')


# In[6]:

print(f'Hello, my name is {instructor}\nand I am looking forward to meeting you!')


# In[7]:

age = 40


# In[8]:

print('Hello, my name is Ahmed Sherif and I am '+str(age)+' years old')


# In[9]:

print(f'Hello, my name is Ahmed Sherif and I am {age/2} years old')


# In[10]:

'f strings as of version 3.6'


# ## import your Dataset and much needed libraries

# In[11]:

get_ipython().system('pip list')


# In[12]:

get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install sklearn')


# In[13]:

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn


# In[14]:

get_ipython().system('pip list')


# In[15]:

print(f'pandas version: {pd.__version__}')
print(f'matplotlib version: {matplotlib.__version__}')

print(f'numpy version: {np.__version__}')


# ## I am cool

# In[16]:

"""
helllo, i am cool!
asdfasdfasdfa
            asfasdf
            
            

"""


# # End of Demo 2

# ## Day 1 -> Section 3 -> Demo 3 

# ## Data Profiling

# In[17]:

# describe what the data is about


# https://www.kaggle.com/zynicide/wine-reviews

# <img src = 'images/kaggle.jpg'>
country: The country that the wine is from

description: Notes from the Tasters

designation:  The vineyard within the winery where the grapes that made the wine are from

points: The number of points WineEnthusiast rated the wine on a scale of 1-100 (though they say they only post reviews for wines that score >=80)

price: The cost for a bottle of the wine

province: The province or state that the wine is from

region_1: The wine growing area in a province or state (ie Napa)

region_2: Sometimes there are more specific regions specified within a wine growing area (ie Rutherford inside the Napa Valley), but this value can sometimes be blank

taster_name: 

taster_twitter_handle: 

title: The title of the wine review, which often contains the vintage if you're interested in extracting that feature

variety: The type of grapes used to make the wine (ie Pinot Noir)

winery: The winery that made the wine
# https://raw.githubusercontent.com/asherif844/DataScienceWithPython/master/winemag.csv

# ## import the dataframe from a .csv file locally or from a website like GitHub

# In[18]:

get_ipython().magic('ls')


# In[19]:

df = pd.read_csv('winemag.csv')


# or you can use the following instead

# In[20]:

df = pd.read_csv('https://raw.githubusercontent.com/asherif844/DataScienceWithPython/master/winemag.csv')


# In[21]:

speaker = 'ahmed'


# In[22]:

df.head()


# In[23]:

df.tail()


# In[24]:

df.head(3)


# ## slicing and dicing

# In[25]:

# use iloc (numbers)


# In[26]:

df.iloc[100:105, 0:3]


# In[27]:

# use loc (column header names)


# In[28]:

df.loc[100:105, ['country', 'description', 'taster_name']]


# In[29]:

df.describe()


# In[30]:

df.info()


# In[31]:

df.columns


# In[32]:

df[['price', 'points']]


# In[33]:

print(df.describe())
print(df.info())


# In[34]:

df.describe()


# In[35]:

df.info()


# In[ ]:




# # Data Cleansing & Data Wrangling

# <img src = 'images/wrangling 1.jpg'>

# <img src = 'images/wrangling 2.jpg'>

# https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/#620c94ce6f63

# ## Day 1 -> Section 3 -> Lab 1

# In[36]:

# drop some data that you won't need because there are so many missing values

1. Data Analysis on Dataframe using pandas
2. Identifying erroneous data
3. Replacing and Imputing Erroneous data
4. Discuss the concept of ‘Tyranny of the Mean’

# ## For Loops and If statements in Python (psst....they are much easier than in other languages)

# In[37]:

# identify null values


# In[38]:

df.info()


# In[39]:

import this


# In[40]:

for column in df.columns:
    if df[column].isnull().any():
        print(f'{column} has {df[column].isnull().sum()} null values')
    else:
        print(f'{column} has no missing values!!!')


# In[41]:

bad_columns = []


# In[42]:

for column in df.columns:
    if df[column].isnull().sum()>=20000:
        bad_columns.append(column)


# In[43]:

print(bad_columns)


# In[44]:

bad_columns.append('Unnamed: 0')


# In[45]:

bad_columns.remove('region_1')


# In[46]:

print(bad_columns)


# In[47]:

df = df.drop(bad_columns, axis = 1)


# In[48]:

df.head()


# In[49]:

df.info()


# In[50]:

df.describe()


# In[51]:

df['price'].notnull()


# In[52]:

df['price'].isnull()


# In[53]:

df[df['price'].isnull()]


# In[54]:

df[df['price'].notnull()]


# In[55]:

df.info()


# In[56]:

df = df[df['region_1'].notnull()]


# In[57]:

df.info()


# In[58]:

# impute with the mean


# In[59]:

mean_price = df['price'].mean()


# In[60]:

df['price_new'] = df['price'].fillna(mean_price)


# In[61]:

df[['price', 'price_new']]


# In[62]:

# lambda functions are great but what are they used for?


# In[63]:

df['price'].apply(lambda x: x if x>=0 else mean_price)


# In[64]:

df = df.drop(['price'], axis = 1)


# In[65]:

df.columns


# In[66]:

# many values in region_1 are also missing, do we need this column?


# In[67]:

df.groupby('region_1').size().sort_values(ascending=False)


# In[68]:

df = df.drop_duplicates()


# In[69]:

df = df.dropna()


# In[70]:

df.info()


# In[71]:

# drop all un-needed values



# In[ ]:




# In[ ]:




# ## Day 1 -> Section 4 -> Demo 4

# ### Day 1 -> Section 4 -> Lab 2
# 
1. Create a visualization to identify correlation between fields in the data set
2. Create a visualization to identify outliers

# # Visualizations

# In[72]:

country_distribution = df.groupby('country').size().sort_values(ascending = False)


# In[73]:

country_distribution


# In[74]:

country_distribution.plot(kind = 'bar')
plt.show()


# In[75]:

country_distribution.plot(kind = 'bar')
plt.show()
plt.rcParams['figure.figsize']= [5,3]

Pop Quiz
Add title and y - labels to the previous chart
# In[76]:

country_distribution.plot(kind = 'bar')
plt.xlabel('country')
plt.ylabel('Reviews')
plt.title('Distribution of Reviews by Country')
plt.show()


# In[77]:

# build a scatterplot matrix
# build a build a correlation matrix in pandas 


# In[78]:

plt.scatter(df.points, df.price_new)
plt.xlabel('point')
plt.ylabel('price')
plt.show()


# In[79]:

#histograms


# In[80]:

plt.hist(df.price_new, bins = 100)
plt.show()


# In[81]:

plt.hist(df.points)
plt.show()


# So, why is correlation useful
# 
# Correlation can help in predicting one quantity from another
# 
# This will become more relevant later as we accumulate more numeric indicators

# ## seaborn vs matplotlib

# In[82]:

df.corr()


# In[83]:

pd.scatter_matrix(df)
plt.show()
plt.rcParams['figure.figsize']= [9,6]


# In[84]:

get_ipython().system('pip install seaborn')


# In[85]:

import seaborn as sns
print(f'seaborn version: {sns.__version__}')


# In[86]:

sns.pairplot(df, kind = 'scatter')
plt.show()


# In[ ]:




# ### Final Lab for Visualization

# In[87]:

sns.lmplot( x = 'points', y= 'price_new', data = df, fit_reg = False,  hue = 'country', legend = True)
plt.show()


# ## Data analysis

# In[88]:

df.groupby('country').agg(['mean'])


# In[89]:

df.groupby('country').agg(['mean', 'min', 'max'])


# In[90]:

df.variety.unique()

