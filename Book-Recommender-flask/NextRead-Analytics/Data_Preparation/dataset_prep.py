import pandas as pd

# Reading all files
df_books=pd.read_csv("Dataset/Books.csv",encoding='latin-1',dtype={3:str})
df_ratings=pd.read_csv("Dataset/Ratings.csv",encoding='latin-1')
df_users=pd.read_csv("Dataset/Users.csv",encoding='latin-1')

# Handling null values
df_books.fillna({
    'Book-Author':'Unknown',
    'Publisher':'Unknown',
    'Image-URL-L':'None'
},inplace=True)

# Handling irrelevant values
df_books.loc[df_books['Year-Of-Publication']=='DK Publishing Inc','Year-Of-Publication']='0'
df_books.loc[df_books['Year-Of-Publication']=='Gallimard','Year-Of-Publication']='0'

# Changing datatype to int
df_books['Year-Of-Publication']=df_books['Year-Of-Publication'].astype(int)

# Handling outliers in age
q3=df_users['Age'].quantile(0.75) # q3
q1=df_users['Age'].quantile(0.25) # q1

# UL & LL
ul=q3+1.5*(q3-q1)
ll=q1-1.5*(q3-q1)

# Capping and flooring
df_users.loc[df_users['Age']>ul,'Age']=ul
df_users.loc[df_users['Age']<ll,'Age']=ll

df_users.loc[df_users['Age']==0,'Age']=None

# Median
age_median=df_users['Age'].median()

# Replacing null with median
df_users.fillna({'Age':age_median},inplace=True)

# Binning ages of users
bins = [0, 12, 19, 30, 50, 100]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
df_users['Age_group'] = pd.cut(df_users['Age'], bins=bins, labels=labels,include_lowest=True)

# Splitting Location for better clarity
df_users[['City','State','Country']]=df_users['Location'].str.split(',',expand=True,n=2)

# Dropping old column
df_users.drop('Location',inplace=True,axis=1)

# Handling nulls in city and country
df_users.fillna({
    'City':'Unknown',
    'Country':'Unknown'
},inplace=True)

# Merging ratings and books dataframe
ratings_with_name=df_ratings.merge(df_books,on='ISBN')
merged_df=df_users.merge(ratings_with_name,on='User-ID')

# Identify users who have given more than 200 ratings
x = merged_df.groupby('User-ID').count()['Book-Rating'] > 200
experienced_users = x[x].index

# Filter the dataframe
filtered_rating = merged_df[merged_df['User-ID'].isin(experienced_users)]

# resetting index
filtered_rating.reset_index(drop=True,inplace=True)

# Identify books with more than 50 ratings
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

# Final filtered dataframe
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

# pivot table
pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')

pt.fillna(0,inplace=True) # Filling null values

from sklearn.metrics.pairwise import pairwise_distances

import numpy as np

import random

# item-item simm
book_sim=1-pairwise_distances(pt,metric='cosine')

# creating a dataframe using book similarity scores
book_sim_df=pd.DataFrame(book_sim,index=pt.index,columns=pt.index)

# Filling diagonals as 0
np.fill_diagonal(book_sim,0)

# transforming dataframe for user-user simm
transformed_pt=pt.T

# User-user simm
user_sim=1-pairwise_distances(transformed_pt.values,metric='cosine')

# user-simm df
user_sim_df=pd.DataFrame(user_sim,index=transformed_pt.index,columns=transformed_pt.index)

# Filling diagonals as 0
np.fill_diagonal(user_sim,0)

import pickle

# Files
files={
    "../models/merged_df.pkl":merged_df,
    "../models/final_ratings_df.pkl":final_ratings,
    "../models/book_sim_df.pkl":book_sim_df,
    "../models/user_sim_df.pkl":user_sim_df
}

for filename,obj in files.items():
  with open(filename,'wb') as f:
    pickle.dump(obj,f)
  print(f"Successfully created: {filename}")

