import numpy as np # linear algebra
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load the data
def recommend(book_name):
    users=pd.read_csv('./data/BX-Users.csv',sep=";",on_bad_lines='skip', encoding='latin-1')
    books = pd.read_csv('./data/BX-Books.csv',sep=";",on_bad_lines='skip', encoding='latin-1', low_memory=False)
    rating=pd.read_csv('./data/BX-Book-Ratings.csv',sep=";",on_bad_lines='skip', encoding='latin-1')

    #Column renaming
    books.rename(columns={'Book-Title':'title','Book-Author':'author','Year-Of-Publication':'year','Publisher':'publisher'},inplace=True) #feature engineering : changing the column names
    users.rename(columns={'User-ID':'user_id','Location':'location','Age':'age'},inplace=True)
    rating.rename(columns={'User-ID':'user_id','Book-Rating':'rating'},inplace=True)
    
    rating_with_books=rating.merge(books,on='ISBN') # Merge ratings with books based on ISBN number

    
    number_rating=rating_with_books.groupby('title')['rating'].count().reset_index() 
    number_rating.rename(columns={'rating':'number of rating'},inplace=True)

    final_ratings=rating_with_books.merge(number_rating,on='title') 
    final_ratings=final_ratings[final_ratings['number of rating']>=50] ## considering those books which has got more than 50 ratings 
    final_ratings.drop_duplicates(['user_id','title'],inplace=True) ## droping the same record 

    book_pivot=final_ratings.pivot_table(columns='user_id',index='title',values='rating') ## creating pivot table, user id vs book title, values will be ratings
    book_pivot.fillna(0,inplace=True) ## Fill na values to 0
    # print(book_pivot)
    model=NearestNeighbors(algorithm='brute') ## model
    model.fit(book_pivot)
    
    book_id=np.where(book_pivot.index==book_name)[0][0]
    distances,suggestions=model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1))
    # print(suggestions)
    
    print("The suggestions for",book_name,"are : ")

    suggestions = book_pivot.index[suggestions[0]]
    
    for index in range(1,len(suggestions)):
        print(suggestions[index])
    # for i in range(1,len(suggestions)):
    # print(book_pivot.index[suggestions[0]])

recommend('Reasonable Doubt')
# print(books.head(2))
# print(users.head(2))
# print(rating.head(2))
