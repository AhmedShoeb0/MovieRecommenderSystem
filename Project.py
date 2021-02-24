import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#To show all columns
pd.set_option('display.max_columns', None)
#To expand the dataframe
pd.set_option('display.width', 1000)
#To resize the column width
pd.set_option('display.max_colwidth', 20)


#reading tsv files  tsv="Tab-separated values"
tsv1='movies.tsv'
df1 = pd.read_csv(tsv1, sep='\t', low_memory=False)
tsv2='ratings.tsv'
df2 = pd.read_csv(tsv2, sep='\t', low_memory=False)
#merging the first 2 datasets and drop some columns
df3=pd.merge(df1,df2, on=['titleId'], how='inner').drop(['titleType','startYear','endYear'],axis=1)
#reading csv file   csv="Comma-separated values"   
df4 = pd.read_csv('metadata.csv', low_memory=False)
#merging the third dataset with the result dataset "df3"
dataset=pd.merge(df3,df4, on=['titleId'], how='inner')
#output final dataset info
print(dataset.info())
print()
#output final dataset first 5 rows
print(dataset.head())
print()
print()

#Now we need to calculate the weighted rating for each movie.
#The weighted rating will make sure that a movie with a 9 rating from 100,000 voters
#gets a higher score than a movie with the same rating but a mere few hundred voters.
#In other words, better and more accurate ranking.

#Firstly,we need these following valuse to calculate the weighted rating:
#Votes_number > is the number of votes for the movie.
#minimum_votes > is the minimum votes required to be to be recommended.
#movie_rating is > the average rating of the movie.
#averageRating > is the mean vote across the whole report.


# Calculate mean of averageRating column
averageRating = dataset['averageRating'].mean()
print('averageRating column mean = ',averageRating)
print()
# Calculate the minimum number of votes required to be recommended
minimum_votes =dataset['numVotes'].quantile(0.90)
print('the minimum number of votes required to be recommended= ',minimum_votes)
print()
#Note that we can consider it as any number as since there is no right value for minimum_votes.
#quantile function returns values at the given quantile over requested percentile using frequency distribution.

# Filter out all qualified movies into a new DataFrame
q_movies = dataset.copy().loc[dataset['numVotes'] >= minimum_votes]


#shape function return the number of (rows,columns) of the dataframe
print('original dataset (rows,columns) = ', dataset.shape)
print('current dataset (rows,columns) = ', q_movies.shape)
print()


#Now after we got all the vavlues we have to define a function that calculates the weighted rating

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=minimum_votes, C=averageRating):
    Votes_number = x['numVotes']
    movie_rating = x['averageRating']
    # Calculation based on the IMDB formula
    return (Votes_number/(Votes_number+minimum_votes) * movie_rating) + (minimum_votes/(Votes_number+minimum_votes) * averageRating)



#Now we'll define a new feature 'score' and calculate its values with weighted_rating() function that we defined above
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sorting movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#We replace the old index with new sequential index and drop the old one
#we will need the new index in content-based model
q_movies=q_movies.reset_index(drop=True)

print()
print()

#Now we'll print the top 15 movies in our dataset
print('Top 15 movies in our dataset :')
print(q_movies.head(15))

"""                                                   CONTENT-BASED MODEL                                                        """
#content-based model construction:
#Recommendations in this model will be based on the similarity of movies' plot descriptions (overviews).
#We need to compute the word vectors of each overview or document as it is not possible to compute 
#the similarity between any two overviews in their raw forms.

#first of all, we need to Import TfIdfVectorizer from scikit-learn
#We Define a TF-IDF Vectorizer Object and Remove all english stop words
tfidf = TfidfVectorizer(stop_words='english')

#Then we need to replace NaN with an empty string.
#fillna() function is used to fill NA/NaN values using the specified method.
q_movies['overview'] = q_movies['overview'].fillna('')

#Now we construct the required TF-IDF matrix by fitting and transforming the data
#fit_transform means to do some calculation and then do transformation
tfidf_matrix = tfidf.fit_transform(q_movies['overview'])

#Note that we use the filtered dataset "q_movies" for "content-based model" 
#to recommend only high rate movies that are relevant to our input
#it's up to you as you can use the original dataset but the calculations will be huge and will take much space and time :)

#Now after we got the TF_IDF matrix we need to calculate the cosine similarity score

#We need to Import linear_kernel
#Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(q_movies.index, index=q_movies['primaryTitle']).drop_duplicates()


#Now we need to define a function that takes in a movie title as an input and outputs a list of the 10 most similar movies.

#Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return q_movies['primaryTitle'].iloc[movie_indices]
print()
title=input("Enter Movie title: ")
print('The following list contains top 10 movies that are similar to your input: ')

#Erorr handling
try:
    print(get_recommendations(title))
except Exception:
    print('Title is wrong or movie is not found')
    title=input("Enter Movie title: ")
    print(get_recommendations(title))

print()


"""                                              COLLABORATIVE FILTERING MODEL                                                   """
#Reading csv files
movies=pd.read_csv('movies_collaborative.csv')
ratings=pd.read_csv('ratings_collaborative.csv')

collaborative_dataset=pd.merge(movies,ratings).drop(['timestamp'],axis=1)

#We use this dataframe to make our new dataframe that we will use in constructing the model
#We put UserId as index(rows) , movies titles as columns and the values of the table will be the rating of each user on each movie
user_ratings= collaborative_dataset.pivot_table(index=['userId'],columns=['title'],values='rating')
#fillna replace each NaN/Na value with the value you place, 0 in our case
#we droped movies that got rated by less than 10 users
user_ratings=user_ratings.dropna(thresh=10,axis=1).fillna(0)

#Now we build similarity matrix for movies in datafram
#corr() is used to find the pairwise correlation of all columns in the dataframe
#Pearson method calculates the effect of change in one variable when the other variable changes.
movies_similarity_df=user_ratings.corr(method='pearson')

#Recommendations function construction
def get_similar_movies(movie_name,user_rating):
    #Getting the serise of scores for the entered title and multiply each score by user_rating-2.5
    similar_score =movies_similarity_df[movie_name]*(user_rating-2.5)

    #Sorting the values descendingly
    similar_score = similar_score.sort_values(ascending=False)
    
    return similar_score


#it's the user preferences u can change movie titles , put your rate and change number o movies if you want.
#the recommender use these preferances to make recommendations similar to your choice.
User_preferences= [('Fight Club (1999)',8),('Forrest Gump (1994)',7),('(500) Days of Summer (2009)',3),('Seven Pounds (2008)',9),('Crank (2006)',5)]

#defining a new dataframe
Similar_movies=pd.DataFrame()

#for loop to pass user preferences and append them to the new datafram
for movie,rating in User_preferences:
    Similar_movies = Similar_movies.append(get_similar_movies(movie,rating),ignore_index=True)
    
#Sum all duplicates (movies and their scores) to get final order that is more accurate to your preferences
Similar_movies=Similar_movies.sum().sort_values(ascending=False)

#Output top 15 movies
print('Top 15 movies that are similar to user previous preferences: ')
print(Similar_movies.head(15))
