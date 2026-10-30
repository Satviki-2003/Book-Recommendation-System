import joblib
import random
import requests

class RecommendationSystem:

    def __init__(self):
        # Loading all the models
        print("Loading models... please wait.")
        self.book_sim_df = joblib.load('models/book_sim_df.pkl', mmap_mode='r')
        self.merged_df = joblib.load('models/merged_df.pkl', mmap_mode='r')
        self.user_sim_df = joblib.load('models/user_sim_df.pkl', mmap_mode='r')
        self.final_ratings = joblib.load('models/final_ratings_df.pkl', mmap_mode='r')

        print("Optimizing data structures for O(1) lookups...")
        
        # Pre-compute user histories as sets for instantaneous lookup
        self.user_history = self.merged_df.groupby('User-ID')['Book-Title'].apply(set).to_dict()
        self.famous_user_history = self.final_ratings.groupby('User-ID')['Book-Title'].apply(set).to_dict()

        # Pre-compute age groups for instant access
        self.user_age_groups = self.merged_df.set_index('User-ID')['Age_group'].to_dict()

        # Pre-compute static global lists 
        self.global_famous_books = self.final_ratings.groupby('Book-Title')['Book-Rating'].count().sort_values(ascending=False).index.tolist()
        self.global_latest_books = self.merged_df.sort_values('Year-Of-Publication', ascending=False)['Book-Title'].unique().tolist()

        # Pre-compute popular books by age group
        self.age_group_popular = self.final_ratings.sort_values('Book-Rating', ascending=False).groupby('Age_group',observed=True)['Book-Title'].apply(lambda x: list(x.unique())).to_dict()

        #Books metadata
        self.book_metadata = self.merged_df.drop_duplicates('Book-Title').set_index('Book-Title')[['Book-Author', 'Year-Of-Publication', 'Image-URL-M']].to_dict('index')


        print("Models loaded and optimized successfully!")
    
    # Recommend books based on item similarity
    def recommend_book_by_item_simm(self, recommendations, userid):
        curr_recommendations = []

        # Getting user history
        books_by_user = self.user_history.get(userid, set())
        famous_books_by_user = list(self.famous_user_history.get(userid, set()))

        if not famous_books_by_user:
            return curr_recommendations

        # Use a set for faster membership tracking during the loop
        seen_recommendations = set(recommendations)
        random_books = random.sample(famous_books_by_user, min(len(famous_books_by_user), 30))

        for book in random_books:
            if book not in self.book_sim_df.index:
                continue

            curr_book = self.book_sim_df.loc[book].idxmax()

            if (curr_book not in seen_recommendations and 
                curr_book not in books_by_user and 
                curr_book not in curr_recommendations):

                curr_recommendations.append(curr_book)

                if len(curr_recommendations) >= 5:
                    break

        return curr_recommendations

    

    # Recommend books based on user similarity
    def recommend_book_by_user_sim(self, recommendations, userid):
        if userid not in self.user_sim_df.index:
            return []

        curr_recommendations = []
        seen_recommendations = set(recommendations)
        books_by_user = self.user_history.get(userid, set())

        # Top similar users
        similar_users = self.user_sim_df.loc[userid].sort_values(ascending=False)[:25].index

        for user in similar_users:
            famous_books = list(self.famous_user_history.get(user, set()))
            if not famous_books:
                continue

            random_book = random.choice(famous_books)

            if (random_book not in seen_recommendations and 
                random_book not in books_by_user and 
                random_book not in curr_recommendations):

                curr_recommendations.append(random_book)

                if len(curr_recommendations) >= 5:
                    break

        return curr_recommendations


    # Get famous books
    def get_famous_books(self, recommendations, userid, cnt):
        books_by_user = self.user_history.get(userid, set())
        seen = set(recommendations) | books_by_user

        # Filtering using list comprehension + set lookup
        remaining_books = [book for book in self.global_famous_books if book not in seen]
        return remaining_books[:cnt]

    
    # Get book recommendations by age group
    def get_book_recommendations_by_age_group(self, recommendations, userid, cnt):
        target_grp = self.user_age_groups.get(userid)
        if not target_grp:
            return []

        books_by_user = self.user_history.get(userid, set())
        seen = set(recommendations) | books_by_user

        popular_books = self.age_group_popular.get(target_grp, [])
        remaining_books = [book for book in popular_books if book not in seen]

        return remaining_books[:cnt]


    # Get newewest books
    def get_newest_books(self, recommendations, userid, cnt):
        books_by_user = self.user_history.get(userid, set())
        seen = set(recommendations) | books_by_user

        remaining_books = [book for book in self.global_latest_books if book not in seen]
        return remaining_books[:cnt]
