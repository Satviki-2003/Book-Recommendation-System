import joblib
import random
import requests


class RecommendationSystem:

  def __init__(self):
    # Keep raw data pointers uninitialized on startup to prevent RAM spikes
    self._book_sim_df = None
    self._user_sim_df = None
    self._merged_df = None
    self._final_ratings = None

    # Cached pre-computed dictionaries (built on first access)
    self._user_history = None
    self._famous_user_history = None
    self._user_age_groups = None
    self._global_famous_books = None
    self._global_latest_books = None
    self._age_group_popular = None
    self._book_metadata = None

    print('RecommendationSystem initialized successfully (Lazy-loaded mode).')

  # --- LAZY-LOAD PROPERTIES FOR FILES ---
  @property
  def book_sim_df(self):
    if self._book_sim_df is None:
      print('Loading book_sim_df from disk...')
      self._book_sim_df = joblib.load('models/book_sim_df.pkl', mmap_mode='r')
    return self._book_sim_df

  @property
  def user_sim_df(self):
    if self._user_sim_df is None:
      print('Loading user_sim_df from disk...')
      self._user_sim_df = joblib.load('models/user_sim_df.pkl', mmap_mode='r')
    return self._user_sim_df

  @property
  def merged_df(self):
    if self._merged_df is None:
      print('Loading merged_df from disk...')
      self._merged_df = joblib.load('models/merged_df.pkl', mmap_mode='r')
    return self._merged_df

  @property
  def final_ratings(self):
    if self._final_ratings is None:
      print('Loading final_ratings from disk...')
      self._final_ratings = joblib.load(
          'models/final_ratings_df.pkl', mmap_mode='r'
      )
    return self._final_ratings

  # --- LAZY-LOADED LOOKUP DICTIONARIES ---
  @property
  def user_history(self):
    if self._user_history is None:
      self._user_history = (
          self.merged_df.groupby('User-ID')['Book-Title'].apply(set).to_dict()
      )
    return self._user_history

  @property
  def famous_user_history(self):
    if self._famous_user_history is None:
      self._famous_user_history = (
          self.final_ratings.groupby('User-ID')['Book-Title']
          .apply(set)
          .to_dict()
      )
    return self._famous_user_history

  @property
  def user_age_groups(self):
    if self._user_age_groups is None:
      self._user_age_groups = (
          self.merged_df.set_index('User-ID')['Age_group'].to_dict()
      )
    return self._user_age_groups

  @property
  def global_famous_books(self):
    if self._global_famous_books is None:
      self._global_famous_books = (
          self.final_ratings.groupby('Book-Title')['Book-Rating']
          .count()
          .sort_values(ascending=False)
          .index.tolist()
      )
    return self._global_famous_books

  @property
  def global_latest_books(self):
    if self._global_latest_books is None:
      self._global_latest_books = (
          self.merged_df.sort_values('Year-Of-Publication', ascending=False)[
              'Book-Title'
          ]
          .unique()
          .tolist()
      )
    return self._global_latest_books

  @property
  def age_group_popular(self):
    if self._age_group_popular is None:
      self._age_group_popular = (
          self.final_ratings.sort_values('Book-Rating', ascending=False)
          .groupby('Age_group', observed=True)['Book-Title']
          .apply(lambda x: list(x.unique()))
          .to_dict()
      )
    return self._age_group_popular

  @property
  def book_metadata(self):
    if self._book_metadata is None:
      self._book_metadata = (
          self.merged_df.drop_duplicates('Book-Title')
          .set_index('Book-Title')[
              ['Book-Author', 'Year-Of-Publication', 'Image-URL-M']
          ]
          .to_dict('index')
      )
    return self._book_metadata

  # --- RECOMMENDATION METHODS (UNCHANGED LOGIC) ---
  def recommend_book_by_item_simm(self, recommendations, userid):
    curr_recommendations = []
    books_by_user = self.user_history.get(userid, set())
    famous_books_by_user = list(self.famous_user_history.get(userid, set()))

    if not famous_books_by_user:
      return curr_recommendations

    seen_recommendations = set(recommendations)
    random_books = random.sample(
        famous_books_by_user, min(len(famous_books_by_user), 30)
    )

    for book in random_books:
      if book not in self.book_sim_df.index:
        continue

      curr_book = self.book_sim_df.loc[book].idxmax()

      if (
          curr_book not in seen_recommendations
          and curr_book not in books_by_user
          and curr_book not in curr_recommendations
      ):
        curr_recommendations.append(curr_book)
        if len(curr_recommendations) >= 5:
          break

    return curr_recommendations

  def recommend_book_by_user_sim(self, recommendations, userid):
    if userid not in self.user_sim_df.index:
      return []

    curr_recommendations = []
    seen_recommendations = set(recommendations)
    books_by_user = self.user_history.get(userid, set())

    similar_users = (
        self.user_sim_df.loc[userid].sort_values(ascending=False)[:25].index
    )

    for user in similar_users:
      famous_books = list(self.famous_user_history.get(user, set()))
      if not famous_books:
        continue

      random_book = random.choice(famous_books)

      if (
          random_book not in seen_recommendations
          and random_book not in books_by_user
          and random_book not in curr_recommendations
      ):
        curr_recommendations.append(random_book)
        if len(curr_recommendations) >= 5:
          break

    return curr_recommendations

  def get_famous_books(self, recommendations, userid, cnt):
    books_by_user = self.user_history.get(userid, set())
    seen = set(recommendations) | books_by_user
    remaining_books = [
        book for book in self.global_famous_books if book not in seen
    ]
    return remaining_books[:cnt]

  def get_book_recommendations_by_age_group(self, recommendations, userid, cnt):
    target_grp = self.user_age_groups.get(userid)
    if not target_grp:
      return []

    books_by_user = self.user_history.get(userid, set())
    seen = set(recommendations) | books_by_user

    popular_books = self.age_group_popular.get(target_grp, [])
    remaining_books = [book for book in popular_books if book not in seen]

    return remaining_books[:cnt]

  def get_newest_books(self, recommendations, userid, cnt):
    books_by_user = self.user_history.get(userid, set())
    seen = set(recommendations) | books_by_user
    remaining_books = [
        book for book in self.global_latest_books if book not in seen
    ]
    return remaining_books[:cnt]
