from flask import Flask, render_template, request
import pickle
import pandas as pd
import random
import requests
import recommendation_system

app = Flask(__name__)

rec_sys=recommendation_system.RecommendationSystem()
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations/<int:user_id>', methods=['GET'])
def recommend(user_id):
    recommendations = []
    userid=int(user_id)
  
    # shelf1 : item-item simmilarity
    shelf1=rec_sys.recommend_book_by_item_simm(recommendations,userid)
    recommendations.extend(shelf1)

    # shelf2 : user-user simmilarity
    shelf2=rec_sys.recommend_book_by_user_sim(recommendations,userid)
    recommendations.extend(shelf2)

    # shelf3 : famous books
    shelf3=rec_sys.get_famous_books(recommendations,userid,5)
    recommendations.extend(shelf3)
  
    # shelf4 : books by age group
    shelf4=rec_sys.get_book_recommendations_by_age_group(recommendations,userid,5)
    recommendations.extend(shelf4)

     # shelf5 : newest books
    shelf5=rec_sys.get_newest_books(recommendations,userid,5)
    recommendations.extend(shelf5)

    # Fast Metadata fetching using Helper Function
    def hydrate_shelf(shelf_books):
        hydrated = []
        for book in shelf_books:
            meta = rec_sys.book_metadata.get(book)
            if meta:
                img_url = meta['Image-URL-M'] if meta['Image-URL-M'] else "/static/images/Generic-book-cover.png"

                hydrated.append([
                    book, 
                    meta['Book-Author'], 
                    meta['Year-Of-Publication'], 
                    img_url
                ])
        return hydrated

    # Hydrate all shelves
    fin_shelf1 = hydrate_shelf(shelf1)
    fin_shelf2 = hydrate_shelf(shelf2)
    fin_shelf3 = hydrate_shelf(shelf3)
    fin_shelf4 = hydrate_shelf(shelf4)
    fin_shelf5 = hydrate_shelf(shelf5)

    return render_template(
        'result_page.html', 
        userId=userid, 
        shelf1=fin_shelf1, 
        shelf2=fin_shelf2, 
        shelf3=fin_shelf3, 
        shelf4=fin_shelf4, 
        shelf5=fin_shelf5
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)