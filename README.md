# Steam Playfinder

Welcome to the Steam Video Game Recommendation System project. Our mission is to develop a sophisticated game recommendation engine by harnessing the extensive dataset offered by Steam. As members of the Machine Learning Operations team, we're committed to transforming our recommendation model from concept to a powerful and efficient tool.

_____________________________________________________________________________________________________________________________________________________________________________________________
## **Table of Contents**
Challenges and Objectives\
Project Workflow\
Data Preparation\
Data Exploration\
API Functionality\
System Deployment\
ML Model Training\
Evaluation Criteria\
Data Resources\
Conclusion

_____________________________________________________________________________________________________________________________________________________________________________________________
## **Challenges and Objectives**

**Challenges:**

In my role as a Data Scientist at Steam, my main objective is to create a user-focused video game recommendation system. Dealing with complex, unstructured data and the lack of automated processes for updating product listings poses considerable challenges, but I am dedicated to developing a custom API to streamline data transformations and updates, guaranteeing smooth integration and accessibility of our recommendation system.

**Objectives:**

Create a data-powered video game recommendation system for Steam, incorporating an intuitive API with FastAPI to deliver personalized suggestions.
Perform thorough Exploratory Data Analysis (EDA) and train a machine learning model, emphasizing the utilization of advanced game similarity algorithms to offer tailored recommendations.\
_____________________________________________________________________________________________________________________________________________________________________________________________
## **Project Workflow**
**Data Preparation**\
The goal of this project was to produce an MVP that focused on refining and structuring three datasets - games, items, and reviews. The data transformation involved simplifying nested structures, removing redundant columns, and addressing missing data points to enhance API performance and model efficiency.

**Data Exploration**
I thoroughly explored each dataset through extensive EDA, with a focus on a personalized approach that guaranteed a full comprehension of the relationships and anomalies in the data. 

**API Functionality**
Utilizing the FastAPI framework, we developed an API with specific endpoints:
  1. *DeveloperItemCountAndFreeContentByYear(developer: str):* Returns the number of items and the percentage of free content per year for a given developer.
  2. *UserSpendingAndRecommendationPercentage(User_id: str):* Returns the amount of money spent by the user, the percentage of recommendation based on reviews.recommend, and the total number of items.
  3. *TopUserForGenreAndYearlyPlaytime(genre: str):* Returns the user with the highest accumulated playtime for the given genre and a list of accumulated playtime per year of release.
  4. *TopRecommendedDevelopersForYear(year: int):* Returns the top 3 developers with the most user-recommended games for the given year (reviews.recommend = True and positive comments).
  5. *DeveloperReviewsSentimentAnalysis(developer: str):* Returns a dictionary with the developer's name as the key and a list containing the total number of user reviews categorized with sentiment analysis as either positive or negative.

**Machine Learning Endpoint:**
*recommend_game(game_id, top_n=5):* Takes a product ID as input and returns a list of 5 recommended games similar to the input game.
_____________________________________________________________________________________________________________________________________________________________________________________________
## **System Deployment**
We deployed the API on Render due to its user-friendly interface and web accessibility.

**ML Model Training**
A machine learning model was crafted to emphasize game-to-game (item-to-item) similarity for recommendation generation. This model seamlessly integrates with the API, empowering users to receive personalized game suggestions.

**Evaluation Criteria**
TBD

**Data Resources**
Notebooks: The folders ETL, EDA, Feature Engineering, and ML tell the story of how this product was made if you wish to follow the process step-by-step. 
Data: The data utilized in this project can be found in the folder datos_STEAM.
Link to API
Link to public video 
_____________________________________________________________________________________________________________________________________________________________________________________________
## **Conclusion**
In summary, the Steam Video Game Recommendation System project exemplifies our dedication to utilizing data-driven insights and machine learning methodologies to enrich user interactions on the Steam platform. It also taught valuable lessons about the importance of data engineering- much of the time-consuming but necessary data transformations could be avoided with properly structured and automated data input processes.  Even still, through dedicated data preprocessing, meticulous exploration, and user-friendly API construction, we've established a strong foundation for an effective recommendation engine. Continual model refinement and assessment will enable us to provide tailored and meaningful game suggestions to Steam users, ultimately enhancing engagement and satisfaction levels.
