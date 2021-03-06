![Animal Crossing: New Horizons official artwork](/images/AnimalCrossingNewHorizons.jpg)

# Predicting Sentiment of Animal Crossing: New Horizons Reviews

## Introduction
I wanted to take on this dataset for analysis because I myself played Animal Crossing: New Horizons during the early days of the pandemic. It gave me something to do and I enjoyed it at the time, but I myself stopped playing after 9 months. Obviously, I didn’t have the same experience as others, and because of that it inspired me to research what other people thought of the game at the time and determine what the general consensus is surrounding this game, which led me to center this project around reviews for Animal Crossing: New Horizons.

This project consists of two parts: the creation of a machine learning model using the [scikit-learn](https://scikit-learn.org/stable/) library that can perform an optimized sentiment rating on reviews for Animal Crossing: New Horizons and the deployment of that model to create an application that can predict the sentiment of a review for this game based on a pure review of the game.

## Business Problem
Animal Crossing: New Horizons is one of the Nintendo Switch's best entries, having sold 33.89 million copies [(as of June 30, 2021)](https://www.nintendo.co.jp/ir/en/finance/software/index.html) and is the second-best selling game in the console's history. Having released during the beginning the COVID-19 pandemic, it served as a cultural icon and played a major part in driving Switch sales. However, since the beginning it has been a growing topic of controversy and debate. Fans of the franchise would (and still do) comment about how lackluster its features were compared to past titles, while others just become burnt out and bored quickly due to the lack of content.

I would like to look at reviews and feedback of this game to make recommendations to Nintendo on how they can improve and grow following the cultural and consumer response to Animal Crossing: New Horizons. To do this, I create a machine learning model using scikit-learn that can accurately predict a user's review sentiment based on the content in the review, then use that model to create an application where users can input their feedback for the game and receive a prediction for their review's sentiment. This would allow us to use the feedback collected from the users who submitted reviews to further investigate what we did well and what we can improve on in later titles. The feedback we receive here can also help in other areas outside of the developmeny side, such as with marketing strategies to promote this game and other future Switch titles.

## Data
The data used comes from Jesse Mostipak on [Kaggle](https://www.kaggle.com/jessemostipak/animal-crossing) and contains several datasets pertaining to Animal Crossing: New Horizons. Since we are primarily concerned with the opinions of the consumer playerbase, we will only be using the `user_reviews.csv` file, which can be accessed from the project repository's `data` folder. 

This dataset contains user-submitted reviews of Animal Crossing: New Horizons from March 2020 to May 2020 from Metacritic with scores from a scale of 0-10, 0 being the lowest and 10 being the highest. Because the goal of our model is to predict a general sentiment "label" of review text, to train our model later I assigned labels to each entry based on a score threshold for reviews defined on [the game's reviews page on Metacritic](https://www.metacritic.com/game/switch/animal-crossing-new-horizons/user-reviews). For reference, these are:
- 8-10 for "positive"
- 5-7 for "neutral"
- 0-4 for "negative"

## Methods
### Data Exploration
An initial look at the data showed us that overall the reviews are heavily polarized; most users either gave 0's or 10's for the game, so they either loved it or hated it with hardly anything in between. This reflects in the distribution of sentiment labels as well, as there is a significant imbalance between negative reviews and positive/neutral reviews.

To further analyze the review text, I use various functions from the NLTK library. First I employ a tokenizer to split each review into lists of individual words instead of whole strings. When tokenizing the text, I apply a regular expression pattern so that the tokenizer can recognize contractions as one word instead of splitting them by the root word and their apostrophes. Finally, I filter out stopwords from our reviews to remove unnecessary words that would be irrelevant in our data exploration and our modeling later, which include commonly used English words (e.g. "the", "and"), redundant words regarding the game (the word "game" and the title), and punctuation symbols. After all those steps are taken, we can create a WordCloud from the frequency distribution created using all the filtered review data:

![ACNH word cloud](/images/ACNH_word_cloud.png)

While we do see which words are common, these words don't tell us much insight by themselves. I decided to also create bigrams out of each review, which can help us see which words were often used in pairs with each other. 

`
[('one island', 1298),  
 ('island per', 1178),  
 ('per consol', 697),  
 ('per console', 694),  
 ('per switch', 613),  
 ('nd player', 371),  
 ('st player', 351),  
 ('first play', 326),  
 ('first player', 302),  
 ('second play', 271)]
`

This is a list of the top 10 bigrams that were present in our data and reveals much more about what people were saying about the game in their reviews. In this case, we can see that a lot of people criticized the game for limiting users one island per console (meaning everyone who uses the same Switch console can only access the same island!).

### Data Preparation
One last step I take before preparing the data for model fitting is lemmatizing the review text to optimize the runtime of each model and to better "group" together words with same roots but different suffixes. Once I assign this lemmatized text as our primary feature and the sentiment label as our target variable, I perform a train-test split on our selected data to split it into training and holdout sets for fitting and training our models.

## Modeling and Evaluation
For each model, these were the general steps I followed to train each model:
- Initializing the TF-IDF vectorizer to convert the text data into vectors that each classifier can take in
- Resampling the training data using SMOTE since our classes are imbalanced (recall we had more negative reviews than positive and neutral)
- Creating a pipeline with the TF-IDF vectorizer, SMOTE, and the selected model, which also helps prevent data leakage
Each model was evaluated through cross-validation and recall score. We also use confusion matrices to visualize how well our models were able to correctly predict the labels of the reviews in the holdout test set.

I trained five different models: logistic regression, multinomial Naive Bayes, decision tree, random forest, and a gradient boosted model to test which model would best predict the labels for each review. My best model ended up being the **logistic regression** model, which obtained a cross-validated score and F1 score of about 0.82 each. 

![Logistic regression confusion matrix](/images/logreg_cm.png)

### Optimization
I then ran a grid search on the logistic regression model using the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) function to define test parameters and attempt to optimize our model. Overall, optimizing the model helped it perform slightly better; it has a higher F1 score and as we can see from our confusion matrix, it was able to predict more true negatives than our initial model.

![Optimized logistic regression confusion matrix](/images/optimized_logreg_cm.png)

## Application
Our best model can be implemented into an application using Flask. If we wanted to intake newer reviews, we would have an application that would be able to rate the sentiment of a review for us without needing it to also be graded on a scale. The model implemented in the final application and the Jupyter notebook used to create said model can be found in the `backend` folder of this repository, while the app itself can be found at this link: [Animal Crossing: New Horizons Sentiment Rater](https://acnh-sentiment-rater.herokuapp.com/).

## Next Steps
Some future improvements that could be made to the application include:
- Implementing a database within the application that can archive reviews and their sentiment rating. This would help immensely with keeping track of overall sentiment and individual reviews.
- Enabling users to see statistics on the front end after submitting a review, such as how many other users agreed with them among other app users and users who submitted reviews on other websites like Metacritic.
- Expanding the application to contain a database of Switch games that includes details about each game such as trailers, reviews from other sites, etc., allowing for a wider application among more Switch games.
- Implementing a recommender system where users can be recommended games based on whether they liked the game they submit a review for (i.e. if a user gave a negative review for New Horizons they would receive recommendations based on what was popular among other people who also gave negative reviews). The recommender system would also allow for users to be recommended games based on genre.
- Multilingual support. At the moment, the app can only accurately predict the sentiment of English reviews, but as there were also reviews in different languages in our initial dataset (that were already labeled), I would like for my app to be able to take in reviews from a wider variety of consumers who may not speak English.

Additionally, we can aim to apply our application to social media posts, particularly those on Twitter and YouTube, and store feedback from there. In my experience, I have noticed a lot of buzz on social media about New Horizons, especially on those platforms whenever news or posts are released on there, so I believe it would be helpful to gather feedback from social media as well.

## Closing
For more information, you can review the full analysis in the [Jupyter notebook](acnh_review_sentiment.ipynb) or the [presentation](acnh_review_sentiment_presentation.pdf).

For any questions or additional inquiries, please contact me at [nancyho83@yahoo.com](mailto:nancyho83@yahoo.com).

## Sources
Dataset: [Animal Crossing Reviews | Kaggle](https://www.kaggle.com/jessemostipak/animal-crossing)

Libraries used: [scikit-learn](https://scikit-learn.org/stable/index.html), [imbalanced-learn](https://imbalanced-learn.org/stable/index.html), [NLTK](https://www.nltk.org/)

Repository used to deploy application on Heroku using Flask: [nancyho83/flask-model-deployment](https://github.com/nancyho83/flask-model-deployment)

Other research sources:  
- [Data on top selling Switch games (last updated on June 30, 2021 at the time of this project)](https://www.nintendo.co.jp/ir/en/finance/software/index.html)  


## Repository Structure
```
├── README.md                                            <- Top-level README for reviewers of this project 
├── acnh_review_sentiment.ipynb                          <- Narrative documentation of analysis in Jupyter notebook 
├── acnh_review_sentiment_presentation.pdf               <- PDF version of project presentation (to be added later)
├── data                                                 <- Both sourced externally and generated from code 
├── images                                               <- Both sourced externally and generated from code
└── backend                                              <- Contains backend code for this project's deployed application
```

