![Animal Crossing: New Horizons official artwork](/images/AnimalCrossingNewHorizons.jpg)

# Predicting Sentiment of Animal Crossing: New Horizons Reviews

## Introduction
I wanted to take on this dataset for analysis because I myself played Animal Crossing: New Horizons during the early days of the pandemic. It gave me something to do and I enjoyed it at the time, but I myself stopped playing after 9 months. Obviously, I didn’t have the same experience as others, and because of that it inspired me to research what other people thought of the game at the time and determine what the general consensus is surrounding this game, which led me to center this project around reviews for Animal Crossing: New Horizons.

This project consists of two parts: the creation of a machine learning model using the [scikit-learn](https://scikit-learn.org/stable/) library that can perform an optimized sentiment rating on reviews for Animal Crossing: New Horizons and the deployment of that model to create an application that can predict the sentiment of a review for this game based on a pure review of the game.

## Business Problem
Animal Crossing: New Horizons is one of the Nintendo Switch's best entries, having sold 33.89 million copies [(as of June 30, 2021)](https://www.nintendo.co.jp/ir/en/finance/software/index.html) and is the second-best selling game in the console's history. Having released during the beginning the COVID-19 pandemic, it served as a cultural icon and played a major part in driving Switch sales. However, since the beginning it has been a growing topic of controversy and debate. Fans of the franchise would (and still do) comment about how lackluster its features were compared to past titles, while others just become burnt out and bored quickly due to the lack of content.

If I were given the task to help Nintendo's dev team figure out new updates for New Horizons or help contribute ideas for the next title in the Animal Crossing series, I would want to look at reviews and feedback surrounding New Horizons to let them know what they did well and what they could improve on. To do this, first I create and test a machine learning model using modules from the scikit-learn library that can accurately predict a user's review sentiment based on the content in the review. We can then use the best model from our test models and create an application where users can input their own reviews for the game and receive a prediction for the sentiment of their review. This would allow us to use the feedback collected from the users who submitted reviews to further investigate what we did well and what we can improve on in later titles.

## Data
The data used comes from Jesse Mostipak on [Kaggle](https://www.kaggle.com/jessemostipak/animal-crossing) and contains several datasets pertaining to Animal Crossing: New Horizons. Since we are primarily concerned with the opinions of the consumer playerbase, we will only be using the `user_reviews.csv` file, which can be accessed from the project repository's `data` folder. 

This dataset contains user-submitted reviews of Animal Crossing: New Horizons from March 2020 to May 2020 from Metacritic with scores from a scale of 1-10, 1 being the lowest and 10 being the highest. Because the goal of our model is to predict a general sentiment "label" of review text, to train our model later I assigned labels to each entry based on a score threshold for reviews defined on [the game's reviews page on Metacritic](https://www.metacritic.com/game/switch/animal-crossing-new-horizons/user-reviews). For reference, these are:
- 8-10 for "positive"
- 5-7 for "neutral"
- 0-4 for "negative"

## Methods
### Data Exploration
- Reviews are heavily polarized, also significant imbalance between negative reviews and positive/neutral
- NLTK; tokenizer, regex
- Most common words, but that doesn't tell us much; creating bigrams reveals more info like how a lot of people were complaining about the limitation of one island per console
### Data Preparation
- Lemmatizing text to optimize runtime of model and to better "group" together words with same roots but different suffixes
- Train-test split

## Modeling and Evaluation
For all of the models I trained, these were the general steps I followed to run each model:
- Initializing the TF-IDF vectorizer
- Resampling the training data using SMOTE since our classes are imbalanced (recall we had more negative reviews than positive and neutral)
- Creating a pipeline with the TF-IDF vectorizer, SMOTE, and the selected model, which also helps prevent data leakage
Each model is evaluated through cross-validation and recall score. We also use confusion matrices to visualize how well our models were able to correctly predict the labels of the reviews in the holdout test set.

The first model I ran utilized a logistic regression classifier, one of the simplest machine learning classifiers available in scikit-learn. It performed very well, obtaining an average cross-validated score and recall score of ~82% each.

- Multinomial Naive Bayes
- Decision tree
- Random forest

### Optimization
- Grid search

## Application
Our best model can be implemented into an application using Flask. If we wanted to intake newer reviews, we would have an application that would be able to rate the sentiment of a review for us without needing it to also be graded on a scale.

## Next Steps
Some future improvements that could be made to the application include:
- Implementing a database within the application that can archive reviews and their sentiment rating. This would help immensely with keeping track of overall sentiment and individual reviews.
- Enabling users to see statistics on the front end after submitting a review, such as how many other users agreed with them among other app users and users who submitted reviews on other websites like Metacritic.
- Expanding the application to contain a database of Switch games that includes details about each game such as trailers, reviews from other sites, etc., allowing for a wider application among more Switch games.
- Implementing a recommender system where users can be recommended games based on whether they liked the game they submit a review for (i.e. if a user gave a negative review for New Horizons they would receive recommendations based on what was popular among other people who also gave negative reviews). The recommender system would also allow for users to be recommended games based on genre.
- Multilingual support. At the moment, the app can only accurately predict the sentiment of English reviews, but as there were also reviews in different languages in our initial dataset (that were already labeled), I would like for my app to be able to take in reviews from a wider variety of consumers who may not speak English.

## Closing
For more information, you can review the full analysis in the [Jupyter notebook](animal-crossing-review-sentiment.ipynb) or the [presentation].

For any questions or additional inquiries, please contact me at [nancyho83@yahoo.com].

## Sources
Dataset: [Animal Crossing Reviews | Kaggle](https://www.kaggle.com/jessemostipak/animal-crossing)

## Repository Structure
├── README.md                                            <- Top-level README for reviewers of this project 
├── animal-crossing-review-sentiment.ipynb               <- Narrative documentation of analysis in Jupyter notebook 
├── presentation.pdf                                     <- PDF version of project presentation (to be added later)
├── data                                                 <- Both sourced externally and generated from code 
├── images                                               <- Both sourced externally and generated from code
└── backend                                              <- Both sourced externally and generated from code

