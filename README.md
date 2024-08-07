# Predicting_Movie_Success_in_the_CIS
This research aimed to predict the success of movies in the CIS using machine learning tools. Kinopoisk dataset was manually collected from their website, and contains title, rating, genre, year, director, star, country, budget, company, runtime, votes, and score information of 1263 movies between 2010 and 2020.

The model selection was used to identify the best classifier for the dataset which showed that XGBoost classifier achieved the largest AUC equal to 0.835. It acquired 94.8% accuracy on training data, and 77.9% on test data. However, the model revealed overfitting which could be explained by threshold tuning on the whole training data, small size of the collected dataset, and imbalance of the classes. It also demonstrated that precision is smaller than recall, which is not preferred due to the possibility of high promotional cost of the movie in the CIS. Generally, the model shows a considerable ability to predict the success of movies due to the outstanding accuracy on the test data.

This project is done in collaboration with [@iliyar.arupzhanov](https://github.com/iliyararupzhanov).
