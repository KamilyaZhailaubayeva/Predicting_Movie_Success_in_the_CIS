<h1 align="center">
 Predicting the Success of the Movie in the CIS 
</h1>

This research aimed to predict the success of movies in the CIS using machine learning tools. Kinopoisk dataset was manually collected from their website, and contains title, rating, genre, year, director, star, country, budget, company, runtime, votes, and score information of 1263 movies between 2010 and 2020.

## Introduction
Movies are a popular source of entertainment, offering a vast selection of films to choose from. However, to improve the user experience, a better recommendation system is required. Making successful films can be very profitable as the film industry is one of the largest businesses in the world. Therefore, predicting the success of the movie is essential for investors when making decisions. Utilization of machine learning techniques can help us to predict a movie's success based on various factors such as genre, actors, actresses, directors, and release date. Aim of our study is to predict a success of a particular movie in the Commonwealth of Independent States (CIS), where movie platforms in Russian language, such as Kinopoisk, Okko, and Ivi have their popularity. This prediction can help movie makers understand the types of movies, which are the most successful for people living in the CIS region.

Most of the research related to movie classification are directed towards the classification of movies or generation of movie recommendation systems, based on reviews of viewers, visual features but not the attributes of the movie, which are available before the release of the movie. Even though some researchers tried to classify the movie based on the success using the mentioned attributes of the movies using IMDb dataset, no other research was found using Kinopoisk dataset. Therefore, this research is quite novel in the prediction of movie success based on preferences of people in the CIS.

## Methodology
### Dataset Collection
To construct the dataset of this research, firstly, IMDb dataset with available title, rating, genre, year, director, star, country, budget, company, and runtime information was obtained from [here](https://github.com/danielgrijalva/movie-stats/blob/master/movies.csv). This dataset is taken from official IMDb website, and it consists of 7668 movies between 1980 and 2020. Though the dataset has almost all budget information, few were missing. Since budget data of movies are tended to have large number of outliers, they were replaced by **Median Imputation**. In order to ease a manual collection of Kinopoisk dataset, only movies from 2010 to 2020 were chosen. Then, score and votes information were acquired from [Kinopoisk](https://www.kinopoisk.ru/). Finally, two datasets were combined into a total of 1263 movies. 

### Description of Dataset
The Table 1 provides a description and type of each feature of the dataset. There are 11 features in total: rating, genre, year, director, star, company, country, budget, runtime, and votes. The output is to find whether the movie will be successful or unsuccessful based on score information which is ranged between 0 and 10.
<p align="center">
  Table 1. Description and type of each feature in the dataset.
</p>

<div align="center">
  
| Feature      | Description           | Type          |
| :---         |         :---:         |          ---: |
|Rating|Rate a movie’s suitability for certain audiences based on its content. Four major categories: R, PG-13, PG, others| Categorical|
|Genre|Nine major categories: action, adventure, animation, biography, comedy, crime, drama, horror, others|Categorical|
| Year         | Year in which movie was released | Numerical |
| Director     | Two categories: directed one movie, directed two or more movies  | Categorical |
| Star         | Two categories: played in one movie, played in two or more movies| Categorical |
| Company      | Two categories: produced one movie, produced two or more movies  | Categorical |
| Country      | Three major categories: the U.S., the United Kingdom, others     | Categorical |
| Budget       | Budget of the movie in dollars                                   | Numerical   |
| Runtime      | Duration of the movie in minutes                                 | Numerical   |
| Votes        | Number of people who voted for the movie on Kinopoisk            | Numerical   |

</div>

### Classifiers
We utilized 8 classifiers that have been previously used in studies predicting movie success. These include Gaussian Naïve Bayes (GNB), K-nearest neighbors (KNN), Logistic Regression with 𝐿2 ridge penalty (LRR), Random Forest classifier (RFC), Adaptive Boosting classifier (ADB), Gradient Boosting classifier (GBC), eXtreme Gradient Boosting classifier (XGB), and Perceptron (PER). Table 2 shows their hyperparameters along with the their values used in the model selection section for these classifiers.

<p align="center">
  Table 2. Classifiers and their hyperparameter values chosen for grid search with cross-validation model selection.
</p>

<div align="center">

| Classifiers               | Hyperparameter      | Candidate Hyperparameter Space |
| :---                      |         :---:       | ---: |
| Gaussian Naive Bayes (GNB)|      -              | -    |
| K-nearest Neighbors (KNN) | number of neighbors | 3, 5 |
| Logistic Regression (LRR) | penalty             | 𝐿2   |
|                           | regularization parameter C | 100, 10, 1.0, 0.1, 0.01 |
| Random Forest Classifier (RFC) | number of estimators | 10, 100, 1000 |
|                                | maximum features     | 'auto', 'sqrt', 'log2' |
| Adaptive Boost Classifier (ADB)| number of estimators | 10, 100, 1000 |
|                                | learning rate        | 0.001, 0.01, 0.1 |
| Gradient Boosting Classifier (GBC)      | number of estimators | 10, 100, 1000 |
|                                         | learning rate        | 0.001, 0.01, 0.1 |
| eXtreme Gradient Boosting Classifier (XGB)    | maximum depth        | 5, 10, 100       |
|                                               | number of estimators | 10, 100, 1000    |
|                                               | learning rate        | 0.001, 0.01, 0.1 |
| Perceptron (PER) | alpha    | 0.0001, 0.001, 0.01 |
|                  | penalty  | 𝐿2, 𝐿1, None        |

</div>

### Model Selection
The model selection is performed using the 5-fold cross-validation (5-CV) based on the AUC performance metric. It uses a training data that is 80% of the dataset. Other portion of the dataset, which is a test data, is used in evaluating the selected model. The Fig.1 illustrates the model selection process of this research. For each model, training data is, firstly, split into 5 folds where one fold is held-out for validation. After that, the candidate hyperparameters of the model are used to develop classifier based on other 4 folds. This procedure is iterated 5 times holding out each fold. After the last iteration, the overall CV AUC is determined for these candidate hyperparameters. Then, this process is repeated for the next candidate hyperparameters until all of them are covered. Similarly, the next model goes through these steps. Finally, the model with the lowest error is selected to construct classifier on initial training data. Its performance metrics are then acquired for training data and test data.

<p align="center">
  Figure 1. The model selection process for Kinopoisk dataset.
</p>

<p align="center">
  <img width="550" src=Figure1.png>
</p>

The selection of the model was based on the AUC metric, which is independent of any specific decision threshold. Once the model was selected, the decision threshold was further finetuned using the training data to maximize the accuracy score. After tuning the decision threshold, the final classifier was constructed using the full training data and evaluated on the test data based on several performance metrics.

## Results
Table 3 shows the 5-fold CV estimates of the AUC for each classifier. As can be observed from Table 3 XGB outperformed other classifiers, achieving the highest AUC of 0.835.

<p align="center">
  Table 3. The estimated ROC AUC mean over 5 folds of 5-fold cross-validation obtained on the training set. Classifier with the highest AUC mean is defined in bold.
</p>

<div align="center">
  
| Classifiers               | Hyperparameter      | ROC AUC |
| :---                      |         :---:       |    ---: |
| Gaussian Naive Bayes (GNB)|      -              | 0.756   |
| K-nearest Neighbors (KNN) | number of neighbors: 5 | 0.684 |
| Logistic Regression (LRR) | penalty: 𝐿2            | 0,790   |
|                                | regularization parameter C: 100 ||
| Random Forest Classifier (RFC) | number of estimators: 50 | 0.811 |
|                                | maximum features: 'log2' |       |
| Adaptive Boost Classifier (ADB)| number of estimators: 1000 | 0.829 |
|                                | learning rate: 0.1         |       |
| Gradient Boosting Classifier (GBC)      | number of estimators: 100 | 0.814 |
|                                         | learning rate: 0.1        |       |
| ***eXtreme Gradient Boosting Classifier (XGB)***  | ***maximum depth: 5***          | ***0.835*** |
|                                                   | ***number of estimators: 100*** |             |
|                                                   | ***learning rate: 0.1***        |             |
| Perceptron (PER) | alpha: 0.0001    | 0.719 |
|                  | penalty: 𝐿1      |       |

</div>

After performing model selection, the best classification algorithm (XGBoost) classifier with the hyperparameter values obtained in the model selection process was fine-tuned using the training data to maximize the accuracy score. The resulting classifier was then used to construct a final model on the entire training set. Then to evaluate the performance of the final trained classifier was evaluated on the test set using the following performance metrics: accuracy, recall, precision, and F1 score. Table 4 shows the results of the model evaluation.

<p align="center">
  Table 4. Performance metrics of XGBoost classifier evaluated on the test data.
</p>

<div align="center">
  
 | Classifier | Threshold | Accuracy | Recall | Precision | F1 score |
 | :---:      | :---:     | :---:    | :---:  | :---:     | :---:    |
 | XGB        | 0.493     | 0.779    | 0.849  | 0.808     | 0.828    |

</div>
XGB achieved accuracy of 0.779, indicating a considerable ability to predict the success of the movies. However, results show that model has higher recall than precision. In the trade-off between precision is more desirable than recall, as the cost of promoting movies in CIS by production company can be high.

To evaluate the overfitting or underfitting in a model, performance metrics were calculated for training set. Table 5 shows the comparison of performance metrics evaluated on training set and test set.

<p align="center">
  Table 5. Performance metrics of XGBoost classifier evaluated on the training and test data.
</p>

<div align="center">
  
 |              | Accuracy | Recall | Precision | F1 score |
 | :---:        | :---:    | :---:  | :---:     | :---:    |
 | Training set | 0.948    | 0.972  | 0.948     | 0.959    |
 | Test set     | 0.779    | 0.849  | 0.808     | 0.828    |

</div>

As it can be seen from the Table 5 constructed model performs exceptionally well on the training data in comparison with test data meaning that model is overfitting. Possible reasons for the overfitting can be threshold tuning on the entire training set, small size of the dataset and imbalance of the classes.


## Conclusion
The model selection was used to identify the best classifier for the dataset which showed that XGBoost classifier achieved the largest AUC equal to 0.835. It acquired 94.8% accuracy on training data, and 77.9% on test data. However, the model revealed overfitting which could be explained by threshold tuning on the whole training data, small size of the collected dataset, and imbalance of the classes. It also demonstrated that precision is smaller than recall, which is not preferred due to the possibility of high promotional cost of the movie in the CIS. Generally, the model shows a considerable ability to predict the success of movies due to the outstanding accuracy on the test data.

## P.S. :black_nib:
This project is done in collaboration with [@iliyar.arupzhanov](https://github.com/iliyararupzhanov).
