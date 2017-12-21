# Titanic_Survival_Exploration

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/c73bee6015bf485d8ce4184cbb135b03)](https://www.codacy.com/app/prateekkol21/titanic_survival_exploration?utm_source=github.com&utm_medium=referral&utm_content=prateekiiest/titanic_survival_exploration&utm_campaign=badger)
[![Build status](https://ci.appveyor.com/api/projects/status/vps8mifg5qyqqu7g?svg=true)](https://ci.appveyor.com/project/prateekiiest/titanic-survival-exploration)
[![Maintainability](https://api.codeclimate.com/v1/badges/4c2f2473dae6c52f64a1/maintainability)](https://codeclimate.com/github/prateekiiest/titanic_survival_exploration/maintainability)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1098228.svg)](https://doi.org/10.5281/zenodo.1098228)

<a href="https://github.com/prateekiiest/boston_housing"><img style="position: relative; top: 0; left: 0; border: 0;" src="https://68.media.tumblr.com/38ae897f20630ef88e6484dea00db3b3/tumblr_mm8fhitR3u1rwwvg9o1_500.gif" alt=" Fork this repo" data-canonical-></a>



### This repository contains project file for Project 0 - Titanic Survival Exploration as part of Udacity's Machine Learning Nanodegree.

---------------------------------------------------------------------------------

### KWOC

We are glad to partner with IIT Kharagpur as a part of the Kharagpur Winter of Code. We are proud to host this Open Source event during the winter months and we hope you have a great winter this year.

#### See Project Ideas [here](https://github.com/prateekiiest/titanic_survival_exploration/wiki/Winter-of-Code-Project)


-------------------------------------------------------------------------


### Description

>The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew.  This sensational tragedy shocked the international community and led to better safety regulations for ships.

>One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.  Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

>In this problem, we ask you to complete the analysis of what sorts of people were likely to survive.  In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.




In this optional project, you will create decision functions that attempt to predict survival outcomes from the 1912 Titanic disaster based on each passenger’s features, such as sex and age. Start with a simple algorithm and increase its complexity until you are able to accurately predict the outcomes for at least 80% of the passengers in the provided data. This project will introduce you to some of the concepts of machine learning as you start the Nanodegree program.




### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

Udacity recommends our students install [Anaconda](https://www.continuum.io/downloads), i pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

### Code

Template code is provided in the notebook `titanic_survival_exploration.ipynb` notebook file. Additional supporting code can be found in `titanic_visualizations.py`. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project.

#### This Notebook will show basic examples of:
#### Data Handling
*   Importing Data with Pandas
*   Cleaning Data
*   Exploring Data through Visualizations with Matplotlib

#### Data Analysis
*    Supervised Machine learning Techniques:
    +   Logit Regression Model
    +   Plotting results
    +   Support Vector Machine (SVM) using 3 kernels
    +   Basic Random Forest
    +   Plotting results

#### Valuation of the Analysis
*   K-folds cross validation to valuate results locally
*   Output the results from the IPython Notebook to Kaggle



### Run

In a terminal or command window, navigate to the top-level project directory `titanic_survival_exploration/` (that contains this README) and run one of the following commands:

```ipython notebook titanic_survival_exploration.ipynb```
```jupyter notebook titanic_survival_exploration.ipynb```

This will open the iPython Notebook software and project file in your browser.

## Data

The dataset used in this project is included as `titanic_data.csv`. This dataset is provided by Udacity and contains the following attributes:

- `survival` ? Survival (0 = No; 1 = Yes)
- `pclass` ? Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
- `name` ? Name
- `sex` ? Sex
- `age` ? Age
- `sibsp` ? Number of Siblings/Spouses Aboard
- `parch` ? Number of Parents/Children Aboard
- `ticket` ? Ticket Number
- `fare` ? Passenger Fare
- `cabin` ? Cabin
- `embarked` ? Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

## Results
 Check here [Udacity Reviews](https://github.com/prateekiiest/titanic_survival_exploration/blob/master/Udacity_Reviews_titanic.pdf)

## Contribution

See CONTRIBUTING.md

### Some Video Resources
- [ ] [Coursera Lectures by Andrew Ng](https://www.coursera.org/learn/machine-learning/lecture/zcAuT/welcome-to-machine-learning) are not very mathematically heavy and provide a good introduction to ML algorithms.
- [ ] [Standford Lectures](https://www.youtube.com/watch?v=UzxYlbK2c7E)
- [ ] [Unsupervised Learning](https://www.coursera.org/learn/machine-learning/lecture/olRZo/unsupervised-learning)
- [ ] [Udacity Lectures (Intro to ML)](https://in.udacity.com/course/intro-to-machine-learning--ud120)
- [ ] [Udacity Lectures (ML)](https://in.udacity.com/course/machine-learning--ud262)
- [ ] [Sentdex Lectures on Introduction to ML](https://www.youtube.com/watch?v=OGxgnH8y2NM&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)
- [ ] [Udemy Lectures on ML using Python as well as R](https://www.udemy.com/machinelearning/)
- [ ] [Udemy Course on various Data science and Machine Learning Techniques](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/)
- [ ] Machine Learning A-Z™: Hands-On Python & R In Data Science (https://www.udemy.com/machinelearning/)
- [ ] EdX: Learning From Data (Introductory Machine Learning) (https://www.edx.org/course/learning-data-introductory-machine-caltechx-cs1156x-0)

### Online Reading Material
- [ ] Advanced Introduction to Machine Learning (http://www.cs.cmu.edu/~epxing/Class/10715/lecture.html)
- [ ] CS229: Machine Learning (https://cs229.stanford.edu/)

#### Happy Coding                                                                                           -- Prateek Chanda

