# ECS 171 2022 Fall Final Project

> **Note**
> 
> **Group 30**: [Google Colab Notebook](https://colab.research.google.com/drive/1AqbJA3umP6qhSGoXuhuWKKh4dS5_917D?usp=sharing)

## Abstract

<img src="https://thumbs.gfycat.com/GlumWastefulAdouri-max-1mb.gif">

The dataset is about car accidents in the USA from 2016 to 2021, which was collected using multiple Traffic APIs ([Reference](https://smoosavi.org/datasets/us_accidents)). 

## Introduction

## Method

### Data Exploration
From the heatmap, we can see that there is not sepcific feature that has a strong correlation with `Serverity`.
<img width="1520" alt="Screenshot 2022-12-05 at 7 40 50 AM" src="https://user-images.githubusercontent.com/76938794/205678919-fe92775a-e867-4c15-b6bc-e7f47b4fd388.png">

From the two plots below, we see that Miami is the city that has the most number of car accidents and California is the state that has most accidents.

<img width="895" alt="Screenshot 2022-12-05 at 7 38 45 AM" src="https://user-images.githubusercontent.com/76938794/205678968-e446616e-3c13-4d6e-ab0c-9ec5e7836c85.png">

<img width="902" alt="Screenshot 2022-12-05 at 7 38 55 AM" src="https://user-images.githubusercontent.com/76938794/205678969-bc7a904f-045f-4601-9666-9382e7341fa6.png">

Majority of the car accidents collected in the dataset are from 2021 (53.1% of overall).

<img width="987" alt="Screenshot 2022-12-05 at 7 39 30 AM" src="https://user-images.githubusercontent.com/76938794/205678970-ea9f034c-4f90-4d50-8b7c-271222036753.png">


### Data Preprocessing
We decide to delete some categorical variables because they won't affect the severity of car accidents. In addition, in order to futher explore the reason behind a car accident, we will add some new variables, such as the driving time and the terrain, which are consisting of the `Start_Time` ,`End_Time`, `Start_Lat`, `End_Lat`, `Start_lng` and `End_lng`. What's more, considering the percentage of missing value, we decide to delete any variables which its percentage is over 15%.

Colunms to drop
1.   `ID`
2.   `Description`
3.   `Airport_Code`
4.   `Timezone`
5.   `Start_Time`
6.   `End_Time`             
7.   `Start_Lat`              
8.   `Start_Lng`             
9.   `End_Lat`            
10.  `End_Lng`
11.  `Street`
12.  `Country`
13.  `Weather_Timestamp`
14.  `City`
15.  `Zipcode`

Based on Null and missing values percentage, we also will delete:
1. `Number`
2. `Precipitation(in)`
3. `Wind_Chill(F)`

Given that we have 2845341 observations, we removed the missing values of the following labels: `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`, and `Sunrise_Sunset`.

#### Convert catorical variables using Orinal Encoder
For `Wind_Direction`, we categorize wind direction to main 4 directions: north, west, south and east
For `Weather_Condition`, the description about weather is really in detail. After some researches, we decided to categorize weather condition into 6 main categories: rain, fog, snow, cloud, clear, and thunderstorms. These are the main causes in car accidents.
For columns contain boolean values and strings, such as `state` and `side`, we converted them to dummy value using ordinal encoder.
```
ord_enc = OrdinalEncoder()

new_data.iloc[:,11:27] = ord_enc.fit_transform(new_data.iloc[:,11:27]).astype(int)
new_data.iloc[:,29:] = ord_enc.fit_transform(new_data.iloc[:,29:]).astype(int)
```

#### Test and Split
Instead of spliting the data base on certain percentage, we decided to split data base on the year the acciendts occur.
We use accidents happened before 2021 as training data, and accidents happened in 2021 as testing data.



#### Filling the missing values of categorical variables 
In the `Wind_Speed(mph)` column of the data, we decided to impute data. We used mean value of the `Win_Speed(mph)` to replace the NaN value in this column of data.

Regarding to `Humidity(%)` column of the data, we also use mean to replace nan values.

For `Visibility(mi)` column of the data, we use mean to replace nan values.

For `Temperature(F)` column of the data, we use mean to replace nan values.

For `Pressure(in)` column of the data, we use mean to replace nan values.

We do these because these columns of data contains at least 2 to 5 percents of missing values, which is a big part of a data, so we need to use mean value to replace the nan instead of drop them.

We also make severity in the data as 2 classes: high, low, which stand for 1, 0.


### Linear Regression Model
Our first model is a linear regression model, which we get a mean squared error of around 0.1045.

### Neural network Model
Our second model is a neural network model. With first three layers using activation funciton relu but with different units, the output layer using activation sigmoid, we were able to achive accuracy of 89%.

## Evaluation
We use classificaiton_report and training, testing loss/error to evaluate our model.

### How is our model in fitting graph
Confusion Matrix
, training error vs test error. From the graph, it looks not good because the training and testing line are far away from each other. It shows a overfitting.

## Discussion


## Reference:
https://stackoverflow.com/questions/38152356/matplotlib-dollar-sign-with-thousands-comma-tick-labels
https://www.kaggle.com/code/nikitagrec/usa-accidents-plotly-maps-text-classification
https://www.kaggle.com/code/tusharsingh1411/us-road-accident-eda
https://www.atmosera.com/blog/binary-classification-with-neural-networks/

## Contribution
### Author: 
Yinyin Guan:  
Randy Li:   
Shuying Li:   
Zuge Li:   
Hugo Lin:   
Jianfeng Lin:   

