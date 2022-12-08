# ECS 171 2022 Fall Final Project

> **Note**
> 
> **Group 30**: 
> * Link: [Google Colab Notebook](https://colab.research.google.com/drive/1AqbJA3umP6qhSGoXuhuWKKh4dS5_917D?usp=sharing)
> * Updated link: [Google Colab Notebook](https://colab.research.google.com/drive/1FxCVbPT-rJyXKdZTW4e3w5YCoAarldDA?usp=sharing)

<img src="https://thumbs.gfycat.com/GlumWastefulAdouri-max-1mb.gif">

The dataset is about car accidents in the USA from 2016 to 2021, which was collected using multiple Traffic APIs ([Reference](https://smoosavi.org/datasets/us_accidents)). 

## Introduction

Car is the most popular and convenient transportation nowadays, bringing accidents to people’s lives. Worldwide, approximately 1.3 million people die in car accidents each year. However, the accident could be avoided if we knew what factors would increase the possibility of a car accident. Therefore, in our project, we decided to analyze the US Accidents (2016 - 2021) dataset, by utilizing the dataset of car accidents covering 49 states of the USA, this project explores the factors that affect car accidents. These factors, such as weather, humidity, wind speed, weather conditions, etc, have a significant impact on severity. This project develops uses EDA to visualize data and help us better understand it, and finally build reliable models to predict it. Three supervised learning methods, including logistic regression, neural network, and linear regression will be used to evaluate the relationships between variables.

## Method

### Data Exploration
From the heatmap, we can see that there is not sepcific feature that has a strong correlation with `Serverity`.

<img width="1516" alt="Screenshot 2022-12-07 at 9 34 15 PM" src="https://user-images.githubusercontent.com/76938794/206364942-055c5bdf-dda4-48cd-8a79-80ec39f869c0.png">


From the two plots below, we see that Miami is the city that has the most number of car accidents and California is the state that has most accidents.

<img width="878" alt="Screenshot 2022-12-07 at 8 53 58 PM" src="https://user-images.githubusercontent.com/76938794/206364774-fae1250d-837d-443d-b550-5a7e88fffc1e.png">
<img width="861" alt="Screenshot 2022-12-07 at 8 54 10 PM" src="https://user-images.githubusercontent.com/76938794/206364777-cb840e7c-77a5-447d-9f6a-c0286f3f964f.png">


Majority of the car accidents collected in the dataset are from 2021 (53.1% of overall).
<img width="867" alt="Screenshot 2022-12-07 at 9 33 42 PM" src="https://user-images.githubusercontent.com/76938794/206364888-68c6ee5c-6bd7-4d3a-8e2d-892201034965.png">


Severity per year
<img width="987" alt="Screenshot 2022-12-07" src = "https://user-images.githubusercontent.com/118643840/206361875-32799656-9e00-418b-b624-04e000af300d.png">

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

Given that we have over 2 millions observations, we removed the missing values of the following labels: `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`, and `Sunrise_Sunset`.

#### Dropping the NaN rows(We have very large dataset)
Given that we have 2 millions observations and only a few thousands of rows have missing value, it is okay to drop them.

#### Feature Engineering -- Wind Direction
In `wind direction column`, we observe that there are some duplicative values, such as SSE and SSW, which they can be reclassified to south so that we can turn them into dummy variables in future.

#### Feature Engineering -- Weather Condition
In order to simplify the `weather_condition` column, we look up some online resources.
According to road weather management program, it tells that snow/sleet, rain and fog are the main weather condition cause car accidents. Therefore, we are only going to focus on these weather conditions.

#### Feature Engineering -- Duration

```
data['End_Time'] = pd.to_datetime(data['End_Time'])
data['Start_Time'] = pd.to_datetime(data['Start_Time'])
data['Duration'] = data.End_Time - data.Start_Time
new_data['Duration'] = new_data['Duration'].dt.total_seconds()
new_data['Duration'] = new_data['Duration'] / 60
```

#### Convert catorical variables using Orinal Encoder
For `Wind_Direction`, we categorize wind direction to main 4 directions: north, west, south and east
For `Weather_Condition`, the description about weather is really in detail. After some researches, we decided to categorize weather condition into 6 main categories: rain, fog, snow, cloud, clear, and thunderstorms. These are the main causes in car accidents.
For columns contain boolean values and strings, such as `state` and `side`, we converted them to dummy value using ordinal encoder.
```
ord_enc = OrdinalEncoder()

new_data.iloc[:,11:27] = ord_enc.fit_transform(new_data.iloc[:,11:27]).astype(int)
new_data.iloc[:,29:] = ord_enc.fit_transform(new_data.iloc[:,29:]).astype(int)
```

#### Dependent variable change
By observing the severity per year graph, we find that there is no `severity = 1` in any of the years except year 2020. Therefore, we decide to combine it with `severity = 2` and classify it as a level 1 traffic accident, which we also call a minor traffic accident. In addition, we also rename `severity = 3` and `severity = 4` to 2 and 3 respectively.

```
new_data.loc[new_data['Severity'] == 2, 'Severity'] = 1
new_data.loc[new_data['Severity'] == 3, 'Severity'] = 2
new_data.loc[new_data['Severity'] == 4, 'Severity'] = 3
```


#### Test and Split
Instead of spliting the data base on certain percentage, we decided to split data base on the year the accidents occur.
We use accidents happened before 2020 as training data, and accidents happened in 2020 as testing data. 
```
training_data = new_data[new_data["year"] < 2020].sample(frac = 0.5, random_state = 42)
test_data = new_data[new_data["year"] == 2020].sample(frac = 0.5, random_state = 42)
training_data = training_data.drop(['year'], axis = 1)
test_data = test_data.drop(['year'], axis = 1)
```

### Logistic Regression Model
Since we try to find out what is the main factor of car accidents, we do a logistic regression model and make a graph on the weight of each feature.
```
model = LogisticRegression()
model.fit(X_train, y_train)
importances = pd.DataFrame(data={'Feature': X_train.columns,'Weight': np.abs(model.coef_[0])})
importances = importances.sort_values(by='Weight', ascending=False)
```

### Neural network Model
For the first three layers, we used 32, 16, and 8 units respectively; and activation function 'relu'.
For the output layers, we used 5 units and 'softmax' as the activation.
```
nn_model = Sequential()
nn_model.add(Dense(units = 32, activation = 'relu', input_dim = 32))
nn_model.add(Dense(units = 16, activation = 'relu'))
nn_model.add(Dense(units = 8, activation = 'relu'))
nn_model.add(Dense(units = 4, activation = 'softmax'))
nn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
```

## Evaluation

<img width="858" alt="Screenshot 2022-12-07 at 9 35 55 PM" src="https://user-images.githubusercontent.com/76938794/206365499-6fc08760-4604-4b73-97c5-13013744f1bf.png">
<img width="579" alt="Screenshot 2022-12-07 at 9 36 01 PM" src="https://user-images.githubusercontent.com/76938794/206365507-957e3b90-118d-4aca-9b14-ca9e05b7e18b.png">


Our second model is a neural network model, we were able to achieve accuracy of 86%.

<img width="496" alt="Screenshot 2022-12-05 at 9 37 16 AM" src="https://user-images.githubusercontent.com/118643840/206365929-8d505cf6-e33d-4cab-a16d-49fbcb030aa3.png">
<img width="495" alt="Screenshot 2022-12-05 at 9 37 24 AM" src="https://user-images.githubusercontent.com/118643840/206366099-192dfe0f-dd8e-4ce2-a168-beb1f54051cf.png">
<img width="567" alt="Screenshot 2022-12-05 at 9 37 32 AM" src="https://user-images.githubusercontent.com/118643840/206366205-816d2d52-4210-4ba1-a856-ed3a86a0c46c.png">


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

