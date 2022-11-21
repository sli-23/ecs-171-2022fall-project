# ECS 171 2022 Fall Final Project

> **Note**
> 
> **Group 30**: [Google Colab Notebook](https://colab.research.google.com/drive/1AqbJA3umP6qhSGoXuhuWKKh4dS5_917D?usp=sharing)

## Introduction

<img src="https://thumbs.gfycat.com/GlumWastefulAdouri-max-1mb.gif">

The dataset is about car accident dataset in the USA from 2016 to 2021, which was collected using multiple Traffic APIs ([Reference](https://smoosavi.org/datasets/us_accidents)). More introductions and discriptions of the dataset are included in our project Jupyter Notebook.

## Data Preprocessing

In the `Wind_Speed(mph)` column of the data, we decided to impute data. We used mean value of the `Win_Speed(mph)` to replace the NaN value in this column of data.

Regarding to `Humidity(%)` column of the data, we also use mean to replace nan values.

For `Visibility(mi)` column of the data, we use mean to replace nan values.

For `Temperature(F)` column of the data, we use mean to replace nan values.

For `Pressure(in)` column of the data, we use mean to replace nan values.

We do these because these columns of data contains at least 2 to 5 percents of missing values, which is a big part of a data, so we need to use mean value to replace the nan instead of drop them.
