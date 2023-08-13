# Linear Regression Assignment
## An Analysis of Bike Sharing Data

## Table of Content
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)

## General Information
### Business Understanding
- A US-based bike sharing provider **Boom Bikes**, provides bikes on sharing basis.
- They have seen a dip in revenues due to the ongoing Corona Pandemic.
- They want to understand the demands of the market when the quarantine situation ends allowing them an edge over the competitors
**Business Goal**:
- model the demand with the available independent variables
- the client wants to understand how exactly the demands vary with different features
- to manipulate the business strategy to meet the demands
- understand the new business dynamics
### We intend to 
- understand the factors on which the demands for these shared bikes depends in the American Market
- know:
    - Which variables are significant in predicting the demand for shared bikes
    - How well those varibles describe the bikes demand

## Technologies Used

Please refer to the `requirements.txt` file to see all the libraries used in the project.

## Conclusions

The following are the columns to be used. 

| variable | $\beta_i$ value|
|--------------|:-----:|
|const|0.1962|
|yr|0.2161|
|holiday|-0.0603|
|workingday|0.0066|
|atemp|0.6006|
|hum|-0.2122|
|windspeed|-0.1281|
|season_2|0.0822|
|season_4|0.1287|
|mnth_9|0.1102|
|weathersit_3|-0.1475|
We can infer that:
- On windy days, the number of customers will be less
- Same can be said about a holiday
- On high humidity days the number of customers will be even less.
  with increare in temperature, the number of customers will increase

## Contact
Created by [eeshangpt](https://github.com/eeshangpt) - feel free to contact me!
