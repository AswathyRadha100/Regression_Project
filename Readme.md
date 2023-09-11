--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                                                                                                                      #
# **Home Value Forecast Model **                                               
#                                                                                                                                                                      # 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Description** 

###  This project aims to construct a regression model for predicting property tax assessed values ('taxvaluedollarcnt') of Single Family Properties using property attributes. By analyzing the key drivers of property value, the project seeks to improve Zillow's existing model and provide valuable insights for property valuation in different counties.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Goal** 

###  The main goals of the project are
  - &#9733; Construct an ML Regression model for property tax assessment
  - &#9733; Identify key drivers of property value
  - &#9733; Enhance property valuation accuracy
  - &#9733; Provide insights into property value variations
  - &#9733; Deliver a comprehensive report for Zillow's data science team


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Initial Thoughts**

### In the early stages, we're focusing on understanding the data, exploring property value drivers, and building regression models to improve property tax assessment accuracy. This foundational step sets the path for using data effectively to predict property values for single family properties.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **The Plan** 

- &#9733; Acquire data from SQL:
- &#9733; Prepare data that's acquired:
  -  Feature Selection/Engineering
     - &#9642; Encode categorical columns
     - &#9642; Preprocess numerical columns
- &#9733; Explore data in search of key property value drivers:
  -  Answer important questions
     - &#9642; Why do some properties have higher values than others nearby?
     - &#9642; Why do properties with similar attributes differ in value?
     - &#9642; Relationship between bathrooms, bedrooms, and property value
- &#9733; Model Selection:
  -   Choose regression algorithms 
     - &#9642; Ordinary Least Squares
     - &#9642; Polynomial Regression
     - &#9642; Generalized Linear Model
- &#9733; Data Splitting and Model Training:
  -  Divide the dataset into train and test sets 
     - &#9642; Train chosen models on training dataset             
- &#9733; Model Evaluation:
  -   Check the performance of models on the test dataset
  - Metrics used 
     - &#9642; Mean Absolute Error (MAE)
     - &#9642; Mean Squared Error (MSE)
     - &#9642; R-squared (R2)
     - &#9642; R-squared (RMSE)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Data Dictionary** 



| #   | Column     | Dtype    | Definition                                       |
| --- | ---------- | -------  | ------------------------------------------------ |
| 0   | bedrooms   | int64    | Number of bedrooms in the house                 |
| 1   | bathrooms  | float64  | Number of bathrooms in the house                |
| 2   | area       | int64    | Total square footage of the property            |
| 3   | taxvalue   | int64    | The assessed tax value based on property valuation|
| 4   | yearbuilt  | int64    | Year when the respective property was originally constructed |
| 5   | county     | object   | County where the property is located            |
| 6   | lotsqft    | int64    | Square footage of the property's lot            |
| 7   | la         | int64    | Indicator for properties in Los Angeles County  |
| 8   | orange     | int64    | Indicator for properties in Orange County       |
| 9   | ventura    | int64    | Indicator for properties in Ventura County      |

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Steps to Reproduce** 

## Ordered List:
     1. Clone this repo.
     2. Acquire the data from SQL DB.
     3. Run data preprocessing and feature engineering scripts.
     4. Explore data using provided notebooks.
     5. Train and evaluate regression models using the provided notebook.
     6. Replicate the property tax assessment process using the provided instructions.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Recommendations**

## Actionable recommendations based on project's insights:
- &#9733; Explore additional property attributes for better prediction accuracy
- &#9733; Consider county-specific property valuation models
- &#9733; Continuously update property data for more accurate assessments
- &#9733; Monitor property value fluctuations and assess external factors


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Takeaways and Conclusions** 

In conclusion, this analysis provides valuable insights into predicting property tax assessed values for single family properties. Key drivers such as property area, year built, and county location influence property values significantly. By implementing the recommended actions, Zillow can improve property tax assessment accuracy, offer better insights to customers, and enhance the overall property valuation experience.

It is evident that a data-driven approach to property valuation, considering both property attributes and county-specific variations, will lead to more accurate assessments and ultimately benefit Zillow's mission of empowering property buyers and sellers with data-driven decisions.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


