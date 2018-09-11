# Data-analytics-pipeline-using-Apache-Spark

OVERVIEW:

The hands-on practical learning components of the course comprises two types of activities: labs covering one or two knowledge units (skills, competencies) of data-intensive computing and a single term project serving as a capstone covering the entire data pipeline. In the first half of the course we learned data analytics and quantitative reasoning using R language. In the second half we focused on big data approaches for data analytics with Hadoop MapReduce and Apache Spark. In this lab, we will work on understanding concepts related to data analysis using Apache Spark [1].
In Lab1 we learned to analyze data using R language and R Studio. In lab2, we explored approaches that deal with big data, especially text data, using the Google’s MapReduce algorithm. In Lab 3 we will explore processing graph data using Spark [1]. Here is a chance to show case your knowledge in Apache Spark data pipelines and big-data analytics. You can apply the model you build here, to numerous applications. Apache Spark is fantastically suited for this application we will describe here. And the application has uses in many vertical domains.

LEARNING OUTCOMES:

In this lab work you will be able to
1. Explore the Apache Spark framework and programming: spark context (sc), dataflow operations in transformations, actions, pipelines and MLib.
2. Apply your data analytics knowledge (word frequency, word-co-occurrence) and machine learning skills to perform multi-class classification of text data using Apache Spark.
3. Build a data pipeline using
a. Data from sources such as NY Times articles using the APIs provided by the data sources.
b. Split the data into training and test data set.
c. Extract features that will determine the class or category of the article {politics, sports,
business, one of your choice}
d. Build a model for classification using any two of the several classification algorithms.
e. Assess the accuracy using a query text or new article for each news “category”
f. Compare the classification accuracy of at least two well-known classification algorithms,
for a given text data set.
4. Document the design and implementation of the project using either markup or markdown [ref]
language.
5. Apply the knowledge and skills learned to solve classification problems in other domains.
  
LAB DESCRIPTION:

Introduction: In this age of analytics, data science process plays a critical role for many organizations. Several organizations including the federal government (data.gov) have their data available to the public for various purposes. Social network applications such as Twitter and Facebook collect enormous amount of data contributed by their numerous and prolific user. For other businesses such as Amazon and NYTimes data is a significant and valuable byproduct of their main business. Nowadays everybody has data. Most of these data generator businesses make subset of their data available for use by registered users for free. Some of them as downloadable data files (.csv, .xlsx) as a database (.db, .db3). Sometimes the data that needs to be collected is not in a specific format but is available as a web page content. In this case, typically a web crawler is used to crawl the web (pages) and scrap the data from these web pages and extract the information needed. Data generating organizations have realized the need to share at least a subset of their data with users interested in developing applications. Entire data sets are sold as products. Very often data collected is not in the format required for the downstream processes such as EDA, analysis, prediction and visualization. The data needs to be cleaned, curated and munged before modeling and algorithmic processing. All the data pre-processing will be done in the MR or Spark framework and not outside.
