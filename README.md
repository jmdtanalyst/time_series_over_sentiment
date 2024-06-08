## Project Description: Time Series Over Sentiment Analysis

### Overview
This project involves the extraction, processing, and analysis of a large dataset of tweets to perform sentiment analysis and time series forecasting. The primary goal is to understand the sentiment trends over time and predict future sentiments using advanced machine learning techniques.

The dinamic dashboard can be accessed at:  [http://oci.jmcloudpro.com:8050/](http://oci.jmcloudpro.com:8050/) 
### Dataset
The dataset consists of 1,600,000 tweets extracted using the Twitter API. It includes the following fields:
- **ids**: The ID of the tweet.
- **date**: The date and time when the tweet was posted.
- **flag**: The query used (if any); otherwise, it is marked as NO_QUERY.
- **user**: The username of the person who posted the tweet.
- **text**: The content of the tweet.

### Data Processing
1. **Library Installation**: Essential libraries such as `wordcloud`, `vaderSentiment`, `dash`, `tensorflow`, `keras`, and `nltk` were installed.
2. **Data Loading**: The data was loaded into a local MongoDB database using PySpark.
3. **Data Cleaning**: The text data was cleaned by converting to lowercase, removing user mentions, links, and punctuation.
4. **Conversion to Pandas**: The cleaned dataset was converted from PySpark to a Pandas DataFrame for easier manipulation and analysis.

### Exploratory Data Analysis (EDA)
- **User Activity**: Analysis of the most active users and the distribution of tweets.
- **Word Cloud**: Visualization of the most frequent words in the tweets.
- **Word Count**: Analysis of the number of words per tweet.

### Sentiment Analysis
- **VADER Sentiment Analysis**: Applied VADER (Valence Aware Dictionary and Sentiment Reasoner) to classify tweets into positive, neutral, and negative sentiments.
- **Sentiment Distribution**: Visualization of the distribution of sentiments using pie charts and bar charts.

### Time Series Analysis
- **Data Grouping**: Grouped sentiments by day to create a time series dataset.
- **Handling Missing Dates**: Filled gaps in the time series data using linear interpolation.
- **Autocorrelation**: Analyzed the autocorrelation of sentiments to understand the relationship between successive time intervals.
- **Decomposition**: Decomposed the time series data into trend, seasonality, and residual components.
- **Stationarity Test**: Conducted the Augmented Dickey-Fuller test to check for stationarity.

### Forecasting Models
1. **SARIMAX**: Applied Seasonal ARIMA with eXogenous factors (SARIMAX) for time series forecasting. Hyperparameters were optimized using Randomized Search Cross-Validation.
2. **LSTM**: Built a Long Short-Term Memory (LSTM) neural network model for forecasting future sentiments.
3. **GRU**: Implemented a Gated Recurrent Unit (GRU) model for time series prediction.

### Dynamic Dashboard
A dynamic dashboard was created using Plotly Dash to visualize the results interactively. The dashboard includes:
- **Sentiment Analysis**: Visualizations of sentiment distribution and word counts.
- **Time Series Forecasting**: Interactive sliders to forecast sentiments for different time periods.

### Conclusion
This project demonstrates the application of various data processing, sentiment analysis, and time series forecasting techniques on a large dataset of tweets. The results provide insights into sentiment trends over time and enable predictions of future sentiments using advanced machine learning models.

### Tools and Technologies
- **Programming Languages**: Python
- **Libraries**: PySpark, Pandas, Numpy, Matplotlib, Plotly, Dash, TensorFlow, Keras, NLTK, Statsmodels
- **Database**: MongoDB
- **Machine Learning Models**: SARIMAX, LSTM, GRU

### Author
Jose Mario da Cruz Costa

For more details, visit [Jose Mario Costa's GitHub](https://github.com/jmdtanalyst) or [Jose Mario Costa's Website](http://www.jmcloudpro.com).
