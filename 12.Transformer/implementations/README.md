# Overview
Implementation of RNN, LSTM, GRU, or Transformer varaint is allowed
This task focuses on binary sentiment classification of movie reviews using the IMDB Movie Reviews Dataset.
The goal is to classify each review as positive or negative based on its textual content.

Please apply any model of your choice (RNNs, LSTM, GRU, Transformer).

## Dataset
The dataset used in this project is the IMDB Dataset of 50K Movie Reviews, available on Kaggle:

### Dataset Link:
[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)

Dataset Description:
- Total samples: 50,000 movie reviews
- Labels: positive and negative
- Task: Binary text classification (sentiment analysis)
- Balanced dataset: Equal number of positive and negative reviews

### Data Loading
The dataset is provided as a CSV file and can be loaded using pandas as shown below:
```
import pandas as pd
data = pd.read_csv("./IMDB Dataset.csv")
```
