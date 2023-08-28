from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    #first case when there is x and y
    if y is not None:
        X = X.dropna().drop_duplicates()
        X = X.drop(columns=["id", 'sqft_living15', 'sqft_lot15', 'long', 'date', 'lat'])
        df["zipcode"] = df["zipcode"].astype(float)
        #find the bad index and drops it
        X = X[(X["bedrooms"] >= 0) & (X["bathrooms"] >= 0) & (X["floors"] >= 0) &
              (X["sqft_living"] > 0) & (X["sqft_lot"] > 0) & (X["sqft_above"] > 0)
              &(X["yr_built"] > 0)& (X["sqft_basement"] > 0) & (X["yr_renovated"] >= 0)
              & (X["bedrooms"] < 20) & (X["sqft_lot"] < 1250000)]
        X = pd.get_dummies(X, prefix='zipcode_', columns=['zipcode'])
        y = y.loc[y.index.isin(X.index)]
        return X, y
    else:
        #second case when y is none
        X = X.drop(columns=["id", 'sqft_living15', 'sqft_lot15', 'long', 'date', 'lat'])
        X["zipcode"] = X["zipcode"].astype(float)
        # Find the bad index
        bad_index = X.index[(X["bedrooms"] < 0) | (X["bathrooms"] < 0) | (X["floors"] < 0) |
                            (X["yr_renovated"] < 0) | (X["yr_built"] <= 0)|(X["sqft_living"] <= 0) |
                            (X["sqft_lot"] <= 0) | (X["sqft_above"] <= 0) |(X["sqft_basement"] <= 0)
                            | (X["bedrooms"] >= 20) | (X["sqft_lot"] >= 1250000)]
        # Replace the bad values with the average value
        X = pd.get_dummies(X, prefix='zipcode_', columns=['zipcode'])
        for col in X.select_dtypes(include=np.number).columns:
            X.loc[bad_index, col] = X[col].mean()
        X.fillna(X.select_dtypes(include=np.number).mean(), inplace=True)
        return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    #  here i excluded  the columns that contain the str 'zipcode_'
    cols_to_drop = X.columns[X.columns.str.contains('^zipcode_', case=False)].tolist()
    X = X.drop(cols_to_drop, axis=1)
    for column in X.columns:
        covariance = X[column].cov(y)
        standard_deviation = (np.std(X[column]) * np.std(y))
        pearson_correlation = covariance / standard_deviation
        layout = {
            "title": f"Correlation between {column} and variable {pearson_correlation}",
            "xaxis": {"title": f"{column} values"},
            "yaxis": {"title": "Response variable"}
        }
        fig1 = go.Figure([go.Scatter(x=X[column], y=y, mode='markers')], layout=layout)
        fig1.show()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    # Question 1 - split data into train and test sets
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, y_train, X_test, y_test = split_train_test(X, y, 0.75)
    # Question 2 - Preprocessing of housing prices dataset
    data_frame, prices = preprocess_data(X_train, y_train)

    dt = preprocess_data(X_test)
    # This code is used to reindex two DataFrames in order to have the same order
    dt = dt.reindex(columns=data_frame.columns, fill_value=0)
    data_frame = data_frame.reindex(columns=dt.columns, fill_value=0)
    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(data_frame, prices)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    """
    this code  is a linear regression with varying sample
    sizes and calculates the mean loss and standard deviation of the loss over
    ten trials for each sample size, there is  while loop that will 
    iterate over different sample size and also enters another while loop that
    will run ten trials for the current sample size
    when the ten trials are finished, we calculates the mean and standard deviation of the loss values
    and  It then calculates the lower and upper bounds of the loss distribution
    """
    _range = list()
    _mean = list()
    p = 0.1
    lower = list()  # mean-2*std
    upper = list()  # mean+2*std
    while p < 1.0:
        _range.append(p)
        loss = []
        idx = 0
        while idx < 10:
            temp_x = data_frame.sample(frac=p)
            temp_y = prices.loc[temp_x.index]
            lin_r = LinearRegression()
            lin_r.fit(temp_x.to_numpy(), temp_y)
            loss.append(lin_r.loss(dt.to_numpy(), y_test.to_numpy()))
            idx += 1
        plot_1 = np.mean(loss) - 2 * np.std(loss)
        plot_2 = np.mean(loss) + 2 * np.std(loss)
        _mean.append(np.mean(loss))
        lower.append(plot_1)
        upper.append(plot_2)
        p += 0.01
    _range = np.array(_range) * 100
    layout = {
        "title": "Mean Loss Prediction for Different Train/Test Sample Sizes",
        "xaxis": {"title": "Sample Size Percentage"},
        "yaxis": {"title": "Mean Loss"}
    }

    fig2 = go.Figure([
        go.Scatter(x=_range, y=_mean, mode="markers+lines", name="Mean Loss",
                   marker=dict(color="blue")),
        go.Scatter(x=_range, y=lower, fill=None, mode="lines",
                   line=dict(color="grey"), showlegend=False),
        go.Scatter(x=_range, y=upper, fill='tonexty', mode="lines",
                   line=dict(color="grey"), showlegend=False)
    ], layout=layout)

    fig2.update_traces(marker=dict(size=3))  # increase the size of the markers
    fig2.update_layout(plot_bgcolor="white")  # change the plot background color
    fig2.show()
