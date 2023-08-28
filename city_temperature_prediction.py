import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data_frame = pd.read_csv(filename, parse_dates=["Date"])
    data_frame = data_frame.dropna().drop_duplicates()
    data_frame = data_frame[(data_frame["Temp"] >= 1)]
    data_frame["DayOfYear"] = data_frame["Date"].dt.dayofyear
    return data_frame.drop(["Date", "Day"], axis=1)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    city_temp_dataset = load_data("../datasets/City_Temperature.csv")
    data = city_temp_dataset
    # Question 2 - Exploring data for specific country
    israel_df = city_temp_dataset[city_temp_dataset['Country'] == 'Israel'].copy()
    israel_df['Year'] = israel_df['Year'].apply(str)
    fig = px.scatter(israel_df, x='DayOfYear', y='Temp', color='Year',
                     title="Change in Average Temperature(Israel)",
                     color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.write_image("daily_temp.png")
    month_df = israel_df.groupby('Month', as_index=False).agg(temp_std=('Temp', 'std'))
    fig = px.bar(month_df, x='Month', y='temp_std', title="Monthly Temperature Variation(Israel)",
                 color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.write_image("monthly_temp.png")
    # Question 3 - Exploring differences between countries
    '''
    for each country we analyze the temp and we create line that shows the average 
    temp and the standard deviation for each month of year 
    i used color argument in order to assign different color to each country
    groupby method is used to group the data by country and month, and then the agg method
    is used to calculate the mean and standard deviation.
    '''
    city_temp_dataset = (city_temp_dataset.groupby(["Country", "Month"])['Temp']
                         .agg([np.std, np.mean]).reset_index()
                         .rename(columns={"std": "Temp_std", "mean": "Temp_mean"}))
    fig = px.line(city_temp_dataset, x="Month", y="Temp_mean", error_y="Temp_std", color="Country",
        title="The average and standard deviation of the temperature",
        color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.update_layout(xaxis_title="Month",yaxis_title="Temperature (°C)")
    fig.write_image("differences_between_countries2.png")
    # Question 4 - Fitting model for different values of `k`
    arr = list()
    day = israel_df["DayOfYear"]
    temp = israel_df["Temp"]
    train_x, train_y, test_x, test_y = split_train_test(day, temp, 0.75)
    idx = 1
    while idx <= 10:
        poly_fit = PolynomialFitting(k=idx)
        train_x_arr = train_x.to_numpy()
        train_y_arr = train_y.to_numpy()
        test_x_arr = test_x.to_numpy()
        test_y_arr = test_y.to_numpy()
        poly_fit.fit(train_x_arr, train_y_arr)
        error = poly_fit.loss(test_x_arr, test_y_arr)
        arr.append(error)
        print("degree=%d,error=%.2f" % (idx, error))
        idx += 1
    degree_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data_dict = {"degree": degree_list, "error": arr}
    df = pd.DataFrame(data_dict)
    fig = px.bar(df, x="degree", y="error", title="Calculating test error over polynomial regression")
    fig.update_layout(title_font_color="purple")
    fig.update_xaxes(title_text="Polynomial Degree", color="blue")
    fig.update_yaxes(title_text="Test Error", color="green")
    fig.update_traces(marker_color="orange")
    fig.write_image("difference_is.png")
    # Question 5 - Evaluating fitted model on different countries
    """
    here we  analyzes temperature data for different countries.
    i called PolynomialFitting in order to use the fit a polynomial curve
    to temp data for Israel. The fitted curve is used to calculate the diff between the 
    predicted and actual temperature values = loss.
    and i calculated the loss for temperature data from other countries
    and stores the results in a dictionary. Finally, the results are visualized using a bar chart that shows
    the loss for each country.
    """
    p_f = PolynomialFitting(5)  # in q4 we took k =5
    df_sub = data[data["Country"] == "Israel"]
    day_arr = df_sub["DayOfYear"].to_numpy()
    temp_arr = df_sub["Temp"].to_numpy()
    arr_loss = list()
    p_f.fit(day, temp)
    countries_list = list(data[data["Country"] != "Israel"]["Country"].drop_duplicates())
    index = 0
    while index < len(countries_list):
        c = countries_list[index]
        sub = data[data["Country"] == c]
        day_sub = sub["DayOfYear"]
        temp_sub = sub["Temp"]
        loss_sub = p_f.loss(day_sub, temp_sub)
        arr_loss.append(loss_sub)
        index += 1
    c_dict = {"country": countries_list, "loss": arr_loss}
    df = pd.DataFrame(c_dict)
    fig = px.bar(df, x="country", y="loss", title="The model’s error over each of the other countries",
                 color=countries_list)
    fig.write_image("others.png")
