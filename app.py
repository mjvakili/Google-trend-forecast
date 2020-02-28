from vega_datasets import data
import streamlit as st
import altair as alt
import numpy as np
#import fbprophet
#import pytrends
#from fbprophet import Prophet
#import numpy as np
#from pytrends.request import TrendReq
#from pytrends.exceptions import ResponseError
from forecast import Trend_Forecast



def main():

    #df = load_data()
    page = st.sidebar.selectbox("Choose a page", ["Exploration", "Forecast"])
    
    group = st.sidebar.selectbox("Choose a platform on Google", ["images", "youtube", "news"])
    keyword = st.sidebar.selectbox("Choose a search keyword", ["Pytorch", "Tensorflow", "Blockchain"])
    start_year = st.sidebar.slider("Start Year", min_value=2016, max_value=2020, value=2018, step=1)
    start_month = st.sidebar.slider("Start Month", min_value=1, max_value=12, value =1,  step=1)
    end_year = st.sidebar.slider("End Year", min_value=2016, max_value=2020, value=2018, step=1)
    end_month = st.sidebar.slider("End Month", min_value=1, max_value=12, value =1,  step=1)
    num_days = st.sidebar.slider("Number of days from now", min_value=10, max_value=365, value =100,  step=5)

    trend_forecast = Trend_Forecast(keyword = keyword, group = group, loc = "", 
               year_start = start_year, month_start = start_month, day_start = 1, 
               year_end = end_year, month_end = end_month, day_end = 1, num_future_days = num_days)
     
    if page == "Exploration":
        st.header("This is your data explorer.")
        st.write("In order to see the future forecasts, select the forecast page on the left.")
        #st.write(df)
        df = trend_forecast.fetch_from_google()
        st.write(df)
        df["Date"] = df.index
        visualize_data(df, "Date", keyword)

    elif page == "Forecast":
        st.title("Forecasting the future trends in Google Searches")
        #x_axis = st.selectbox("Choose a variable for the x-axis", df.columns, index=3)
        #y_axis = st.selectbox("Choose a variable for the y-axis", df.columns, index=4)
        #visualize_data(df, x_axis, y_axis)
        fig = trend_forecast.plot("prophet")
        st.pyplot(fig, clear_figure = False)

@st.cache
def load_data():
    df = data.cars()
    return df

def visualize_data(df, x_axis, y_axis):
    
    #graph = alt.Chart(df).mark_circle(size=60).encode(
    #    x=x_axis,
    #    y=y_axis,
    #    color='Origin',
    #    #tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    #).interactive()

    graph = alt.Chart(df).mark_line().encode(
        x= x_axis,
        y= y_axis).interactive()

    st.write(graph)

if __name__ == "__main__":
    main()

