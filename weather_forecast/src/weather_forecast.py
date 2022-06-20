import numpy as np # For Linear Algebra
import pandas as pd # To Work With Data
# for visualizations
import argparse
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime # Time Series analysis.
from sklearn.cluster import KMeans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_file', type=str, default = "Weather.csv",
                        help='please input a csv file')
    
    
    args = parser.parse_args()
    input_data = args.input_data_file
    df = pd.read_csv(input_data)
    df1 = pd.melt(df, id_vars='YEAR', value_vars=df.columns[1:])
    df1['Date'] = df1['variable'] + ' ' + df1['YEAR'].astype(str)  
    df1.loc[:,'Date'] = df1['Date'].apply(lambda x : datetime.strptime(x, '%b %Y')) ## Converting String to datetime object
    df1.head()
    df1.columns=['Year', 'Month', 'Temprature', 'Date']
    df1.sort_values(by='Date', inplace=True)
    fig = go.Figure(layout = go.Layout(yaxis=dict(range=[0, df1['Temprature'].max()+1])))
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Temprature']), )
    fig.update_layout(title='Temprature Throught Timeline:',
                     xaxis_title='Time', yaxis_title='Temprature in Degrees')
    fig.update_layout(xaxis=go.layout.XAxis(
        rangeselector=dict(
            buttons=list([dict(label="Whole View", step="all"),
                          dict(count=1,label="One Year View",step="year",stepmode="todate")                      
                         ])),
            rangeslider=dict(visible=True),type="date")
    )
    fig.show()
    
    km = KMeans(3)
    km.fit(df1['Temprature'].to_numpy().reshape(-1,1))
    df1.loc[:,'Temp Labels'] = km.labels_
    fig = px.scatter(df1, 'Date', 'Temprature', color='Temp Labels')
    fig.update_layout(title = "Temprature clusters.",
                     xaxis_title="Date", yaxis_title="Temprature")
    fig.show()
    
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split 
    from sklearn.metrics import r2_score 

    df2 = df1[['Year', 'Month', 'Temprature']].copy()
    df2 = pd.get_dummies(df2)
    y = df2[['Temprature']]
    x = df2.drop(columns='Temprature')

    dtr = DecisionTreeRegressor()
    train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3)
    dtr.fit(train_x, train_y)
    pred = dtr.predict(test_x)
    print(r2_score(test_y, pred))
    next_Year = df1[df1['Year']==2017][['Year', 'Month']]
    next_Year.Year.replace(2017,2018, inplace=True)
    next_Year= pd.get_dummies(next_Year)
    temp_2018 = dtr.predict(next_Year)

    temp_2018 = {'Month':df1['Month'].unique(), 'Temprature':temp_2018}
    temp_2018=pd.DataFrame(temp_2018)
    temp_2018['Year'] = 2018
    
    forecasted_temp = pd.concat([df1,temp_2018], sort=False).groupby(by='Year')['Temprature'].mean().reset_index()
    fig = go.Figure(data=[
        go.Scatter(name='Yearly Mean Temprature', x=forecasted_temp['Year'], y=forecasted_temp['Temprature'], mode='lines'),
        go.Scatter(name='Yearly Mean Temprature', x=forecasted_temp ['Year'], y=forecasted_temp['Temprature'], mode='markers')
    ])
    fig.update_layout(title='Forecasted Temprature:',
                     xaxis_title='Time', yaxis_title='Temprature in Degrees')
    fig.show()
    
    temp_2018.to_csv("weather_forecast_output.csv", index=False)
    