import plotly.graph_objects as go # or plotly.express as px

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import pickle
df=pd.read_csv('esp_data.csv').drop(['Unnamed: 0'],1)
df['LOW Supply Voltage']=df['Low Supply Voltage']|df['Low Supply Voltage-Lockout'] # OR
df=df.drop(['Low Supply Voltage','Low Supply Voltage-Lockout'],1)
X=df.iloc[:,:-10]
y=df.iloc[:,-10:]
newy=y.shift(-60)
df=pd.concat([X,newy],1)
df= df.dropna()
X=df.iloc[:,:-10]
y=df.iloc[:,-10:]
cols=y.columns
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=63)
fig = go.Figure() 
import dash
import dash_core_components as dcc
import dash_html_components as html

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense


model = Sequential()
model.add(Dense(64, activation='relu', input_dim = X_train.shape[1]))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history= model.fit(X_train, y_train,validation_data=(X_test, y_test), batch_size = 32, epochs = 40, verbose = 2) # Verbose The Way of Visualization
p= model.predict(X_test)
table_data=pd.DataFrame(p*100,columns=cols).mode()
Features= X.iloc[:,:7]

################### DASH
app = dash.Dash(__name__)
server = app.server
app.title = 'ESP Dashboard'
fig1 = go.Figure(data=[go.Table(
    header=dict(values=list(table_data.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=table_data.T,
               fill_color='lavender',
               align='left'))
])


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=Features[Features.columns[0]],
    name = Features.columns[0]
))


fig.add_trace(go.Scatter(
    x=df.index,
    y=Features[Features.columns[1]],
    name=Features.columns[1],
    yaxis="y2"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=Features[Features.columns[2]],
    name=Features.columns[2],
    yaxis="y3"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=Features[Features.columns[3]],
    name=Features.columns[3],
    yaxis="y4"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=Features[Features.columns[4]],
    name=Features.columns[4],
    yaxis="y5"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=Features[Features.columns[5]],
    name=Features.columns[5],
    yaxis="y6"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=Features[Features.columns[6]],
    name=Features.columns[6],
    yaxis="y7"
))


# Create axis objects
fig.update_layout(
    xaxis=dict(
        domain=[0.3, 0.7]
    ),
    yaxis=dict(
        title="yaxis title",
        titlefont=dict(
            color="#1f77b4"
        ),
        tickfont=dict(
            color="#1f77b4"
        )
    ),
    yaxis2=dict(
        title="yaxis2 title",
        titlefont=dict(
            color="#ff7f0e"
        ),
        tickfont=dict(
            color="#ff7f0e"
        ),
        anchor="free",
        overlaying="y",
        side="left",
        position=0.15
    ),
    yaxis3=dict(
        title="yaxis3 title",
        titlefont=dict(
            color="#d62728"
        ),
        tickfont=dict(
            color="#d62728"
        ),
        anchor="x",
        overlaying="y",
        side="right"
    ),
    yaxis4=dict(
        title="yaxis4 title",
        titlefont=dict(
            color="#9467bd"
        ),
        tickfont=dict(
            color="#9467bd"
        ),
        anchor="free",
        overlaying="y",
        side="right",
        position=0.85
    ),
    yaxis5=dict(
        title="yaxis5 title",
        titlefont=dict(
            color="#9437bd"
        ),
        tickfont=dict(
            color="#9437bd"
        ),
        anchor="free",
        overlaying="y",
        side="right",
        position=0.85
    ),
    yaxis6=dict(
        title="yaxis6 title",
        titlefont=dict(
            color="#9237bd"
        ),
        tickfont=dict(
            color="#9237bd"
        ),
        anchor="free",
        overlaying="y",
        side="right",
        position=0.85
    ),
    yaxis7=dict(
        title="yaxis7 title",
        titlefont=dict(
            color="#9137bd"
        ),
        tickfont=dict(
            color="#9137bd"
        ),
        anchor="free",
        overlaying="y",
        side="right",
        position=0.85
    )
)



# Update layout properties
fig.update_layout(

)
app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
    ,dcc.Graph(figure=fig1)
])


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)