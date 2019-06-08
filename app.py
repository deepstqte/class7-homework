# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go

# Loading data using sklearn dataset
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.config['suppress_callback_exceptions']=True

boston = load_boston()

columns_names = boston.feature_names
y = boston.target
X = boston.data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

lm = LinearRegression()
lm.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lm.predict(X_test)

table_1 = []
for i in range(len(lm.coef_)):
    table_1.append({'Feature': boston.feature_names[i], "Coeff": lm.coef_[i]})

table_2 = []
for (real, predicted) in list(zip(y_test, predicted_values)):
    table_2.append({'Value': real, 'Predicted': predicted, 'Difference': (real - predicted)})

residuals = y_test - predicted_values

children = [
    html.Div(
        children=[
            html.H1(children='Boston Housing Dataset Regression', style={'textAlign': 'center'}),
            html.Div(children=[html.P(
                                    children='Coefficients table:'
                                ),
                                dash_table.DataTable(
                                    id='table_coeff',
                                    sorting=True,
                                    columns=[{"name": i, "id": i} for i in ['Feature', 'Coeff']],
                                    data=table_1,
                                ),
                                html.P(
                                    children='Performance table:'
                                ),
                                dash_table.DataTable(
                                    id='table_2',
                                    sorting=True,
                                    columns=[{"name": i, "id": i} for i in ['Value', 'Predicted', 'Difference']],
                                    data=table_2,
                                )
                            ], className='five columns'
                    ),
            html.Div(children=[dcc.Graph(
                                    id='figure1',
                                    figure={
                                        'data': [
                                            go.Scatter(
                                                x=y_test,
                                                y=predicted_values,
                                                mode='markers',
                                            ),
                                            go.Scatter(
                                                x = [0, 50],
                                                y = [0, 50],
                                                mode = 'lines'
                                            )
                                        ],
                                        'layout': go.Layout(title="Figure 1", xaxis={"title": "Real Value"}, yaxis={"title": "Predicted Value"})
                                    },
                                ),
                                dcc.Graph(
                                    id='figure2',
                                    figure={
                                        'data': [
                                            go.Scatter(
                                                x=y_test,
                                                y=residuals,
                                                mode='markers',
                                            ),
                                            go.Scatter(
                                                x = [50, 0],
                                                y = [0, 0],
                                                mode = 'lines'
                                            )
                                        ],
                                        'layout': go.Layout(title="Figure 2", xaxis={"title": "Real Value"}, yaxis={"title": "Residuals (Difference)"})
                                    },
                                ),
                                dcc.Graph(
                                    id='figure3',
                                    figure={
                                        'data': [
                                            go.Histogram(x=residuals, nbinsx = 20),
                                            go.Scatter(
                                                x = [0, 0],
                                                y = [50, 0],
                                                mode = 'lines'
                                            )
                                        ],
                                        'layout': go.Layout(title="Residual (difference) Distribution")
                                    },
                                )
                            ], className='five columns'
                    )
        ]
    )
]

app.layout = html.Div(children=[
        html.Div(children=children),
    ])

if __name__ == '__main__':
    app.run_server(debug=False)
