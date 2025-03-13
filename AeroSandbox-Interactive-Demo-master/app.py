
# %load app.py
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sub
import dash
from dash import html,dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import aerosandbox as asb
import casadi as cas
import hyperspy.api as hs
from particlespy.particle_analysis import parameters

from airplane import make_airplane
import numpy as np
import pandas as pd

app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                html.H2("Solar Aircraft Design with AeroSandbox and Dash"),
                html.H5("Peter Sharpe check"),
            ], width=True),

        ], align="end"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Key Parameters"),
                    html.P("Number of booms:"),
                    dcc.Slider(
                        id='n_booms',
                        min=1,
                        max=3,
                        step=1,
                        value=1,
                        marks={
                            1: "1",
                            2: "2",
                            3: "3",
                        }
                    ),
                    html.P("Wing Span [m]:"),
                    dcc.Input(id='wing_span', value=10, type="number"),
                    html.P("Angle of Attack check[deg]:"),
                    dcc.Input(id='alpha', value=7.0, type="number"),
                ]),
                html.Hr(),
                html.Div([
                    html.H5("Commands"),
                    dbc.Button("Display (1s)", id="display_geometry", color="primary", style={"margin": "5px"},
                               n_clicks_timestamp='0'),

                ]),
                html.Hr(),

            ], width=3),
            dbc.Col([
                # html.Div(id='display')
                dbc.Spinner(
                    dcc.Graph(id='display', style={'height': '80vh'}),
                    color="primary"
                )
            ], width=True)
        ]),

    ],
    fluid=True
)

@app.callback(
    Output('display', 'figure'),
    Input('display_geometry', 'n_clicks_timestamp'),
    State('n_booms', 'value'),
    State('wing_span', 'value'),
    State('alpha', 'value'))

def display_geometry(
        xx,
        n_booms,
        wing_span,
        alpha,
):

    ### Make the airplane
    airplane = make_airplane(
        n_booms=n_booms,
        wing_span=wing_span,
    )

        # Display the geometry
#     figure1 = airplane.draw(show=False, colorbar_title=None)
#     output = "TESTING....."
    
    data = hs.load('autoSTEM_9.dm4')
    image = data.data
    
    params = parameters()
    params.generate()
    
    figure = px.imshow(image)

    figure.update_layout(
        autosize=True,
        # width=1000,
        # height=700,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
        )
    )

    return figure


try:  # wrapping this, since a forum post said it may be deprecated at some point.
    app.title = "Aircraft Design with Dash"
except:
    print("Could not set the page title!")
app.run_server(port=8058,debug=True)
