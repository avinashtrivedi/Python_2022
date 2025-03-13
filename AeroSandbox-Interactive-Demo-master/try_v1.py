import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sub
import dash
from dash import html,dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import aerosandbox as asb
import casadi as cas
from airplane import make_airplane
import numpy as np
import pandas as pd
from skimage.util import invert
import hyperspy.api as hs
from skimage import measure
from particlespy.segptcls import process
from skimage.segmentation import flood, flood_fill, mark_boundaries
from particlespy.particle_analysis import parameters


# def threshold_choice(th):
#     if th == "Otsu":
#          return "otsu"
#     elif th == "Mean":
#         return "mean"
#     elif th == "Minimum":
#         return "minimum"
#     elif th == "Yen":
#         return "yen"
#     elif th == "Isodata":
#         return "isodata"
#     elif th == "Li":
#         return "li"
#     elif th == "Local":
#         return "local"
#     elif th == "Local Otsu":
#         return "local_otsu"
#     elif th == "Local+Global Otsu":
#         return "lg_otsu"
#     elif th == "Niblack":
#         return "niblack"
#     elif th == "Sauvola":
#         return "sauvola"

app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                html.H2("PARTICLE SPY"),
                html.H5("AUTO TAB"),
            ], width=True),
        ], align="end"),
        
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Key Parameters"),
                    html.P("Rolling Ball Size:"),
                    dcc.Input(id='rolling', value=43, type="number"),
                    html.P("Gaussian filter kernal Size:"),
                    dcc.Input(id='gaussian', value=7.0, type="number"),
                    html.P("Thresholding Options"),
                    dcc.Dropdown(
                    id="Threshold",
                    options=[
                        {
                            "label": "Li",
                            "value": "li",
                        },
                        {
                            "label": "Otsu",
                            "value": "otsu",
                        },
                        {
                            "label": "Mean",
                            "value": "mean",
                        },
                        {
                            "label": "Maximum",
                            "value": "maximum",
                        },
                        {
                            "label": "Yen",
                            "value": "yen",
                        },
                        {
                            "label": "Isodata",
                            "value": "isodata",
                        },
                    ],
                    value="otsu",
                ),
                    
#                 html.P("Checklist"),
                 dcc.Checklist(
                            options=[
                                {"label": "Watershed", "value": "Watershed"},
                            
                            ],
                            value=[],
                            id="checklist-Watershed-options",
                            className="checklist-Watershed",
                ),
                    
                html.P("Watershed Seed Separation"),
                dcc.Input(id='wing_span2', value=43, type="number"),
                html.P("Watershed Seed Erosion"),
                dcc.Input(id='alpha1', value=7.0, type="number"),   
#                 html.P("Checklist"),
                 dcc.Checklist(
                            options=[
                                {"label": "Invert", "value": "Invert"},

                            ],
                            value=[],
                            id="checklist-Invert-options",
#                             className="checklist-Invert",
                ),

                html.P("Min Particle Size(px)"),
                dcc.Input(id='MinParticle', value=10, type="number"),
                    
                 html.P("Display"),
                    dcc.Dropdown(
                    id="image_label",
                    options=[
                        {
                            "label": "Labels",
                            "value": "Labels",
                        },
                        {
                            "label": "Image",
                            "value": "Image",
                        }
                    ],
                    value="Image",
                ),   
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                ]),

                html.Hr(),
                html.Div([
                    html.H5("Commands"),
                    dbc.Button("Display", id="display_geometry", color="primary", style={"margin": "5px"},
                               n_clicks_timestamp='0'),
                    dbc.Button("Get Params", id="run_ll_analysis", color="secondary", style={"margin": "5px"},
                               n_clicks_timestamp='0'),
                ]),
                
            ], width=3),
            dbc.Col([
                # html.Div(id='display')
                dbc.Spinner(
                    dcc.Graph(id='display', style={'height': '80vh'}),
                    color="primary"
                )
            ], width=True)
        ]),
        html.Hr(),
    ],
    fluid=True
)


@app.callback(
    Output('display', 'figure'),
    Input('display_geometry', 'n_clicks_timestamp'),
    State('rolling', 'value'),
    State('gaussian', 'value'), 
    State('Threshold', 'value'),
    State('checklist-Invert-options', 'value'), 
    State('MinParticle', 'value'),
    State('image_label', 'value'))

def display_geometry(
        xx,
        roll,
        gauss,
        th,
        checklist,
        min_particle_size,
        img_label
):
    print('check*********',xx,roll,gauss,checklist,min_particle_size,img_label,sep='\n')
    data = hs.load('autoSTEM_9.dm4')# autoSTEM_9
    image = data.data
    
    params = parameters()
    params.generate()
    
#     if len(checklist)==1:
#         params.segment['invert'] = True
        
    
    if xx == 0:
        fig = px.imshow(image)
    else:
        
        if roll==1:
            params.segment['rb_kernel'] = 0
        else:
            params.segment['rb_kernel'] = roll

        params.segment['gaussian'] = gauss
        params.segment['threshold'] = th
        params.segment['min_size'] = min_particle_size

        labels = process(data,params)
        labels = np.uint8(labels*(256/labels.max()))
        if img_label == 'Image':
            print('INSIDE IMAGE')
            b = np.uint8(mark_boundaries(image, labels, color=(1,1,1))[:,:,0]*255)
            if len(checklist)==1:
                figure = px.imshow(invert(b).data)
            else:
                figure = px.imshow(b.data) 

        elif img_label == 'Labels':
            print('INSIDE Labels',params.segment.items())

            figure = px.imshow(labels.data)
     
    
#     if len(checklist)==1:
#         figure = px.imshow(invert(image))
#     else:
#         figure = px.imshow(image)
    

    figure.update_layout(
        autosize=True,
        yaxis={'visible': False, 'showticklabels': False},
        xaxis={'visible': False, 'showticklabels': False},
        margin=dict(l=0,r=0,b=0,t=0))
    
    figure.update_traces(hoverinfo='skip')
    figure.update(layout_coloraxis_showscale=False)

    return figure

app.run_server(port=8059,debug=True)