from gc import disable
from pickle import TRUE
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sub
import dash
from dash import html,dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
from skimage.util import invert
import hyperspy.api as hs
from skimage import measure
from particlespy.segptcls import process
from skimage.segmentation import flood, flood_fill, mark_boundaries
from particlespy.particle_analysis import parameters

app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                html.H2("PARTICLE SPY"),
                html.H5("AUTO TAB"),
            ], width=True),
        ], align="start"),
        
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Pre Filtering Options"),
                    html.P("Rolling Ball Size:"),
                    dcc.Input(id='rolling', value=0, min=0, type="number"),
                    html.P("Gaussian filter kernal Size:"),
                    dcc.Input(id='gaussian', value=0, min=0, type="number"),
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
                            "label": "Minimum",
                            "value": "minimum",
                        },
                        {
                            "label": "Yen",
                            "value": "yen",
                        },
                        {
                            "label": "Isodata",
                            "value": "isodata",
                        },
                        {
                            "label": "Local",
                            "value": "local",
                        },
                        {
                            "label": "Local Otsu",
                            "value": "local_otsu",
                        },
                        {
                            "label": "Local+Global Otsu",
                            "value": "lg_otsu",
                        },
                        {
                            "label": "Niblack",
                            "value": "niblack",
                        },
                        {
                            "label": "Sauvola",
                            "value": "sauvola",
                        },
                    ],
                    value="otsu",
                ),
                html.P("Local filter kernel"),
                dcc.Input(id='local_kernel', value=1, min=1,step=2, type="number"),
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
                dcc.Input(id='wing_span2', value=0, min=0, type="number", disabled=False),
                html.P("Watershed Seed Erosion"),
                dcc.Input(id='alpha1', value=0, min=0, type="number", disabled=False),
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
                dcc.Input(id='MinParticle', value=0, min=0, type="number"),
                    
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

def process_image(im_hs):
    im = im_hs.data.astype(np.float64)
    im = im-np.min(im)
    image = np.uint8(255*im/np.max(im))
    return image

@app.callback(
    Output("local_kernel", "disabled"),
    Input("Threshold", "value")
)

def disable_local_kernel(threshold):
    if threshold in ["local", "local_otsu", "lg_otsu", "niblack", "sauvola"]:
        return False
    return True

@app.callback(
    Output("wing_span2", "disabled"),
    Output("alpha1", "disabled"),
    Input('checklist-Watershed-options', 'value')
)

def watershed_options(value):
    if len(value)==0:
        return True,True
    return False,False

@app.callback(
    Output('display', 'figure'),
    Input('display_geometry', 'n_clicks_timestamp'),
    State('rolling', 'value'),
    State('gaussian', 'value'), 
    State('Threshold', 'value'),
    State('local_kernel', 'disabled'),
    State('local_kernel', 'value'),
    State('checklist-Watershed-options', 'value'),
    State('wing_span2', 'value'),
    State('alpha1', 'value'),
    State('checklist-Invert-options', 'value'), 
    State('MinParticle', 'value'),
    State('image_label', 'value'))

def display_geometry(
        xx,
        roll,
        gauss,
        th,
        local_kernel,
        local_kernel_value,
        watershed,
        wing_span2,
        alpha1,
        checklist,
        min_particle_size,
        img_label
):
    # print('check*********',xx,roll,gauss,th,watershed,wing_span2,alpha1,checklist,min_particle_size,img_label,sep='\n')
    print('local kernal',local_kernel)
    print('local_kernel value',local_kernel_value)
    data = hs.load('JEOL HAADF Image.dm4')# autoSTEM_9
    image = process_image(data)
    
    params = parameters()
    params.generate()
#     print('watershed',watershed)
#     print('VALUE OF XX',xx,type(xx))
    if xx=='0':
#         print('FIRST IMAGE')
        figure = px.imshow(image,color_continuous_scale='gray')
    else:
#         print('OTHER IMAGE')
        if roll==1:
            params.segment['rb_kernel'] = 0
        else:
            params.segment['rb_kernel'] = roll

        params.segment['gaussian'] = gauss
        params.segment['threshold'] = th

        if local_kernel==False:
            params.segment['local_size'] = local_kernel_value

        if len(watershed)==1:
            params.segment['watershed'] = True
            params.segment['watershed_size'] = wing_span2
            params.segment['watershed_erosion'] = alpha1
        else:
            params.segment['watershed'] = False
        
        params.segment['min_size'] = min_particle_size

        print('params',params.segment)

        labels = process(data,params)
        labels = np.uint8(labels*(256/labels.max()))
        if img_label == 'Image':
#             print('INSIDE IMAGE')
            b = np.uint8(mark_boundaries(image, labels, color=(1,1,1))[:,:,0]*255)
            if len(checklist)==1:
                figure = px.imshow(invert(b).data, color_continuous_scale='gray')
            else:
                figure = px.imshow(b.data, color_continuous_scale='gray') 

        elif img_label == 'Labels':
            print('Labels:',params.segment)

            figure = px.imshow(labels.data, color_continuous_scale='gray')

    figure.update_layout(
        autosize=True,
        yaxis={'visible': False, 'showticklabels': False},
        xaxis={'visible': False, 'showticklabels': False},
        margin=dict(l=0,r=0,b=0,t=0))

    figure.layout.xaxis.fixedrange = True
    figure.layout.yaxis.fixedrange = True
    
    figure.update_traces(hovertemplate=None,hoverinfo='skip')
    figure.update(layout_coloraxis_showscale=False)

    return figure

app.run_server(port=8088,debug=True)