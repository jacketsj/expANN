import dash
from dash import dcc, html
import plotly.graph_objs as go
import json
import pandas as pd
from dash.dependencies import Input, Output
import webbrowser
from threading import Thread

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
ds_name = config['ds_name']
dataset = config['dataset']
k_value = config['k']

data_folder = f'./data/{ds_name}/data/'


def get_param_options(datavec):
    param_options = set()
    for data in datavec:
        param_options.update(data.get('param_list', {}).keys())
    return [{'label': param, 'value': param} for param in param_options]


def prepare_data(file_name, x_axis, y_axis, use_param_x, use_param_y):
    with open(data_folder + file_name, 'r') as dataset_file:
        datavec = json.load(dataset_file)

    data_dict = {'x': [], 'y': [], 'annotations': [], 'engine': []}
    for benchdata in datavec:
        if use_param_x and x_axis not in benchdata.get('param_list', {}):
            continue
        if use_param_y and y_axis not in benchdata.get('param_list', {}):
            continue
        x_value = benchdata.get('param_list', {}).get(
            x_axis, 0) if use_param_x else benchdata.get(x_axis, 0)
        y_value = benchdata.get('param_list', {}).get(
            y_axis, 0) if use_param_y else benchdata.get(y_axis, 0)

        if x_axis == 'time_to_build_ns':
            x_value /= 1e9
        if y_axis == 'time_to_build_ns':
            y_value /= 1e9

        if x_axis == 'time_per_query_ns':
            x_value = 1e9 / x_value if x_value != 0 else 0
        if y_axis == 'time_per_query_ns':
            y_value = 1e9 / y_value if y_value != 0 else 0

        if str(float(x_value)) == x_value:
            x_value = float(x_value)
        if str(float(y_value)) == y_value:
            y_value = float(y_value)
        data_dict['x'].append(x_value)
        data_dict['y'].append(y_value)

        annotation = "<b>Statistics:</b><br>"
        annotation += "<br>".join([
            f"{k}: {v}" for k, v in benchdata.items()
            if k not in ['param_list', x_axis, y_axis]
        ])
        annotation += "<br><br><b>Param List:</b><br>"
        annotation += "<br>".join(
            [f"{k}: {v}" for k, v in benchdata.get('param_list', {}).items()])
        annotation = annotation.replace('"', '').replace('{', '').replace(
            '}', '').replace(',', '')
        data_dict['annotations'].append(annotation)
        data_dict['engine'].append(benchdata['engine_name'])
    return pd.DataFrame(data_dict)


app = dash.Dash(__name__)

data_options_map = {
    'recall': 'Recall',
    'time_per_query_ns': 'Queries per Second',
    'average_distance': 'Average Distance',
    'time_to_build_ns': 'Time to Build (s)'
}

data_options = [{
    'label': 'Recall',
    'value': 'recall'
}, {
    'label': 'Queries per Second',
    'value': 'time_per_query_ns'
}, {
    'label': 'Average Distance',
    'value': 'average_distance'
}, {
    'label': 'Time to Build (s)',
    'value': 'time_to_build_ns'
}]

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label('X-axis:'),
            dcc.Dropdown(
                id='x-axis-selector', options=data_options, value='recall'),
            dcc.Checklist(id='x-use-param',
                          options=[{
                              'label': 'Use Param',
                              'value': 'use_param'
                          }],
                          value=[]),
            dcc.Checklist(id='x-log-scale',
                          options=[{
                              'label': 'Log Scale',
                              'value': 'log'
                          }],
                          value=[])
        ],
                 style={
                     'width': '33%',
                     'display': 'inline-block'
                 }),
        html.Div([
            html.Label('Y-axis:'),
            dcc.Dropdown(id='y-axis-selector',
                         options=data_options,
                         value='time_per_query_ns'),
            dcc.Checklist(id='y-use-param',
                          options=[{
                              'label': 'Use Param',
                              'value': 'use_param'
                          }],
                          value=[]),
            dcc.Checklist(id='y-log-scale',
                          options=[{
                              'label': 'Log Scale',
                              'value': 'log'
                          }],
                          value=['log'])
        ],
                 style={
                     'width': '33%',
                     'display': 'inline-block'
                 }),
        html.Div([
            html.Label('Data:'),
            dcc.Dropdown(id='plot-type-selector',
                         options=[{
                             'label': 'All Data',
                             'value': 'all.json'
                         }, {
                             'label': 'Latest Data',
                             'value': 'latest.json'
                         }],
                         value='latest.json')
        ],
                 style={
                     'width': '34%',
                     'display': 'inline-block'
                 })
    ],
             style={
                 'display': 'flex',
                 'width': '100%'
             }),
    dcc.Graph(id='data-plot', style={'height': 'calc(95vh - 60px)'}),
    dcc.Slider(id='recall-slider',
               min=0.0,
               max=1.0,
               step=0.1,
               value=0.0,
               marks={i / 10: str(i / 10)
                      for i in range(0, 11)})
],
                      style={
                          'display': 'flex',
                          'flexDirection': 'column',
                          'height': '95vh'
                      })


@app.callback([
    Output('x-axis-selector', 'options'),
    Output('y-axis-selector', 'options')
], [
    Input('plot-type-selector', 'value'),
    Input('x-use-param', 'value'),
    Input('y-use-param', 'value')
])
def update_axis_options(plot_type, use_param_x, use_param_y):
    with open(data_folder + plot_type, 'r') as dataset_file:
        datavec = json.load(dataset_file)
    param_options = get_param_options(datavec)
    x_options = param_options if 'use_param' in use_param_x else data_options
    y_options = param_options if 'use_param' in use_param_y else data_options
    return x_options, y_options


@app.callback(Output('data-plot', 'figure'), [
    Input('x-axis-selector', 'value'),
    Input('y-axis-selector', 'value'),
    Input('plot-type-selector', 'value'),
    Input('x-log-scale', 'value'),
    Input('y-log-scale', 'value'),
    Input('x-use-param', 'value'),
    Input('y-use-param', 'value')
])
def update_graph(x_axis, y_axis, plot_type, x_log_scale, y_log_scale,
                 x_use_param, y_use_param):
    x_axis_name = x_axis
    if x_axis in data_options_map:
        x_axis_name = data_options_map[x_axis]
    y_axis_name = y_axis
    if y_axis in data_options_map:
        y_axis_name = data_options_map[y_axis]
    df = prepare_data(plot_type, x_axis, y_axis, 'use_param' in x_use_param,
                      'use_param' in y_use_param)
    fig = go.Figure()
    for engine in df['engine'].unique():
        engine_df = df[df['engine'] == engine]
        fig.add_trace(
            go.Scatter(x=engine_df['x'],
                       y=engine_df['y'],
                       mode='markers',
                       name=engine,
                       hovertext=engine_df['annotations'],
                       hoverinfo='text',
                       hoverlabel=dict(namelength=-1)))
    fig.update_layout(title=f"{x_axis_name} vs {y_axis_name}",
                      xaxis_title=x_axis_name,
                      yaxis_title=y_axis_name,
                      xaxis_type="log" if 'log' in x_log_scale else None,
                      yaxis_type="log" if 'log' in y_log_scale else None)
    return fig


def open_browser():
    webbrowser.open_new_tab("http://127.0.0.1:8050/")


if __name__ == '__main__':
    Thread(
        target=lambda: app.run_server(debug=True, use_reloader=False)).start()
    open_browser()
