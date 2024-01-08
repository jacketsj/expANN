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


def prepare_data(file_name, x_axis, y_axis):
    with open(data_folder + file_name, 'r') as dataset_file:
        datavec = json.load(dataset_file)
    data_dict = {'x': [], 'y': [], 'annotations': [], 'engine': []}
    for benchdata in datavec:
        x_value = benchdata.get(x_axis, 0)
        y_value = benchdata.get(y_axis, 0)

        if x_axis == 'time_to_build_ns':
            x_value /= 1e9
        if y_axis == 'time_to_build_ns':
            y_value /= 1e9

        if x_axis == 'time_per_query_ns':
            x_value = 1e9 / x_value if x_value != 0 else 0
        if y_axis == 'time_per_query_ns':
            y_value = 1e9 / y_value if y_value != 0 else 0

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

data_options = {
    'recall': 'Recall',
    'time_per_query_ns': 'Queries per Second',
    'average_distance': 'Average Distance',
    'time_to_build_ns': 'Time to Build (s)'
}

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div([
                    html.Label('X-axis:'),
                    dcc.Dropdown(id='x-axis-selector',
                                 options=[{
                                     'label': v,
                                     'value': k
                                 } for k, v in data_options.items()],
                                 value='recall'),
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
                html.Div(
                    [
                        html.Label('Y-axis:'),
                        dcc.Dropdown(id='y-axis-selector',
                                     options=[{
                                         'label': v,
                                         'value': k
                                     } for k, v in data_options.items()],
                                     value='time_per_query_ns'),
                        dcc.Checklist(id='y-log-scale',
                                      options=[{
                                          'label': 'Log Scale',
                                          'value': 'log'
                                      }],
                                      value=['log'])  # Default to log-scale
                    ],
                    style={
                        'width': '33%',
                        'display': 'inline-block'
                    }),
                html.Div(
                    [
                        html.Label('Data:'),
                        dcc.Dropdown(id='plot-type-selector',
                                     options=[{
                                         'label': 'All Data',
                                         'value': 'all.json'
                                     }, {
                                         'label': 'Latest Data',
                                         'value': 'latest.json'
                                     }],
                                     value='latest.json'
                                     )  # Default to latest data
                    ],
                    style={
                        'width': '33%',
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


@app.callback(Output('data-plot', 'figure'), [
    Input('x-axis-selector', 'value'),
    Input('y-axis-selector', 'value'),
    Input('plot-type-selector', 'value'),
    Input('x-log-scale', 'value'),
    Input('y-log-scale', 'value')
])
def update_graph(x_axis, y_axis, plot_type, x_log_scale, y_log_scale):
    df = prepare_data(plot_type, x_axis, y_axis)
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
    fig.update_layout(
        title=f"{data_options[x_axis]} vs {data_options[y_axis]}",
        xaxis_title=data_options[x_axis],
        yaxis_title=data_options[y_axis],
        xaxis_type="log" if x_log_scale else None,
        yaxis_type="log" if y_log_scale else None)
    return fig


def open_browser():
    webbrowser.open_new_tab("http://127.0.0.1:8050/")


if __name__ == '__main__':
    Thread(
        target=lambda: app.run_server(debug=True, use_reloader=False)).start()
    open_browser()
