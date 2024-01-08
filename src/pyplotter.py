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


def prepare_data(file_name, recall_threshold=0.0):
    with open(data_folder + file_name, 'r') as dataset_file:
        datavec = json.load(dataset_file)
    data_dict = {'recall': [], 'qps': [], 'annotations': [], 'engine': []}
    for benchdata in datavec:
        if benchdata['recall'] >= recall_threshold:
            data_dict['recall'].append(benchdata['recall'])
            data_dict['qps'].append(1e9 / benchdata['time_per_query_ns'])

            # Separate properties and param_list
            properties = {
                k: v
                for k, v in benchdata.items() if k != 'param_list'
            }
            param_list = benchdata.get('param_list', {})

            # Move 'engine_name' to the front if it exists
            if 'engine_name' in properties:
                properties = {
                    'engine_name': properties['engine_name'],
                    **properties
                }

            # Custom pretty-printing
            properties_str = "<br>".join([
                f"{k}: {json.dumps(v, indent=4)}"
                for k, v in properties.items()
            ])
            param_list_str = "<br>".join([
                f"{k}: {json.dumps(v, indent=4)}"
                for k, v in param_list.items()
            ])

            annotation = (f"<b>Statistics:</b><br>{properties_str}"
                          f"<br><br><b>Param List:</b><br>{param_list_str}")
            annotation = annotation.replace('"', '').replace('{', '').replace(
                '}', '').replace(',', '')
            annotation = annotation.replace("engine_name: ", "")
            data_dict['annotations'].append(annotation)
            data_dict['engine'].append(benchdata['engine_name'])
    return pd.DataFrame(data_dict)


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(id='file-selector',
                 options=[{
                     'label': 'All Data',
                     'value': 'all.json'
                 }, {
                     'label': 'Latest Data',
                     'value': 'latest.json'
                 }],
                 value='all.json'),
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


# Update the graph
@app.callback(
    Output('data-plot', 'figure'),
    [Input('file-selector', 'value'),
     Input('recall-slider', 'value')])
def update_graph(selected_file, recall_threshold):
    df = prepare_data(selected_file, recall_threshold)
    fig = go.Figure()
    for engine in df['engine'].unique():
        engine_df = df[df['engine'] == engine]
        fig.add_trace(
            go.Scatter(
                x=engine_df['recall'],
                y=engine_df['qps'],
                mode='markers',
                name=engine,
                hovertext=engine_df['annotations'],
                hoverinfo='text',
                hoverlabel=dict(
                    namelength=-1)  # Allow long text in hover labels
            ))
    fig.update_layout(title=f"Recall-QPS of {k_value}-NN for {dataset}",
                      xaxis_title="Recall",
                      yaxis_title="QPS",
                      yaxis_type="log")
    return fig


def open_browser():
    webbrowser.open_new_tab("http://127.0.0.1:8050/")


if __name__ == '__main__':
    Thread(
        target=lambda: app.run_server(debug=True, use_reloader=False)).start()
    open_browser()
