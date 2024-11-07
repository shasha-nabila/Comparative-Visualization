import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from dash.exceptions import PreventUpdate

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Global variables to track experiment state
TOTAL_TRIALS = 10
BLANK_DURATION = 1000  # 1 second in milliseconds

# Function to generate random data
def generate_random_data():
    schools = ['School A', 'School B', 'School C', 'School D', 'School E']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    
    data = []
    for school in schools:
        for month in months:
            absences = np.random.randint(5, 50)
            data.append({
                'School': school,
                'Month': month,
                'Absences': absences
            })
    
    return pd.DataFrame(data)

# Function to create heatmap
def create_heatmap(df):
    pivot_df = df.pivot(index='School', columns='Month', values='Absences')
    
    return go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Blues',
            colorbar=dict(title='Absences')
        ),
        layout=go.Layout(
            title='Student Absences by School and Month',
            height=400
        )
    )

# Function to create scatterplot
def create_scatterplot(df):
    return px.scatter(
        df,
        x='Month',
        y='Absences',
        color='School',
        title='Student Absences Over Time',
        height=400
    )

# App layout
app.layout = html.Div([
    # Hidden div for storing experiment state
    dcc.Store(id='experiment-state', data={
        'current_trial': 0,
        'times': [],
        'choices': [],
        'start_time': None
    }),
    
    # Display current trial number
    html.H2(id='trial-header'),
    
    # Container for charts
    html.Div([
        html.Div([
            html.H3('Heatmap'),
            dcc.Graph(id='heatmap', figure={}, clickData=None)
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3('Scatterplot'),
            dcc.Graph(id='scatterplot', figure={}, clickData=None)
        ], style={'width': '50%', 'display': 'inline-block'})
    ], id='charts-container'),
    
    # Interval for timing
    dcc.Interval(id='interval', interval=BLANK_DURATION, disabled=True),
    
    # Results display (hidden until experiment complete)
    html.Div(id='results', style={'display': 'none'})
])

@callback(
    [Output('experiment-state', 'data'),
     Output('charts-container', 'style'),
     Output('trial-header', 'children'),
     Output('interval', 'disabled'),
     Output('heatmap', 'figure'),
     Output('scatterplot', 'figure'),
     Output('results', 'style'),
     Output('results', 'children')],
    [Input('heatmap', 'clickData'),
     Input('scatterplot', 'clickData'),
     Input('interval', 'n_intervals')],
    [State('experiment-state', 'data')]
)
def update_experiment(heatmap_click, scatter_click, n_intervals, exp_state):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        # Initial load
        return exp_state, {'display': 'block'}, f"Trial {exp_state['current_trial'] + 1}/{TOTAL_TRIALS}", True, create_heatmap(generate_random_data()), create_scatterplot(generate_random_data()), {'display': 'none'}, None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id in ['heatmap', 'scatterplot']:
        # Chart was clicked
        current_time = time.time()
        if exp_state['start_time'] is None:
            exp_state['start_time'] = current_time
            return exp_state, {'display': 'block'}, f"Trial {exp_state['current_trial'] + 1}/{TOTAL_TRIALS}", True, dash.no_update, dash.no_update, {'display': 'none'}, None
        
        response_time = current_time - exp_state['start_time']
        exp_state['times'].append(response_time)
        exp_state['choices'].append(trigger_id)
        exp_state['current_trial'] += 1
        exp_state['start_time'] = None
        
        # Show blank screen
        if exp_state['current_trial'] < TOTAL_TRIALS:
            return exp_state, {'display': 'none'}, f"Trial {exp_state['current_trial'] + 1}/{TOTAL_TRIALS}", False, dash.no_update, dash.no_update, {'display': 'none'}, None
        else:
            # Experiment complete, show results
            results_html = html.Div([
                html.H2('Experiment Complete'),
                html.H3('Results:'),
                html.Ul([
                    html.Li(f"Average response time for Heatmap: {np.mean([t for i, t in enumerate(exp_state['times']) if exp_state['choices'][i] == 'heatmap']):.2f} seconds"),
                    html.Li(f"Average response time for Scatterplot: {np.mean([t for i, t in enumerate(exp_state['times']) if exp_state['choices'][i] == 'scatterplot']):.2f} seconds")
                ])
            ])
            return exp_state, {'display': 'none'}, 'Experiment Complete', True, dash.no_update, dash.no_update, {'display': 'block'}, results_html
    
    elif trigger_id == 'interval':
        # Interval triggered - show new charts
        return exp_state, {'display': 'block'}, f"Trial {exp_state['current_trial'] + 1}/{TOTAL_TRIALS}", True, create_heatmap(generate_random_data()), create_scatterplot(generate_random_data()), {'display': 'none'}, None
    
    raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=False)
