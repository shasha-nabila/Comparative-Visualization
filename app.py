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
server = app.server  # Add this for deployment

# Global variables to track experiment state
TOTAL_TRIALS = 10
BLANK_DURATION = 1000  # 1 second in milliseconds

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
    dcc.Store(id='experiment-state', data={
        'current_trial': 0,
        'times': [],
        'choices': [],
        'start_time': None,
        'waiting_for_click': True  # New flag to track if we're waiting for a click
    }),
    
    html.H2(id='trial-header'),
    
    # Instructions
    html.Div([
        html.P("Click on the visualization that helps you complete the task more quickly.", 
               style={'fontSize': '18px', 'marginBottom': '20px'})
    ], id='instructions'),
    
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
    
    dcc.Interval(id='interval', interval=BLANK_DURATION, disabled=True),
    
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
        data = generate_random_data()
        return exp_state, {'display': 'block'}, f"Trial {exp_state['current_trial'] + 1}/{TOTAL_TRIALS}", True, create_heatmap(data), create_scatterplot(data), {'display': 'none'}, None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Only process clicks if we're waiting for one
    if trigger_id in ['heatmap', 'scatterplot'] and exp_state['waiting_for_click']:
        current_time = time.time()
        
        # Record the click and timing
        if exp_state['start_time'] is not None:
            response_time = current_time - exp_state['start_time']
            exp_state['times'].append(response_time)
            exp_state['choices'].append(trigger_id)
            exp_state['current_trial'] += 1
            exp_state['waiting_for_click'] = False
            
            # Show blank screen
            if exp_state['current_trial'] < TOTAL_TRIALS:
                exp_state['start_time'] = None
                return exp_state, {'display': 'none'}, f"Trial {exp_state['current_trial'] + 1}/{TOTAL_TRIALS}", False, dash.no_update, dash.no_update, {'display': 'none'}, None
            else:
                # Experiment complete, show results
                heatmap_times = [t for i, t in enumerate(exp_state['times']) if exp_state['choices'][i] == 'heatmap']
                scatter_times = [t for i, t in enumerate(exp_state['times']) if exp_state['choices'][i] == 'scatterplot']
                
                results_html = html.Div([
                    html.H2('Experiment Complete'),
                    html.H3('Results:'),
                    html.Div([
                        html.P(f"Number of Heatmap selections: {len(heatmap_times)}"),
                        html.P(f"Number of Scatterplot selections: {len(scatter_times)}"),
                        html.P(f"Average response time for Heatmap: {np.mean(heatmap_times):.2f} seconds"),
                        html.P(f"Average response time for Scatterplot: {np.mean(scatter_times):.2f} seconds")
                    ])
                ])
                return exp_state, {'display': 'none'}, 'Experiment Complete', True, dash.no_update, dash.no_update, {'display': 'block'}, results_html
        
        # Start timing if this is the first click
        exp_state['start_time'] = current_time
        return exp_state, {'display': 'block'}, f"Trial {exp_state['current_trial'] + 1}/{TOTAL_TRIALS}", True, dash.no_update, dash.no_update, {'display': 'none'}, None
    
    elif trigger_id == 'interval':
        # Show new charts after blank screen
        data = generate_random_data()
        exp_state['waiting_for_click'] = True
        return exp_state, {'display': 'block'}, f"Trial {exp_state['current_trial'] + 1}/{TOTAL_TRIALS}", True, create_heatmap(data), create_scatterplot(data), {'display': 'none'}, None
    
    raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)