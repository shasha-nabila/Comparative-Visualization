import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from dash.exceptions import PreventUpdate
import random

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variables to track experiment state
TOTAL_PAIRS = 10  # 10 pairs of trials (20 total trials)
BLANK_DURATION = 1000  # 1 second in milliseconds
TOTAL_SHEETS = 10  # Total number of data sheets

# Function to generate trial sequence
def generate_trial_sequence():
    sheet_indices = list(range(TOTAL_SHEETS))
    random.shuffle(sheet_indices)
    
    sequence = []
    for sheet_idx in sheet_indices:
        if random.random() < 0.5:
            sequence.extend([
                {'trial_type': 'heatmap', 'sheet_index': sheet_idx},
                {'trial_type': 'scatterplot', 'sheet_index': sheet_idx}
            ])
        else:
            sequence.extend([
                {'trial_type': 'scatterplot', 'sheet_index': sheet_idx},
                {'trial_type': 'heatmap', 'sheet_index': sheet_idx}
            ])
    return sequence

# Function to load Excel data
def load_excel_data(sheet_name):
    # Replace this with your actual Excel file path
    df = pd.read_excel('cwk.xlsx', sheet_name=sheet_name)
    return df

# Function to create random question and correct answer
def generate_question(df, chart_type):
    if chart_type == 'heatmap':
        school = np.random.choice(df.index)
        month = np.random.choice(df.columns)
        value = df.loc[school, month]
        question = f"Click on the value {value:.2f} in the heatmap"
        correct_answer = {'school': school, 'month': month, 'value': value}
    else:
        month = np.random.choice(df.columns)
        school = np.random.choice(df.index)
        value = df.loc[school, month]
        question = f"Click on the point for School {school} in {month}"
        correct_answer = {'school': school, 'month': month, 'value': value}
    
    return question, correct_answer

# Function to create heatmap
def create_heatmap(df, question):
    return go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale='Blues',
            colorbar=dict(title='Values')
        ),
        layout=go.Layout(
            title=question,
            height=600
        )
    )

# Function to create scatterplot
def create_scatterplot(df, question):
    melted_df = df.reset_index().melt(id_vars='index', var_name='Month', value_name='Value')
    melted_df.columns = ['School', 'Month', 'Value']
    
    return go.Figure(
        data=go.Scatter(
            x=melted_df['Month'],
            y=melted_df['Value'],
            mode='markers',
            marker=dict(size=12),
            text=melted_df['School'],
            hovertemplate='School: %{text}<br>Month: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ),
        layout=dict(
            title=question,
            height=600,
            showlegend=False
        )
    )

# App layout
app.layout = html.Div([
    dcc.Store(id='experiment-state', data={
        'current_trial': 0,
        'times': [],
        'choices': [],
        'correct_answers': [],
        'start_time': None,
        'current_answer': None,
        'correct_count': 0,
        'trial_sequence': generate_trial_sequence()
    }),
    
    html.H2(id='trial-header'),
    html.H3(id='question-text'),
    
    html.Div([
        dcc.Graph(id='chart', figure={})
    ], id='chart-container'),
    
    dcc.Interval(id='interval', interval=BLANK_DURATION, disabled=True),
    
    html.Div(id='results', style={'display': 'none'})
])

@callback(
    [Output('experiment-state', 'data'),
     Output('chart-container', 'style'),
     Output('trial-header', 'children'),
     Output('question-text', 'children'),
     Output('interval', 'disabled'),
     Output('chart', 'figure'),
     Output('results', 'style'),
     Output('results', 'children')],
    [Input('chart', 'clickData'),
     Input('interval', 'n_intervals')],
    [State('experiment-state', 'data')]
)
def update_experiment(click_data, n_intervals, exp_state):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        # Initial load - show first chart
        current_trial = exp_state['trial_sequence'][0]
        df = load_excel_data(f'Data{current_trial["sheet_index"] + 1}')
        question, correct_answer = generate_question(df, current_trial['trial_type'])
        exp_state['current_answer'] = correct_answer
        exp_state['start_time'] = time.time()
        
        figure = create_heatmap(df, question) if current_trial['trial_type'] == 'heatmap' else create_scatterplot(df, question)
        
        return (
            exp_state,
            {'display': 'block'},
            f"Trial {exp_state['current_trial'] + 1}/{TOTAL_PAIRS * 2}",
            question,
            True,
            figure,
            {'display': 'none'},
            None
        )
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'chart' and click_data is not None:  # Added click_data check
        # Chart was clicked
        current_time = time.time()
        response_time = current_time - exp_state['start_time']
        
        # Record trial data
        current_trial = exp_state['trial_sequence'][exp_state['current_trial']]
        exp_state['times'].append(response_time)
        exp_state['choices'].append(current_trial['trial_type'])
        
        # Simplified answer checking (always correct for now)
        exp_state['correct_answers'].append(True)
        exp_state['correct_count'] += 1
        
        # Move to next trial
        exp_state['current_trial'] += 1
        
        # Check if experiment is complete
        if exp_state['current_trial'] >= TOTAL_PAIRS * 2:
            # Calculate results
            heatmap_times = [t for i, t in enumerate(exp_state['times']) 
                           if exp_state['choices'][i] == 'heatmap']
            scatter_times = [t for i, t in enumerate(exp_state['times'])
                           if exp_state['choices'][i] == 'scatterplot']
            
            heatmap_correct = sum(1 for i, correct in enumerate(exp_state['correct_answers'])
                                if exp_state['choices'][i] == 'heatmap' and correct)
            scatter_correct = sum(1 for i, correct in enumerate(exp_state['correct_answers'])
                                if exp_state['choices'][i] == 'scatterplot' and correct)
            
            results_html = html.Div([
                html.H2('Experiment Complete'),
                html.H3('Results:'),
                html.Ul([
                    html.Li(f"Average response time for Heatmap: {np.mean(heatmap_times):.2f} seconds"),
                    html.Li(f"Average response time for Scatterplot: {np.mean(scatter_times):.2f} seconds"),
                    html.Li(f"Correct answers for Heatmap: {heatmap_correct}/{TOTAL_PAIRS}"),
                    html.Li(f"Correct answers for Scatterplot: {scatter_correct}/{TOTAL_PAIRS}"),
                    html.Li(f"Total correct answers: {exp_state['correct_count']}/{TOTAL_PAIRS * 2}")
                ])
            ])
            
            return (
                exp_state,
                {'display': 'none'},
                'Experiment Complete',
                "",
                True,
                dash.no_update,
                {'display': 'block'},
                results_html
            )
        
        # Show blank screen between trials
        return (
            exp_state,
            {'display': 'none'},
            f"Trial {exp_state['current_trial'] + 1}/{TOTAL_PAIRS * 2}",
            "",
            False,
            dash.no_update,
            {'display': 'none'},
            None
        )
    
    elif trigger_id == 'interval':
        # Show next chart after blank screen
        current_trial = exp_state['trial_sequence'][exp_state['current_trial']]
        df = load_excel_data(f'Data{current_trial["sheet_index"] + 1}')
        question, correct_answer = generate_question(df, current_trial['trial_type'])
        exp_state['current_answer'] = correct_answer
        exp_state['start_time'] = time.time()
        
        figure = create_heatmap(df, question) if current_trial['trial_type'] == 'heatmap' else create_scatterplot(df, question)
        
        return (
            exp_state,
            {'display': 'block'},
            f"Trial {exp_state['current_trial'] + 1}/{TOTAL_PAIRS * 2}",
            question,
            True,
            figure,
            {'display': 'none'},
            None
        )
    
    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)