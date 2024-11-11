import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from dash.exceptions import PreventUpdate
import random
import csv
import os

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variables to track experiment state
TOTAL_PAIRS = 10  # 10 pairs of trials (20 total trials)
BLANK_DURATION = 1000  # 1 second in milliseconds
TOTAL_SHEETS = 10  # Total number of data sheets

# Function to generate trial sequence
# def generate_trial_sequence():
#     # Create pairs for sheets 1-5 (highest value questions)
#     high_value_pairs = []
#     for sheet_idx in range(5):
#         if random.random() < 0.5:
#             high_value_pairs.extend([
#                 {'trial_type': 'heatmap', 'sheet_index': sheet_idx},
#                 {'trial_type': 'scatterplot', 'sheet_index': sheet_idx}
#             ])
#         else:
#             high_value_pairs.extend([
#                 {'trial_type': 'scatterplot', 'sheet_index': sheet_idx},
#                 {'trial_type': 'heatmap', 'sheet_index': sheet_idx}
#             ])
    
#     # Create pairs for sheets 6-10 (lowest value questions)
#     low_value_pairs = []
#     for sheet_idx in range(5, 10):
#         if random.random() < 0.5:
#             low_value_pairs.extend([
#                 {'trial_type': 'heatmap', 'sheet_index': sheet_idx},
#                 {'trial_type': 'scatterplot', 'sheet_index': sheet_idx}
#             ])
#         else:
#             low_value_pairs.extend([
#                 {'trial_type': 'scatterplot', 'sheet_index': sheet_idx},
#                 {'trial_type': 'heatmap', 'sheet_index': sheet_idx}
#             ])
    
#     # Combine and shuffle all pairs
#     all_pairs = high_value_pairs + low_value_pairs
#     random.shuffle(all_pairs)
#     return all_pairs


def generate_trial_sequence():
    """
    Generates a randomized sequence of trials ensuring:
    1. Each dataset (0-9) is used exactly twice - once for heatmap and once for scatterplot
    2. The presentation order is randomized
    3. The pairing between visualization types is not fixed
    
    Returns:
        list: List of dictionaries containing trial_type and sheet_index
    """
    # Create list of all possible trials (10 heatmaps and 10 scatterplots)
    all_trials = []
    
    # Add one heatmap and one scatterplot trial for each dataset
    for sheet_idx in range(10):  # 0-9 for all datasets
        all_trials.extend([
            {'trial_type': 'heatmap', 'sheet_index': sheet_idx},
            {'trial_type': 'scatterplot', 'sheet_index': sheet_idx}
        ])
    
    # Shuffle all trials to randomize presentation order
    random.shuffle(all_trials)
    
    return all_trials


# Function to load Excel data
def load_excel_data(sheet_name):
    # Replace this with your actual Excel file path
    df = pd.read_excel('cwk.xlsx', sheet_name=sheet_name, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    # Convert the dataframe to numeric values, excluding the index/column headers
    numeric_df = df.copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    
    return numeric_df

# Function to find extreme value and its location
def find_extreme_value(df, find_highest=True):
    """
    Find extreme value (highest or lowest) and its location in the dataframe
    with proper handling of string-to-numeric conversion.
    
    Args:
        df (pd.DataFrame): Input dataframe
        find_highest (bool): If True, find maximum value; if False, find minimum
    
    Returns:
        dict: Dictionary containing value, school, and month
    """
    # Create a copy to avoid modifying original
    numeric_df = df.copy()
    
    # Ensure all values are properly converted to numeric
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    
    # Get values array
    values = numeric_df.values
    
    if find_highest:
        value = np.nanmax(values)
        locations = np.where(values == value)
    else:
        value = np.nanmin(values)
        locations = np.where(values == value)
    
    # Get all instances of the extreme value
    extreme_locations = list(zip(locations[0], locations[1]))
    
    # If there are multiple instances of the extreme value,
    # prioritize by position (can modify this logic if needed)
    row_idx, col_idx = extreme_locations[0]
    
    # Get the school (row) and month (column) for the extreme value
    school = df.index[row_idx]
    month = df.columns[col_idx]
    
    # Create return dictionary with proper type conversion
    result = {
        'value': float(value),
        'school': str(school).strip(),
        'month': str(month).strip()
    }
    
    return result

def find_extreme_value_in_month(df, month, find_highest=True):
    """
    Find extreme value in a specific month
    """
    # Get the values for the specified month
    month_values = df[month].astype(float)
    
    if find_highest:
        value = month_values.max()
    else:
        value = month_values.min()
    
    # Get the school (row index) where this value occurs
    school = month_values[month_values == value].index[0]
    
    return {
        'value': float(value),
        'school': str(school).strip(),
        'month': str(month).strip()
    }

# Function to generate question and answer
def generate_question(df, chart_type, sheet_index):
    """
    Generate question and answer based on sheet index.
    For sheets 0-4: Find highest value overall
    For sheets 5-9: Find lowest value in specific month
    """
    # Predefined months for sheets 5-9
    predetermined_months = {
        5: 'February',
        6: 'March',
        7: 'April',
        8: 'June',
        9: 'November'
    }
    
    if sheet_index < 5:
        # For sheets 0-4, find highest value as before
        extreme_info = find_extreme_value(df, find_highest=True)
        question = f"Click on the highest absence rate in {extreme_info['month']} in the {'heatmap' if chart_type == 'heatmap' else 'scatter plot'}"
    else:
        # For sheets 5-9, find lowest value in predetermined month
        target_month = predetermined_months[sheet_index]
        extreme_info = find_extreme_value_in_month(df, target_month, find_highest=False)
        question = f"Click on the lowest absence rate in {target_month} in the {'heatmap' if chart_type == 'heatmap' else 'scatter plot'}"
    
    return question, extreme_info

# Function to check if clicked value matches the answer
def check_answer(click_data, correct_answer, chart_type):
    """
    Check if clicked value matches the answer
    """
    try:
        if chart_type == 'heatmap':
            clicked_row = str(click_data['points'][0]['y'])
            clicked_col = str(click_data['points'][0]['x'])
            clicked_value = float(click_data['points'][0]['z'])
        else:  # scatterplot
            clicked_value = float(click_data['points'][0]['y'])
            clicked_col = str(click_data['points'][0]['x'])
            clicked_row = str(click_data['points'][0]['text'])
        
        # Check if clicked position matches the correct answer
        value_matches = abs(clicked_value - float(correct_answer['value'])) < 0.001
        position_matches = (clicked_row.strip() == str(correct_answer['school']).strip() and 
                          clicked_col.strip() == str(correct_answer['month']).strip())
        
        return value_matches and position_matches
    except Exception as e:
        print(f"Error checking answer: {e}")
        return False
    

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
            title="Percentage of Pupil Absence Rates Across 10 Schools in Town",
            title_x=0.5,
            height=600
        )
    )

# Function to create scatterplot
def create_scatterplot(df, question):
    melted_df = df.reset_index().melt(id_vars='index', var_name='Month', value_name='Value')
    melted_df.columns = ['School', 'Month', 'Value']
    
    # Create a list to hold individual Scatter traces for each school
    scatter_traces = []

    # Loop over each unique school to create a trace with its own color
    for school in melted_df['School'].unique():
        school_data = melted_df[melted_df['School'] == school]
        
        # Add a scatter trace for each school with unique color and legend entry
        scatter_traces.append(
            go.Scatter(
                x=school_data['Month'],
                y=school_data['Value'],
                mode='markers',
                marker=dict(size=12, opacity=0.7),  # Set opacity for visibility
                name=f'School {school}',  # Legend entry
                text=school_data['School'],
                hovertemplate='School: %{text}<br>Month: %{x}<br>Value: %{y:.2f}<extra></extra>'
            )
        )

    # Return the figure with all individual school traces
    return go.Figure(
        data=scatter_traces,
        layout=dict(
            title="Percentage of Pupil Absence Rates Across 10 Schools in Town",
            title_x=0.5,  # Center title
            height=600,
            showlegend=True  # Show the legend
        )
    )

    '''
    return go.Figure(
        data=go.Scatter(
            x=melted_df['Month'],
            y=melted_df['Value'],
            mode='markers',
            marker=dict(
                size=12,
                opacity=0.7 # Set opacity to 70% for overlappping visibility
            ),
            text=melted_df['School'],
            hovertemplate='School: %{text}<br>Month: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ),
        layout=dict(
            title="Percentage of Pupil Absence Rates Across 10 Schools in Town",
            title_x=0.5,
            height=600,
            showlegend=True
        )
    )
    '''


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
        question, answer = generate_question(df, current_trial['trial_type'], current_trial['sheet_index'])
        exp_state['current_answer'] = answer
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
    
    if trigger_id == 'chart' and click_data is not None:
        # Chart was clicked
        current_time = time.time()
        response_time = current_time - exp_state['start_time']
        
        # Record trial data
        current_trial = exp_state['trial_sequence'][exp_state['current_trial']]
        exp_state['times'].append(response_time)
        exp_state['choices'].append(current_trial['trial_type'])
        
        # Check if answer is correct
        is_correct = check_answer(click_data, exp_state['current_answer'], current_trial['trial_type'])
        exp_state['correct_answers'].append(is_correct)
        if is_correct:
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
                #html.H2('Experiment Complete'),
                html.H3('Results:'),
                html.Ul([
                    html.Li(f"Average response time for Heatmap: {np.mean(heatmap_times):.2f} seconds"),
                    html.Li(f"Average response time for Scatterplot: {np.mean(scatter_times):.2f} seconds"),
                    html.Li(f"Correct answers for Heatmap: {heatmap_correct}/{TOTAL_PAIRS}"),
                    html.Li(f"Correct answers for Scatterplot: {scatter_correct}/{TOTAL_PAIRS}"),
                    html.Li(f"Total correct answers: {exp_state['correct_count']}/{TOTAL_PAIRS * 2}")
                ])
            ])

            # Prepare data for CSV
            avg_heatmap_time = np.mean(heatmap_times)
            avg_scatter_time = np.mean(scatter_times)
            heatmap_correct_percent = (heatmap_correct / TOTAL_PAIRS) * 100
            scatter_correct_percent = (scatter_correct / TOTAL_PAIRS) * 100
            total_correct = exp_state['correct_count']

            # Define headers and data row
            headers = [
                "Participant ID",
                "Heatmap Average Response Time (s)",
                "Scatterplot Average Response Time (s)",
                "Heatmap % Correct",
                "Scatterplot % Correct",
                "Total correct (out of 20)"
            ]
            data_row = [
                None,  # Placeholder for Participant ID
                f"{avg_heatmap_time:.2f}",
                f"{avg_scatter_time:.2f}",
                f"{heatmap_correct_percent:.1f}",
                f"{scatter_correct_percent:.1f}",
                total_correct
            ]

            # Check if results.csv exists to determine starting Participant ID and add header if necessary
            file_exists = os.path.isfile("results.csv")
            with open("results.csv", mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header if the file is new
                if not file_exists:
                    writer.writerow(headers)
                
                # Determine the next Participant ID
                if file_exists:
                    with open("results.csv", mode="r", newline="") as readfile:
                        reader = csv.reader(readfile)
                        row_count = sum(1 for row in reader) - 1  # Exclude header
                else:
                    row_count = 0  # File is new, so no rows

                # Update Participant ID and write the data row
                data_row[0] = row_count + 1  # Participant ID is next row count
                writer.writerow(data_row)
            
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
        question, answer = generate_question(df, current_trial['trial_type'], current_trial['sheet_index'])
        exp_state['current_answer'] = answer
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