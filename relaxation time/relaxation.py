# relaxation.py

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
from typing import List


class scatter_data():
    def __init__(self, dataframe, column, column_type, label='', scale='linear', marker='.', color='b', legend_label='', legend_location='upper left', marker_size=1, opacity=1):
        """
        Organizes the x-values or y-values of the data for plotting a graph in the scatter_plot function.

        Parameters:
        - self (data): Reference to the instance of the class.
        - dataframe (pd.Dataframe): Input data_frame that has the provided y_column within it.
        - column (str): Column name for the axis.
        - column_type (str): Represents the type of the axis it will be where:
                             - x_column = Column for the x-axis.
                             - y_column = Column for the y-axis.
                             - y_prime_column = Column for the secondary y-axis.
        - label(str, optional): Label to be put on the axis for that data (default is no label "").
        - scale (str, optional): scale for the axis (default is "linear"). 
        - marker (str, optional): Marker that is used to represent that particular data column (default is '.').
        - legend_label (str, optional): Label for the legend (default is no legend label "").
        - legend_location (str, optional): Represents where the legend should be placed. There are 4 general locations:
                                           - upper left
                                           - upper right
                                           - lower left
                                           - lower right
        - marker_size (int, optional): Marker size (default is 1).
        - opacity(double, optional): Opacity (between 0 and 1).
        

        Raises:
        - ValueError: If the specified column_type is not one of the allowed types.
        - ValueError: If the specified legend_location in not one of the allowed types.

        Returns:
        - None
        """

        self.dataframe = dataframe
        self.column = column
        self.column_type = column_type
        self.label = label
        self.scale = scale
        self.marker = marker
        self.legend_label = legend_label
        self.legend_location = legend_location
        self.marker_size = marker_size
        self.color = color
        self.opacity = opacity
       
        if self.column_type not in ['x_column', 'y_column', 'y_prime_column']:
            raise ValueError("Invalid column_type. It must be one of: 'x_column', 'y_column', 'y_prime_column'")
        if self.legend_location not in ['upper left', 'upper right', 'lower left', 'lower right']:
            raise ValueError ("Invalid legend_location. It must be one of: 'upper left', upper right', 'lower left', 'lower right'")
        if self.opacity > 1 or self.opacity < 0:
            raise ValueError ("Value of opacity must be between 0 and 1.")
            
class line_data():
    def __init__(self, x_values, y_values, legend_label='', x_label='', y_label='', x_scale='linear', y_scale='linear', legend_location='upper left', color='', opacity = 1):
        """
        Organizes the x-values or y-values of the data for plotting a line plot in the scatter_plot function.

        Parameters:
        - self (line_data): Reference to the instance of the class.
        - x_values (array-like): Array or list representing the x-values of the data.
        - y_values (array-like): Array or list representing the y-values of the data.
        - legend_label (str, optional): Label for the legend (default is an empty string).
        - x_label (str, optional): Label for the x-axis (default is an empty string).
        - y_label (str, optional): Label for the y-axis (default is an empty string).
        - x_scale (str, optional): Scale for the x-axis (default is "linear").
        - y_scale (str, optional): Scale for the y-axis (default is "linear").
        - legend_location (str, optional): Represents where the legend should be placed. Allowed values are:
                                        - 'upper left'
                                        - 'upper right'
                                        - 'lower left'
                                        - 'lower right'
                                        (default is 'upper left').
        - color (str, optional): Color of the line in the plot (default is 'b' for blue).

        Raises:
        - ValueError: If the specified column_type is not one of the allowed types.

        Returns:
        - None

        """
        self.x_values = x_values
        self.y_values = y_values
        self.legend_label = legend_label
        self.x_label = x_label
        self.y_label = y_label
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.legend_location = legend_location
        self.color = color
        self.opacity = opacity

        if self.legend_location not in ['upper left', 'upper right', 'lower left', 'lower right']:
            raise ValueError ("Invalid legend_location. It must be one of: 'upper left', upper right', 'lower left', 'lower right'")
        if self.opacity > 1 or self.opacity < 0:
            raise ValueError ("Value of opacity must be between 0 and 1.")


# import matplotlib as plt
def scatter_plot(title, x_column, y_column, new_column: dict[scatter_data, scatter_data] = {}, new_line: List[line_data] =[]):

    """
    Plot a scatter plot using data from a pandas DataFrame.

    Parameters:
    - title
    - x_column (scatter_data): Column for x-axis.
    - y_column (scatter_data): Column for y-axis.
    - new_column (dict: scatter_data (key) -> scatter_data (value)): Key is another overlapping x_column with the value being an overlapping y_column. 
                                                  It can also be used to setup a y_prime column on the right 
    - new_line (list: line_data): Array of all the line plots that need to be in there. The line data only goes for y-axis and not the y-prime axis.
    
    Returns:
    - None
    """

    # Create the first plot
    fig, ax1 = plt.subplots()
    ax1.scatter(x_column.dataframe[x_column.column], y_column.dataframe[y_column.column], label=y_column.legend_label, color=y_column.color, marker=y_column.marker, s=y_column.marker_size,alpha=y_column.opacity) # here label means legend_label, not the label of the axis. It's wierd
    ax1.set_xlabel(x_column.label) # this is the x_axis label.
    ax1.set_ylabel(y_column.label) # this is the y-axis label.
    ax1.set_xscale(x_column.scale)
    ax1.set_yscale(y_column.scale)    

    y_prime_init = 0
    if len(new_column) != 0:
        for x_column, y_column in new_column.items():
            if y_column.column_type == 'y_prime_column':
                if y_prime_init == 0:
                    y_prime_init = 1
                    ax2 = ax1.twinx()
                    ax2.scatter(x_column.dataframe[x_column.column], y_column.dataframe[y_column.column], label=y_column.legend_label, color=y_column.color, marker=y_column.marker, s=y_column.marker_size, alpha=y_column.opacity)
                    ax2.set_ylabel(y_column.label)
                else:
                    ax2.scatter(x_column.dataframe[x_column.column], y_column.dataframe[y_column.column], label=y_column.legend_label, color=y_column.color, marker=y_column.marker, s=y_column.marker_size, alpha=y_column.opacity)
            else:
                ax1.scatter(x_column.dataframe[x_column.column], y_column.dataframe[y_column.column], label=y_column.legend_label, color=y_column.color, marker=y_column.marker, s=y_column.marker_size, alpha=y_column.opacity)

    if len(new_line) != 0:
        for line in new_line:
            ax1.plot(line.x_values, line.y_values, label = line.legend_label, color = line.color, alpha=line.opacity)

    # Add a legend
    ax1.legend(loc=y_column.legend_location, bbox_to_anchor=(0, 0.9))
    if y_prime_init==1:
        ax2.legend(loc=y_column.legend_location, bbox_to_anchor=(0.5, 1))    

    # create a title and grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(title)

    plt.show()

def display_dataframe(data_frame):
    """
    Display any given DataFrame.

    Parameters:
    - data_frame (pd.DataFrame): Input DataFrame.

    Returns:
    - None
    """
    
    pd.set_option('display.float_format', '{:e}'.format)
    print("Contents of DataFrame in Scientific Notation:")
    print(data_frame)
    pd.reset_option('display.float_format')

def find_cond_f(df, target_temperature):
    """
    Takes a DataFrame and an integer target_temperature.
    Finds rows where 'Control temperature (K)' is equal to target_temperature.
    Calculates the average of 'Conductivity_(Ohm)' for those rows.

    Parameters:
    - df: pandas DataFrame
    - target_temperature: int

    Returns:
    - average_conductivity: float
    """
    df['Temp_diff'] = np.abs(df['Temperature (K)'] - target_temperature)
    
    # Find the row with the minimum difference
    min_diff_row = df.loc[df['Temp_diff'].idxmin()]
    
    # Extract conductivity from the row with the minimum difference
    conductivity = min_diff_row['Conductivity (Ohm-cm)^-1']
    
    return conductivity

def shortened_csv_file(input_file_paths, num_points, output_file_path):
    """
    Shortens your csv file to the number of points specified.

    Parameters:
    - input_file_paths (str): Input file path.
    - num_points (int): number of points that are needed in the new graph.
    - output_file_path (str): output file path.

    Returns:
    - None
    """

    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(input_file_paths)

    # Create time intervals
    first_exponent = int(math.floor(math.log10(abs(df.iloc[0,0])))) # type: ignore
    last_exponent = int(math.floor(math.log10(abs(df.iloc[-1,0])))) # type: ignore
    time_intervals = []
    for i in range(first_exponent, last_exponent+1):
        time_intervals.append(10**i) 
    
    # Create a dictionary to store DataFrames for each time interval
    categorized_data = {}

    for i in range(len(time_intervals)-1):
        start_time = time_intervals[i]
        end_time = time_intervals[i+1]
        
        # Filter data within the time interval
        filtered_data = df[(df['Time_(s)'] >= start_time) & (df['Time_(s)'] < end_time)]
        
        # Store the filtered data in the dictionary
        key = f"{start_time}-{end_time} seconds"
        categorized_data[key] = filtered_data
    
    num_points_per_unit  = int(num_points/(len(time_intervals)-1))
    for key, df in categorized_data.items():
        # Check if the number of data points is greater than num_points
        if len(df) > num_points_per_unit:
            # If yes, shorten the DataFrame to num_points rows
            df = df.sample(n=num_points_per_unit, random_state=42)
            # Ensure consistency by taking the first num_points rows

        # Store the processed DataFrame back in the dictionary
        categorized_data[key] = df

    # Combine all DataFrames into one large DataFrame
    combined_df = pd.concat(categorized_data.values(), ignore_index=True)

    # Write the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file_path, index=False)


def add_conductivity (file_name):
    """
    Adds the conductivity column into the modified csv

    Parameters:
    - file_name (str): file name to add the column

    Returns:
    - None
    """
        
    df = pd.read_csv(file_name)
    df['Conductivity_(S/cm)'] = df['Current_(A)'] * 10
    df.to_csv(file_name, index=False)
    

def add_delta (file_name, temperature):
    """
    Adds the delta_T column into the modified csv

    Parameters:
    - file_name (str): file name to add the delta_T column
    
    Returns:
    - None
    """
        
    df = pd.read_csv(file_name)

    # Create variables cond_o and cond_f
    cond_o = df['Conductivity_(S/cm)'].iloc[0]
    temp = temperature
    
    cond_f = df['Conductivity_(S/cm)'].max()

    df['Delta_T'] = (df['Conductivity_(S/cm)'] - cond_f) / (cond_o - cond_f) #type:ignore
    df.to_csv(file_name, index=False)

def calculate_derivative(dataframe, x_col, y_col):
    """
    finds the derivative of one column with respect to anoher column. That derivative column is added to the dataframe.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame.
    - x_col (str): find the derivative with respect to x_col.
    - y_cal (str): find the derivative of y_col.

    returns
    - dataframe (pd.DataFrame): Data frame with the derivative column

    Raises:
    - ValueError: If the specified columns are not found in the DataFrame.
    """
    
    # Check if the columns exist in the dataframe
    if x_col not in dataframe.columns or y_col not in dataframe.columns:
        raise ValueError("Columns not found in the dataframe")

    # Extract the values of the specified columns
    x = dataframe[x_col].values
    y = dataframe[y_col].values

    # Calculate the derivative using numpy's gradient function
    derivative = np.gradient(y, x)

    # Create a new column in the dataframe with the calculated derivative
    derivative_col_name = f"der_{y_col}"
    dataframe[derivative_col_name] = derivative

    return dataframe

def absolute(dataframe, column):
    """
    finds the absolute value of a column and adds that value in the dataframe as a different column

    Parameters: 
    - dataframe (pd.DataFrame): Input DataFrame.
    - column (str): find the absolute value of all rows in column (str)
    
    returns:
    - dataframe (pd.DataFrame): Dataframe with the absolute value column. 
    
    Raises:
    - ValueError: If the specified columns are not found in the DataFrame.
    """

    # Check if the column exists in the dataframe
    if column not in dataframe.columns:
        raise ValueError("Column not found in the dataframe")

    # Apply the transformation to the specified column
    transformed_column = np.abs(dataframe[column])

    # Create a new column in the dataframe with the transformed values
    new_column_name = f"abs_{column}"
    dataframe[new_column_name] = transformed_column

    return dataframe

def ln(dataframe, column):
    """
    finds the natural log value of a column and adds that value in the dataframe as a different column

    Parameters: 
    - dataframe (pd.DataFrame): Input DataFrame.
    - column (str): find the natural log value of all rows in column (str)
    
    returns:
    - dataframe (pd.DataFrame): Dataframe with the natural log value column. 

    Raises:
    - ValueError: If the specified columns are not found in the DataFrame.    
    """

    # Check if the column exists in the dataframe
    if column not in dataframe.columns:
        raise ValueError("Column not found in the dataframe")

    # Apply the transformation to the specified column
    transformed_column = (np.log(dataframe[column]))

    # Create a new column in the dataframe with the transformed values
    new_column_name = f"ln_{column}"
    dataframe[new_column_name] = transformed_column

    return dataframe


def linear_fit(dataframe, x_column, y_column, temp):
    """
    Performs a linear fit on specified columns of a DataFrame, removes rows with NaN or infinite values, and plots the original data along with the linear fit.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame containing the data.
    - x_column (str): Name of the column representing the independent variable.
    - y_column (str): Name of the column representing the dependent variable.
    - temp (float): Temperature value used for labeling the plot.

    Returns:
    - None

    Raises:
    - ValueError: If the specified columns are not found in the DataFrame.
    """
    # Check if the columns exist in the dataframe
    if x_column not in dataframe.columns or y_column not in dataframe.columns:
        raise ValueError("Columns not found in the dataframe")

    # Remove rows with NaN or infinite values in the specified columns
    clean_data = dataframe.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column])

    # Extract the cleaned values of the specified columns
    x = clean_data[x_column].values
    y = clean_data[y_column].values

    # Use numpy's polyfit to find the coefficients of the linear fit (degree=1)
    slope, intercept = np.polyfit(x, y, 1)

    # Plot the original data and the linear fit
    plt.scatter(x, y, label=f'Data at {temp:.2f} K')
    plt.plot(x, slope * x + intercept, color='red', label=f'Fit: y = {slope:.2f} x + {intercept:.2f}')
    
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.title(f'Linear Fit: {y_column} vs {x_column}')
    plt.show()

    return slope, intercept

def berthelot_prediction(data_frame):
    """
    Finds the value of kappa and tau using the berthelot process. Also plots the values.

    parameters:
    - data_frame (pd.DataFrame): Input Data frame

    return:
    - kappa (float), tau (float): Returns the value of kappa and tau for the process
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        data_frame = data_frame.copy(deep=True)
        ln(absolute(ln(dataframe=data_frame, column='Delta_T'),column='ln_Delta_T'), column='abs_ln_Delta_T')
        ln(dataframe=data_frame, column='Time_(s)')
        slope, intercept = linear_fit(dataframe=data_frame, x_column='ln_Time_(s)',y_column='ln_abs_ln_Delta_T', temp = round(data_frame.at[0, 'Sample_Temperature_(K)']))
        kappa = slope
        tau = math.exp(intercept/(-slope))
        return kappa, tau

def dispersive_diffusion_prediction(data_frame):
    """
    Finds the value of kappa and tau using the berthelot process. Also plots the values.

    parameters:
    - data_frame (pd.DataFrame): Input Data frame

    return:
    - kappa (float), tau (float): Returns the value of kappa and tau for the process
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        data_frame = data_frame.copy(deep=True)
        data_frame['(1/Delta_-_1)'] = (1/data_frame['Delta_T'] - 1)
        absolute(dataframe=data_frame, column='(1/Delta_-_1)')
        ln(dataframe=data_frame, column='abs_(1/Delta_-_1)')
        ln(dataframe=data_frame, column='Time_(s)')
        slope, intercept = linear_fit(dataframe=data_frame, x_column='ln_Time_(s)',y_column='ln_abs_(1/Delta_-_1)', temp = round(data_frame.at[0, 'Sample_Temperature_(K)']))
        gamma = slope
        tau =  math.exp(- intercept/gamma)
        return gamma, tau

def rewrite_column_names(file):
    """
    Reads a CSV file using pandas, renames specific columns, and writes the modified DataFrame into the same CSV file.

    Parameters:
    - file (str): Path to the CSV file.

    Returns:
    - None
    """

    df = pd.read_csv(file)

    # Define the new column names
    new_column_names = [
        'Time_(s)',
        'Current_(A)',
        'Current_Std_Dev_(A)',
        'Control_Temperature_(K)',
        'Sample_Temperature_(K)'
    ]

    # Rename the columns in the DataFrame
    df.columns = new_column_names + list(df.columns[len(new_column_names):])

    # Write the modified DataFrame back to a the original CSV file
    df.to_csv(file, index=False)