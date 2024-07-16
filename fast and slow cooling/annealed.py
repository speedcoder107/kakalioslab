import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from scipy import constants as c
import math
from IPython.display import Markdown
import annealed as an

x = np.arange(2.0, 3.4, 0.001)

class ParallelLinesError(Exception):
    def __init__(self, message="The lines are parallel and do not intersect."):
        self.message = message
        super().__init__(self.message)

def same_line_finder(d_slow, d_fast, err_lim = 0.01, forgiveness_num = 20):
    len_list_slow = len(d_slow["Temperature (K)"]) - 1
    len_list_fast = len(d_fast["Temperature (K)"]) - 1
    if len_list_fast>len_list_slow:
        len_list = len_list_slow
    else:
        len_list = len_list_fast
        
    err_cond = 0
    if d_slow['Temperature (K)'][0] - d_slow["Temperature (K)"][len_list] > 0:
            trend = "down"
            i = 0
    else:
         trend = "up"
         i = len_list
            

    while np.abs(err_cond) < err_lim:

        last_row_fast = d_fast.loc[i]
        last_temp_fast = last_row_fast['Temperature (K)']
        last_row_slow = d_slow.iloc[(d_slow['Temperature (K)'] - last_temp_fast).abs().argmin()]    

        last_temp_slow = last_row_slow['Temperature (K)']

        last_cond_fast = np.log(last_row_fast['Conductivity (Ohm-cm)^-1'])
        last_cond_slow = np.log(last_row_slow['Conductivity (Ohm-cm)^-1'])

        err_cond = abs(last_cond_fast - last_cond_slow)
        # print(err_cond)
        
        if trend == 'down':
            i+=1
        else:
            i-=1
        if np.abs(err_cond) > err_lim:
            for forgiveness_factor in range(forgiveness_num):
                if trend == 'down':
                    forgive_row_fast = d_fast.loc[i + forgiveness_factor]
                else:
                    forgive_row_fast = d_fast.loc[i - forgiveness_factor]

                forgive_temp_fast = forgive_row_fast['Temperature (K)']
                forgive_row_slow = d_slow.iloc[(d_slow['Temperature (K)'] - forgive_temp_fast).abs().argmin()]    

                forgive_temp_slow = forgive_row_slow['Temperature (K)']

                forgive_cond_fast = np.log(forgive_row_fast['Conductivity (Ohm-cm)^-1'])
                forgive_cond_slow = np.log(forgive_row_slow['Conductivity (Ohm-cm)^-1'])

                forgive_err_cond = forgive_cond_fast - forgive_cond_slow

                if abs(forgive_err_cond) < err_lim:
                    err_cond = forgive_err_cond
                    break

    return (last_row_slow.name, last_row_fast.name)

def intersection(slope1, intercept1, slope2, intercept2):
    # Check if the slopes are equal (parallel lines)
    if slope1 == slope2:
        raise ParallelLinesError()

    # Calculate the x coordinate of the intersection point
    x = (intercept2 - intercept1) / (slope1 - slope2)
    
    # Calculate the y coordinate of the intersection point
    y = np.exp(slope1 * x + intercept1)

    return (x, y)

def contains_up_or_down(input_string):
    input_string = input_string.lower()  # Convert to lowercase to make the search case-insensitive
    if 'up' in input_string:
        return 'up'
    elif 'down' in input_string:
        return 'down'
    else:
        return 'none'



def game(slow_file_name, fast_file_name, temp):
    d_slow = pd.read_csv(slow_file_name)
    d_fast = pd.read_csv(fast_file_name)

    d_slow.columns = d_slow.columns.str.strip()
    cond_slow = d_slow['Conductivity (Ohm-cm)^-1'] 
    thousand_over_temp_slow = 1000/d_slow['Temperature (K)']

    d_fast.columns = d_fast.columns.str.strip()
    cond_fast = d_fast['Conductivity (Ohm-cm)^-1'] 
    thousand_over_temp_fast = 1000/d_fast['Temperature (K)']

    R1_end_slow, R1_end_fast = same_line_finder(d_slow, d_fast)

    eq_temp = 1000/thousand_over_temp_slow[R1_end_slow-1]

    # R1

    R1_cond_slow = cond_slow[:R1_end_slow] 
    R1_thousand_over_temp_slow = thousand_over_temp_slow[:R1_end_slow]
    R1_coefficients_slow = np.polyfit(R1_thousand_over_temp_slow, np.log(R1_cond_slow), 1)
    R1_slope_slow = R1_coefficients_slow[0]  # Slope
    R1_intercept_slow = R1_coefficients_slow[1]  # Intercept
    R1_fit_line_slow = np.exp(R1_slope_slow * x + R1_intercept_slow)

    # R1_cond_fast = cond_fast[:R1_end_fast] 
    # R1_thousand_over_temp_fast = thousand_over_temp_fast[:R1_end_fast]
    # R1_coefficients_fast = np.polyfit(R1_thousand_over_temp_fast, np.log(R1_cond_fast), 1)
    # R1_slope_fast = R1_coefficients_fast[0]  # Slope
    # R1_intercept_fast = R1_coefficients_fast[1]  # Intercept
    # R1_fit_line_fast = np.exp(R1_slope_fast * x + R1_intercept_fast)

    # generally the slope is very similar for both slow and fast but for R1 region, I am using slow slope as default.
    R1_activation = - R1_slope_slow * 1000 * c.physical_constants['Boltzmann constant in eV/K'][0]
    R1_cond_o = math.exp(R1_intercept_slow)



    # R2
    R2_begin_slow = d_slow.iloc[(d_slow['Temperature (K)'] - 410).abs().argmin()].name
    R2_begin_fast = d_fast.iloc[(d_fast['Temperature (K)'] - 410).abs().argmin()].name

    R2_cond_fast = cond_fast[R2_begin_fast:] 
    R2_thousand_over_temp_fast = thousand_over_temp_fast[R2_begin_fast:]
    R2_coefficients_fast = np.polyfit(R2_thousand_over_temp_fast, np.log(R2_cond_fast), 1)
    R2_slope_fast = R2_coefficients_fast[0]  
    R2_intercept_fast = R2_coefficients_fast[1] 
    R2_fit_line_fast = np.exp(R2_slope_fast * x + R2_intercept_fast)
    R2_activation_fast = - R2_slope_fast * 1000 * c.physical_constants['Boltzmann constant in eV/K'][0]
    R2_cond_o_fast = np.exp(R2_intercept_fast)

    R2_cond_slow = cond_slow[R2_begin_slow:] 
    R2_thousand_over_temp_slow = thousand_over_temp_slow[R2_begin_slow:]
    R2_coefficients_slow = np.polyfit(R2_thousand_over_temp_slow, np.log(R2_cond_slow), 1)
    R2_slope_slow = R2_coefficients_slow[0]  
    R2_intercept_slow = R2_coefficients_slow[1] 
    R2_fit_line_slow = np.exp(R2_slope_slow * x + R2_intercept_slow)
    R2_activation_slow = - R2_slope_slow * 1000 * c.physical_constants['Boltzmann constant in eV/K'][0]
    R2_cond_o_slow = np.exp(R2_intercept_slow)

    pt_slow_x, pt_slow_y = an.intersection(R1_slope_slow, R1_intercept_slow, R2_slope_slow, R2_intercept_slow)
    pt_fast_x, pt_fast_y = an.intersection(R1_slope_slow, R1_intercept_slow, R2_slope_fast, R2_intercept_fast)
    fictive_temp_slow = 1000/pt_slow_x
    fictive_temp_fast = 1000/pt_fast_x

    answer_dict = {}
    answer_dict["R1_activation"] = R1_activation
    answer_dict["R1_cond_o"] = R1_cond_o
    answer_dict["R2_activation_fast"] = R2_activation_fast
    answer_dict["R2_cond_o_fast"] = R2_cond_o_fast
    answer_dict["R2_activation_slow"] = R2_activation_slow
    answer_dict["R2_cond_o_slow"] = R2_cond_o_slow
    answer_dict["R2_cond_o_slow"] = R2_cond_o_slow
    answer_dict["thousand_over_temp_slow"] = thousand_over_temp_slow
    answer_dict["thousand_over_temp_fast"] = thousand_over_temp_fast
    answer_dict["cond_slow"] = cond_slow
    answer_dict["cond_fast"] = cond_fast
    answer_dict["R1_end_slow"] = R1_end_slow
    answer_dict["R1_end_fast"] = R1_end_fast
    answer_dict["cond_slow"] = cond_slow
    answer_dict["cond_fast"] = cond_fast
    answer_dict["R1_fit_line_slow"] = R1_fit_line_slow
    # answer_dict["R1_fit_line_fast"] = R1_fit_line_fast
    answer_dict["R2_fit_line_slow"] = R2_fit_line_slow
    answer_dict["R2_fit_line_fast"] = R2_fit_line_fast
    answer_dict["R1_intercept_slow"] = R1_intercept_slow
    # answer_dict["R1_intercept_fast"] = R1_intercept_fast
    answer_dict["R2_intercept_slow"] = R2_intercept_slow
    answer_dict["R2_intercept_fast"] = R2_intercept_fast
    answer_dict["fictive_temp_slow"] = fictive_temp_slow
    answer_dict["fictive_temp_fast"] = fictive_temp_fast
    answer_dict["pt_slow_x"] = pt_slow_x
    answer_dict["pt_fast_x"] = pt_fast_x
    return answer_dict
    
    

