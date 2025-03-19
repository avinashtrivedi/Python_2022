"""
This modue is intended to assist with clarifying any data or variables that are present in the notebook.

It consists of a function that returns information based on key/value pairs. The key is the variable name, 
and the value is the explanation of that variable. This is not a comprehensive dictionary of all variables, 
but will return the primary variables.

The indicator before the variable explanation (i.e. "DF -->") refers to the type of object that is referenced.

If you edit the information in this module, be aware that Python only reads imports on a first pass, 
so you will need to restart your kernel to see any changes.

To use the function, just call explain.what_is() with no parameters.
"""

def what_is():
    """ Takes no parameters. Accepts user input and prints a response. """
    
    df_information = {
        'grouped_tables': """
DF -->  The variable ['grouped_tables'] references a dataframe consisting of
             all possible variables in the 4 datasets available in the ABS datahold.
             
             It is useful for determining which variables are in each dataset.
             """,
        'possible_merge_options': """
DF -->  The variable ['possible_merge_options'] references a dataframe consisting of
             a smaller subset of the grouped_tables dataframe. It's purpose is to assist
             with determining which variables are common in all tables.
             """,
        'common_vars': """
List -> The variable ['common_vars'] references a list variable that contains all variables
            which are present in all tables. Note: Those ending in "_group" have different
            names for the variables in some of the tables.
        """,
        'comp_sum_df': """
DF -->  The variable ['comp_sum_df'] references the 'Company Summary' gathered from the 
        US Census Bureau Annual Business Survey.
        
        --> detail: STATE LEVEL <--
        
        The technical information for this data can be viewed at:
        https://www2.census.gov/programs-surveys/abs/technical-documentation/api/API2019-company-summary-1-26-2021.pdf""",
        'bus_char_df': """
DF -->  The variable ['bus_char_df'] references the 'Characteristics of Business' gathered from the 
        US Census Bureau Annual Business Survey.
        
        --> detail: STATE LEVEL <--
        
        The technical information for this data can be viewed at:
        https://www2.census.gov/programs-surveys/abs/technical-documentation/api/ABS_API_CB-1-26-2021.pdf""",
        'bus_own_df': """
DF -->  The variable ['bus_own_df'] references the 'Characteristics of Business Owners' gathered from the 
        US Census Bureau Annual Business Survey. 
        
        --> detail: NATIONAL LEVEL <--
        
        The technical information for this data can be viewed at:
        https://www2.census.gov/programs-surveys/abs/technical-documentation/api/ABS_API_CBO-1-26-2021.pdf""",
        'bus_tech_df': """
DF -->  The variable ['bus_tech_df'] references the 'Technology Characteristics of Business' gathered from the 
        US Census Bureau Annual Business Survey.
        
        --> detail: STATE LEVEL <--
        
        The technical information for this data can be viewed at:
        https://www2.census.gov/programs-surveys/abs/technical-documentation/api/ABS_API_TCB-2-9-2021.pdf""",
        'slbo': """
DF -->  The variable ['slbo'] references the 'Characteristics of Business Owners' gathered from the 
        US Census Bureau Annual Business Survey. 
        
        --> detail: STATE LEVEL <--
        
        The technical information for this data can be viewed at:
        https://www2.census.gov/programs-surveys/abs/technical-documentation/api/ABS_API_CBO-1-26-2021.pdf""",
    }
    user_query = input('Blank Input returns all information.\nWhich dataframe are you asking about? (partial and complete matches returned) >> ')
    if user_query.isnumeric():
        raise ValueError("Numeric inputs are invalid.")
        
    requested_vars = []
    for key in df_information:
        if user_query.lower() in key.lower():
            requested_vars.append(df_information[key])
    if len(requested_vars) == 0:
        print(f'No matches found.\n\nSome of the possibilities are:\n{", ".join(_ for _ in df_information.keys())} ')
    else:
        if len(requested_vars) > 1:
            print('\nMultiple Matches Found')
        return_string = "\n" + "\n".join(requested_vars)
        print(f'{return_string}')
        
print('\n ----- HELPER MODULE IMPORTED ------')
print(
    """
    For information on the helper module:
        run 'help(explain)' in any cell. 
        
    To directly use:
        run 'explain.what_is()' with no parameters.
    """
)