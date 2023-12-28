from IPython.display import display, HTML


def display_models(models):
    # Gather all variable names
    all_vars = set()
    for model in models:
        all_vars.update(model.summary_data_coefficients['Variable'])

    all_vars = sorted(list(all_vars))  # Sort the variable names

    count = 1
    for model in models:
        model.name = f'Model {count}'
        count += 1

    # Start building the HTML
    header_html = "<div style='text-align: center; font-family: monospace; color: white;'>"

    # Calculate maximum width for each model column
    max_width = max(len(model.name) for model in models) * 10

    # Header for variable names
    header_html += "<span style='display: inline-block; width: 100px;'>Variable</span>"


    for model in models:
        header_html += f"<span style='display: inline-block; width: {max_width}px;'>{model.name}</span>"

    header_html += "</div>"

    # Dotted line under the header
    total_width = 100 + len(models) * max_width
    dotted_lines = f"<div style='text-align: center; font-family: monospace; color: white;'><span style='display: inline-block; width: {total_width}px; border-bottom: 1px dotted;'>&nbsp;</span></div>"

    # Rows for variables and model data
    variable_rows = ""
    for var in all_vars:
        variable_row = f"<span style='display: inline-block; width: 100px;'>{var}</span>"

        # Model estimates for the variable
        for model in models:
            coeffs = model.summary_data_coefficients
            color = 'white'  # Default text color set to white
            coeff_str = '-'
            std_err_str = ''
            if var in coeffs['Variable']:
                idx = coeffs['Variable'].index(var)
                coeff = coeffs['Coefficient'][idx]
                std_err = coeffs['Std-Error'][idx]
                p_val = coeffs['P>|t|'][idx]
                color = '#D2042D' if p_val < 0.05 else 'white'  # Dark red for significant p-values
                coeff_str = f"{coeff}"
                std_err_str = f"({std_err})"
            variable_row += f"<span style='display: inline-block; width: {max_width}px; text-align: center; color: {color};'>{coeff_str}<br><span style='font-size: 0.8em;'>{std_err_str}</span></span>"
        variable_rows += f"<div style='text-align: center; font-family: monospace; color: white;'>{variable_row}</div>"

    # Add model parameter section
    model_params = ['Observations', 'R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)']
    model_params_rows = ""

    for param in model_params:
        param_row = f"<span style='display: inline-block; width: 100px;'>{param}</span>"

        for model in models:
            summary_data = model.summary_data_model
            param_value = summary_data.get(param, '-')
            param_row += f"<span style='display: inline-block; width: {max_width}px; text-align: center; color: white;'>{param_value}</span>"

        model_params_rows += f"<div style='text-align: center; font-family: monospace; color: white;'>{param_row}</div>"

    # Combine all sections
    html = header_html + dotted_lines + variable_rows + dotted_lines + model_params_rows
    display(HTML(html))

