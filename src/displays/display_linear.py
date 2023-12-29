from IPython.display import display, HTML


def display_models(models):

    all_vars = set()
    for model in models:
        all_vars.update(model.summary_data_coefficients['Variable'])

    all_vars = sorted(list(all_vars))

    count = 1
    for model in models:
        model.name = f'Model {count}'
        count += 1

    header_html = "<div style='text-align: center; font-family: monospace;'>"

    max_width = max(len(model.name) for model in models) * 10

    header_html += "<span style='display: inline-block; width: 100px;'>Variable</span>"

    for model in models:
        header_html += f"<span style='display: inline-block; width: {max_width}px;'>{model.name}</span>"

    header_html += "</div>"

    total_width = 100 + len(models) * max_width
    dotted_lines = f"<div style='text-align: center; font-family: monospace;'><span style='display: inline-block; width: {total_width}px; border-bottom: 1px dotted;'>&nbsp;</span></div>"

    variable_rows = ""
    for var in all_vars:
        variable_row = f"<span style='display: inline-block; width: 100px;'>{var}</span>"

        for model in models:
            coeffs = model.summary_data_coefficients
            coeff_str = '-'
            std_err_str = ''
            asterisks = ''
            if var in coeffs['Variable']:
                idx = coeffs['Variable'].index(var)
                coeff = coeffs['Coefficient'][idx]
                std_err = coeffs['Std-Error'][idx]
                p_val = coeffs['P>|t|'][idx]

                if p_val < 0.01:
                    asterisks = '***'
                elif p_val < 0.05:
                    asterisks = '**'
                elif p_val < 0.1:
                    asterisks = '*'

                coeff_str = f"{coeff}{asterisks}"
                std_err_str = f"({std_err})"
            variable_row += f"<span style='display: inline-block; width: {max_width}px; text-align: center;'>{coeff_str}<br><span style='font-size: 0.8em;'>{std_err_str}</span></span>"
        variable_rows += f"<div style='text-align: center; font-family: monospace;'>{variable_row}</div>"

    # After generating variable rows, add rows for outcome variables
    outcome_rows = "<div style='text-align: center; font-family: monospace;'>"
    outcome_row = "<span style='display: inline-block; width: 100px;'>Outcome</span>"

    for model in models:
        outcome_var = model.outcome  # Get the outcome variable for each model
        outcome_row += f"<span style='display: inline-block; width: {max_width}px; text-align: center;'>{outcome_var}</span>"

    outcome_rows += f"{outcome_row}</div>"

    model_params = ['Observations', 'R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)']
    display_params = ['Observations', 'R^2', 'Adj. R^2', 'F-statistic', 'Prob (F)']
    model_params_rows = ""

    for idx, param in enumerate(model_params):
        display_param = display_params[idx]
        param_row = f"<span style='display: inline-block; width: 100px;'>{display_param}</span>"

        for model in models:
            summary_data = model.summary_data_model
            param_value = summary_data.get(param, '-')
            param_row += f"<span style='display: inline-block; width: {max_width}px; text-align: center;'>{param_value}</span>"

        model_params_rows += f"<div style='text-align: center; font-family: monospace;'>{param_row}</div>"

    asterisk_note = "<div style='text-align: center; font-family: monospace; font-size: 0.8em;'>* p < 0.1, ** p < 0.05, *** p < 0.01</div>"

    html = header_html + dotted_lines + variable_rows + dotted_lines + outcome_rows + dotted_lines + model_params_rows + dotted_lines + asterisk_note
    display(HTML(html))

