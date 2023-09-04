"""
io.py part of regressionpipeliner

Copyright (C) <2023>  Giuseppe Marco Randazzo <gmrandazzo@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import logging
import json
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def elaborate_results(y_true,
                      regress_results : dict,
                      emissions_results : dict = None):
    """
    Elaborate the results in terms of precision, recall, and AUC.

    This function takes true values `y_true`, regression results `regress_results`
    genearted with best_regression, and optionally emissions results `emissions_results`.
    It calculates and returns various regression evaluation metrics for each result in
    `regress_results`.

    Parameters:
        y_true (MatrixLike): The true values.
        regress_results (dict): A dictionary of regression results
            where keys are identifiers and values are the
            predicted values.
        emissions_results (dict, optional): A dictionary of emission results
            where keys are identifiers and values are
            emission values. This parameter is optional.

    Returns:
        dict: A dictionary containing evaluation metrics for each result in
            `regress_results`. The metrics include:
            - 'MSE' (Mean Squared Error)
            - 'MAE' (Mean Absolute Error)
            - 'R2' (R-squared)

            If `emissions_results` is provided, it also includes:
            - 'emission' (Emission value)

    Example:
        >>> y_true = [1.0, 2.0, 3.0]
        >>> regress_results = {'model1': [1.1, 2.2, 2.9], 'model2': [0.9, 1.8, 3.1]}
        >>> emissions_results = {'model1': 0.5, 'model2': 0.6}
        >>> elaborate_results(y_true, regress_results, emissions_results)
        {'model1': {'MSE': 0.005, 'MAE': 0.05, 'R2': 0.98, 'emission': 0.5},
         'model2': {'MSE': 0.01, 'MAE': 0.07, 'R2': 0.95, 'emission': 0.6}}
    """
    logging.info(" * Elaborate the results")
    res = {}
    for key in regress_results.keys():
        y_score = regress_results[key]
        res[key] = {}
        res[key]["MSE"] = mean_squared_error(y_true, y_score)
        res[key]["MAE"] = mean_absolute_error(y_true, y_score)
        res[key]["R2"] = r2_score(y_true, y_score)
        if emissions_results is not None:
            res[key]["emission"] = emissions_results[key]
    return res


def write_results(res : dict,
                  out_name : str) -> None:
    """
    Write results as an image with 4 subplots and a JSON file.

    This function takes a dictionary `res` containing evaluation results
    and an output file name `out_name`. It creates a 2x2 subplot of scatter 
    and bar plots to visualize the results and saves them as an image in PNG format.
    It also writes the results to a JSON file.

    Parameters:
        res (dict): A dictionary containing evaluation results. It should have the following format:
            {
                'method1': {'emission': value1, 'R2': value2, 'MSE': value3, 'MAE': value4},
                'method2': {'emission': value5, 'R2': value6, 'MSE': value7, 'MAE': value8},
                ...
            }
        out_name (str): The base name for the output image and JSON file.
            The image will be saved as 'out_name.png',
            and the JSON file will be saved as 'out_name.json'.

    Returns:
        None

    Example:
        >>> results = {
        ...     'Method1': {'emission': 0.5, 'R2': 0.85, 'MSE': 0.02, 'MAE': 0.1},
        ...     'Method2': {'emission': 0.6, 'R2': 0.92, 'MSE': 0.01, 'MAE': 0.08},
        ... }
        >>> write_results(results, 'output_results')
        # This will save 'output_results.png' and 'output_results.json' files.
    """
    _, axs = plt.subplots(2, 2, sharex=False, sharey=False)
    x_emission = []
    y_r2 = []
    y_mse = []
    y_mae = []
    method_name = []
    for key in res.keys():
        method_name.append(key)
        x_emission.append(res[key]["emission"])
        y_r2.append(res[key]["R2"])
        y_mse.append(res[key]["MSE"])
        y_mae.append(res[key]["MAE"])

    axs[0, 0].scatter(x_emission, y_r2, s=80, c="blue", marker="o")
    axs[0, 0].set_ylabel("R2")
    axs[0, 0].set_xlabel("Emission")

    size = 6
    for i, txt in enumerate(method_name):
        axs[0, 0].annotate(txt, (x_emission[i], y_r2[i]), fontsize=size)

    axs[0, 1].scatter(x_emission, y_mse, s=80, c="black", marker="o")
    axs[0, 1].set_ylabel("MSE")
    axs[0, 1].set_xlabel("Emission")

    for i, txt in enumerate(method_name):
        axs[0, 1].annotate(txt, (x_emission[i], y_mse[i]), fontsize=size)

    axs[1, 0].scatter(x_emission, y_mae, s=80, c="green", marker="o")
    axs[1, 0].set_ylabel("MAE")
    axs[1, 0].set_xlabel("Emission")

    for i, txt in enumerate(method_name):
        axs[1, 0].annotate(txt, (x_emission[i], y_mae[i]), fontsize=size)

    axs[1, 1].barh(method_name, y_r2)
    axs[1, 1].set_ylabel("ML Method")
    axs[1, 1].set_xlabel("R2")

    plt.tight_layout()
    plt.savefig(f"{out_name}.png", dpi=300)
    with open(f"{out_name}.json", "w", encoding="utf-8") as write_file:
        json.dump(res, write_file, indent=4)

def write_html_barplot_output(res : dict,
                              html_out: str) -> None:
    """
    Write the result as interactive html with plotlyjs
    """
    print(">> Write the final result")
    with open(html_out, "w", encoding="utf-8") as f_html:
        f_html.write('<!DOCTYPE html>\n')
        f_html.write('<html lang="en">\n')
        f_html.write('<head>\n')
        f_html.write('  <!-- Load plotly.js into the DOM -->\n')
        f_html.write('  <script src="https://cdn.plot.ly/plotly-2.16.1.min.js"></script>\n')
        f_html.write('</head>\n')
        f_html.write('<body>\n')
        f_html.write('  <div id="myDiv"></div>\n')
        f_html.write('<script>\n')

        xrow = ""
        yrow = ""
        for key in res.keys():
            xrow += f"'{key}',"
            yrow += f'{res[key]["Avg-PR"]},'

        f_html.write('var data = [\n')
        f_html.write('  {\n')
        f_html.write(f'    x: [{xrow}],\n')
        f_html.write(f'    y: [{yrow}],\n')
        f_html.write('    type: "bar",\n')
        f_html.write('  }\n')
        f_html.write('];\n')

        f_html.write('Plotly.newPlot("myDiv", data);\n')
        f_html.write('</script>\n')
        f_html.write('</body>\n')

def write_html_variable_importance(res):
    """
    Write variable importance results
    """
    method_keys = []
    first_key = list(res.keys())[0]
    for key in res[first_key].keys():
        method_keys.append(key)


    for keym in method_keys:
        v_imp = {}
        for key in res.keys():
            v_imp[key] = res[key][keym]
        write_html_barplot_output(v_imp, f"{keym}.html")
