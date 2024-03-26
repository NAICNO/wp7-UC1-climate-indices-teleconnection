import ipywidgets as widgets
import argparse

def create_dropdown(options, value, description):
    """
    Helper function to create a Dropdown widget.
    """
    return widgets.Dropdown(options=options, value=value, description=description)

def create_select_multiple(options, value, description):
    """
    Helper function to create a SelectMultiple widget.
    """
    return widgets.SelectMultiple(options=options, value=value, description=description)

def create_text_input(value, description):
    """
    Helper function to create a Text widget.
    """
    return widgets.Text(value=value, description=description)

def create_int_slider(value, min_val, max_val, step, description):
    """
    Helper function to create an IntSlider widget.
    """
    return widgets.IntSlider(value=value, min=min_val, max=max_val, step=step, description=description)

def create_int_input(value, description):
    """
    Helper function to create an IntText widget.
    """
    return widgets.IntText(value=value, description=description)

def create_float_slider(value, min_val, max_val, step, description):
    """
    Helper function to create a FloatSlider widget.
    """
    return widgets.FloatSlider(value=value, min=min_val, max=max_val, step=step, description=description)

def create_checkbox(value, description):
    """
    Helper function to create a Checkbox widget.
    """
    return widgets.Checkbox(value=value, description=description)

def selector_func(multi_select, options, value, description):
    if multi_select:
        return create_select_multiple(options, value, description)
    return create_dropdown(options, value[0], description)

def build_widgets(target_features, data_files, modelnames, multi_select=False):
    """
    Generic function to create widgets. Allows multi-select for some widgets if required.
    
    Args:
        target_features (list): List of target features for selection.
        data_files (list): List of data files for selection.
        modelnames (list): List of model names for selection.
        multi_select (bool): If True, uses SelectMultiple instead of Dropdown for certain widgets.

    Returns:
        Tuple of widgets.
    """
    data_file_widget = selector_func(multi_select, data_files, ['dataset/noresm-f-p1000_shigh_new_jfm.csv'], 'Data File:')
    target_feature_widget = selector_func(multi_select, target_features, ['amo2'], 'Target Feature:')
    modelname_widget = selector_func(multi_select, modelnames, ['LinearRegression'], 'Model Name:')
    
    delete_features_widget = create_text_input('traDP,amo3', 'Delete Features:')
    max_allowed_features_widget = create_int_slider(6, 1, 20, 1, 'Max Features:')
    end_lag_widget = create_int_input(100, 'End Lag:')
    n_ensembles_widget = create_int_input(100, 'Ensembles:')
    main_year_start_widget = create_int_input(850, 'Year Start:')
    splitsize_widget = create_float_slider(0.6, 0.1, 0.9, 0.1, 'Split Size:')
    with_mean_feature_widget = create_checkbox(True, 'With Mean Feature')
    with_wavelet_filter_widget = create_checkbox(False, 'Wavelet Filter')
    step_lag_widget = create_int_slider(20, 1, 50, 1, 'Step Lag:')
    
    return (
        data_file_widget, target_feature_widget, delete_features_widget, modelname_widget,
        max_allowed_features_widget, end_lag_widget, n_ensembles_widget, main_year_start_widget,
        splitsize_widget, with_mean_feature_widget, with_wavelet_filter_widget, step_lag_widget
    )


def create_execution_mode_dropdown():
    """
    Creates a dropdown widget for selecting the execution mode.
    
    Returns:
        A Dropdown widget for selecting the execution mode.
    """
    execution_mode_dropdown = create_dropdown(
        ['Single Run', 'Parallel Run', 'Cluster Run', 'No Run'],
        'No Run',
        'Execution Mode:'
    )

    def handle_dropdown_change(change):
        """
        Handler for dropdown value change. Outputs mode-specific configurations for demonstration.
        """
        config_map = {
            'Single Run': 'Single Run Configuration',
            'Parallel Run': 'Parallel Run Configuration',
            'Cluster Run': 'Cluster Run Configuration',
            'No Run': 'Skip runner, go to analysis of existing results'
        }
        custom_variable = config_map.get(change.new, "No valid option selected")
        print(custom_variable + " # For demonstration purposes")  

    execution_mode_dropdown.observe(handle_dropdown_change, names='value')
    return execution_mode_dropdown


def get_args_from_widgets(data_file_widget, 
                          target_feature_widget, 
                          delete_features_widget, 
                          modelname_widget, 
                          max_allowed_features_widget,
                          end_lag_widget,
                          n_ensembles_widget,
                          main_year_start_widget,
                          splitsize_widget,
                          with_mean_feature_widget,
                          with_wavelet_filter_widget,
                          step_lag_widget
                          ):
    # Get values from widgets
    data_file = data_file_widget.value
    target_feature = target_feature_widget.value
    delete_features = delete_features_widget.value.split(',')
    modelname = modelname_widget.value
    max_allowed_features = max_allowed_features_widget.value
    end_lag = end_lag_widget.value
    n_ensembles = n_ensembles_widget.value
    main_year_start = main_year_start_widget.value
    splitsize = splitsize_widget.value
    with_mean_feature = with_mean_feature_widget.value
    with_wavelet_filter = with_wavelet_filter_widget.value
    step_lag = step_lag_widget.value


    # If model is LinearRegression, set n_ensembles to 1
    if modelname == "LinearRegression":
        n_ensembles = 1

    # Create a Namespace object
    args = argparse.Namespace(
        data_file=data_file,
        target_feature=target_feature,
        delete_features=delete_features,
        modelname=modelname,
        max_allowed_features=max_allowed_features,
        end_lag=end_lag,
        n_ensembles=n_ensembles,
        main_year_start=main_year_start,
        splitsize=splitsize,
        with_mean_feature=with_mean_feature,
        with_wavelet_filter=with_wavelet_filter,
        step_lag=step_lag
    )

    return args

# Example of using the build_widgets function for multi-select mode
# multiple_widgets = build_widgets(target_features, data_files, modelnames, multi_select=True)

# Example of using the build_widgets function for single-select mode
# single_widgets = build_widgets(target_features, data_files, modelnames, multi_select=False)