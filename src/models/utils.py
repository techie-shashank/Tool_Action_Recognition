
def get_model_class(model_name):
    """
    Get the model class based on the model name.

    Args:
        model_name (str): Name of the model.

    Returns:
        class: Model class.
    """
    if model_name.lower() == "tcn":
        from src.models.tcn import TCNClassifier
        return TCNClassifier
    elif model_name.lower() == "fcn":
        from src.models.fcn import FCNClassifier
        return FCNClassifier
    elif model_name.lower() == "lstm":
        from src.models.lstm import LSTMClassifier
        return LSTMClassifier
    else:
        raise ValueError(f"Unknown model name: {model_name}")