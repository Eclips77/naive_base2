import os
import pickle

class ModelSaver:
    """
    Saves a trained model to a file using pickle.
    """
    @staticmethod
    def save(model, model_dir: str, model_name: str) -> str:
        """
        Save the model to the specified directory and file name.

        Args:
            model: Trained model object.
            model_dir (str): Directory to save the model.
            model_name (str): File name for the model.
        Returns:
            str: Full path to the saved model file.
        """
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        return model_path 