import os
import json
import logging
import subprocess
from typing import Dict, List, Optional, Union, Any

import numpy as np
import xgboost as xgb

import bentoml

# Import our configuration manager
from config_manager import config_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the Python packages needed for the service
demo_image = bentoml.images.PythonImage(python_version="3.11") \
    .python_packages(
        "xgboost",
        "scikit-learn",
        "requests",
        "html2text",
        "firecrawl"
    )


@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 30},  # Increased timeout for model retraining
)
class ModelService:
    def __init__(self):
        # Get the project name from environment variable or use default from config
        project_name = os.getenv("BENTO_PROJECT")
        if project_name:
            config_manager.set_current_project(project_name)

        # Get model name from config or environment variable
        self.model_name = os.getenv("MODEL_NAME", config_manager.get_model_name())
        self.model_version = os.getenv("MODEL_VERSION", "latest")
        self.model_tag = f"{self.model_name}:{self.model_version}"

        # Store the current project name
        self.project_name = config_manager.current_project

        logger.info(f"Using project: {self.project_name}, model: {self.model_tag}")

        # Load the model
        self.load_model()

        # Store information about available models
        self.available_models = self.get_available_models()

    def load_model(self, model_tag: Optional[str] = None):
        """Load a model from the BentoML model store."""
        if model_tag is None:
            model_tag = self.model_tag

        logger.info(f"Loading model: {model_tag}")
        try:
            # Retrieve the model from the BentoML model store
            bento_model = bentoml.models.BentoModel(model_tag)
            self.model = bentoml.xgboost.load_model(bento_model)
            self.model_tag = model_tag

            # Configure model based on available resources
            self.configure_model_resources()

            logger.info(f"Successfully loaded model: {model_tag}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_tag}: {e}")
            return False

    def configure_model_resources(self):
        """Configure the model based on available hardware resources."""
        # Check if GPU is available
        if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
            self.model.set_param({"predictor": "gpu_predictor", "gpu_id": 0})  # type: ignore (incomplete XGBoost types)
            logger.info("Using GPU for prediction")
        else:
            # Configure CPU threads
            nthreads = os.getenv("OMP_NUM_THREADS")
            if nthreads:
                nthreads = max(int(nthreads), 1)
            else:
                nthreads = 1
            self.model.set_param({"predictor": "cpu_predictor", "nthread": nthreads})
            logger.info(f"Using CPU for prediction with {nthreads} threads")

    def get_available_models(self) -> List[Dict[str, str]]:
        """Get a list of available models in the BentoML model store."""
        try:
            # Use bentoml.models.list() to get all models
            models = bentoml.models.list()
            # Filter for XGBoost models
            xgboost_models = [model for model in models if model.module == "bentoml.xgboost"]

            # Format the results
            result = []
            for model in xgboost_models:
                result.append({
                    "name": model.tag.name,
                    "version": model.tag.version,
                    "tag": str(model.tag),
                    "creation_time": str(model.creation_time)
                })

            return result
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

    @bentoml.api
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions using the loaded model."""
        return self.model.predict(xgb.DMatrix(data))

    @bentoml.api
    def models(self) -> List[Dict[str, str]]:
        """Return information about available models."""
        # Refresh the list of available models
        self.available_models = self.get_available_models()
        return self.available_models

    @bentoml.api
    def current_model(self) -> Dict[str, str]:
        """Return information about the currently loaded model."""
        return {
            "model_tag": self.model_tag,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "project": self.project_name
        }

    @bentoml.api
    def projects(self) -> List[Dict[str, Any]]:
        """Return information about all available projects."""
        return config_manager.get_all_projects()

    @bentoml.api
    def current_project(self) -> Dict[str, Any]:
        """Return information about the current project."""
        project_config = config_manager.get_project_config()
        return {
            "name": self.project_name,
            "description": project_config.get("description", ""),
            "model_name": project_config.get("model_name", ""),
            "data_source": project_config.get("data_source", ""),
            "source_url": project_config.get("source_url", None),
            "parameters": project_config.get("parameters", {})
        }

    @bentoml.api
    def switch_project(self, project_name: str) -> Dict[str, Any]:
        """Switch to a different project."""
        if config_manager.set_current_project(project_name):
            # Update project name
            self.project_name = project_name

            # Get model name from the new project
            project_config = config_manager.get_project_config()
            model_name = project_config.get("model_name", "cancer")

            # Try to load the model
            model_tag = f"{model_name}:latest"
            success = self.load_model(model_tag)

            if success:
                # Update model information
                self.model_name = model_name
                self.model_version = "latest"
                self.model_tag = model_tag

                return {
                    "success": True,
                    "message": f"Successfully switched to project: {project_name}",
                    "project": project_name,
                    "model_tag": self.model_tag
                }
            else:
                return {
                    "success": False,
                    "message": f"Switched to project {project_name}, but failed to load model {model_tag}",
                    "project": project_name,
                    "model_tag": self.model_tag
                }
        else:
            return {
                "success": False,
                "message": f"Failed to switch to project: {project_name}",
                "project": self.project_name,
                "model_tag": self.model_tag
            }

    @bentoml.api
    def switch_model(self, model_tag: str) -> Dict[str, Any]:
        """Switch to a different model."""
        success = self.load_model(model_tag)
        if success:
            # Update model name and version from the tag
            parts = model_tag.split(":")
            if len(parts) == 2:
                self.model_name, self.model_version = parts
            else:
                self.model_name = model_tag
                self.model_version = "latest"

            return {
                "success": True,
                "message": f"Successfully switched to model: {model_tag}",
                "model_tag": self.model_tag
            }
        else:
            return {
                "success": False,
                "message": f"Failed to switch to model: {model_tag}",
                "model_tag": self.model_tag  # Return the current model tag
            }

    @bentoml.api
    def train_model(self,
                   model_name: Optional[str] = None,
                   project_name: Optional[str] = None,
                   data_source: Optional[str] = None,
                   source_url: Optional[str] = None,
                   max_pages: Optional[int] = None,
                   max_features: Optional[int] = None,
                   positive_ratio: Optional[float] = None,
                   max_depth: Optional[int] = None,
                   eta: Optional[float] = None,
                   save_to_config: bool = False) -> Dict[str, Any]:
        """Train a new model with the specified parameters.

        Args:
            model_name: Name to save the model as
            project_name: Name of the project to use for configuration
            data_source: One of 'default', 'github', 'web', or 'crawl'
            source_url: URL for GitHub repo, web page, or website to crawl (not needed for 'default')
            max_pages: Maximum number of pages to crawl (only for 'crawl')
            max_features: Maximum number of features to extract from text
            positive_ratio: Ratio of positive samples for binary labels
            max_depth: Maximum depth of the XGBoost trees
            eta: Learning rate for XGBoost
            save_to_config: Whether to save the configuration to config.yaml

        Returns:
            Dictionary with training results
        """
        # If project_name is provided, use its configuration
        if project_name:
            if not config_manager.set_current_project(project_name):
                return {
                    "success": False,
                    "message": f"Project not found: {project_name}"
                }
            self.project_name = project_name

        # Get configuration from the current project
        project_config = config_manager.get_project_config()

        # Use provided parameters or fall back to project configuration
        model_name = model_name or project_config.get("model_name", "cancer")
        data_source = data_source or project_config.get("data_source", "default")
        source_url = source_url or project_config.get("source_url")

        # Get parameters from project config
        parameters = project_config.get("parameters", {})
        max_pages = max_pages or parameters.get("max_pages", 10)
        max_features = max_features or parameters.get("max_features", 1000)
        positive_ratio = positive_ratio or parameters.get("positive_ratio", 0.5)
        max_depth = max_depth or parameters.get("max_depth", 3)
        eta = eta or parameters.get("eta", 0.3)

        # Validate inputs
        if data_source not in ['default', 'github', 'web', 'crawl']:
            return {
                "success": False,
                "message": f"Invalid data source: {data_source}. Must be one of 'default', 'github', 'web', or 'crawl'."
            }

        if data_source != 'default' and not source_url:
            return {
                "success": False,
                "message": f"URL is required for data source: {data_source}"
            }

        # Build the command
        cmd = ["python", "save_model.py", f"--model-name={model_name}"]

        if data_source == 'default':
            cmd.append("--use-default-dataset")
        elif data_source == 'github':
            cmd.append(f"--github-repo-url={source_url}")
        elif data_source == 'web':
            cmd.append(f"--web-url={source_url}")
        elif data_source == 'crawl':
            cmd.append(f"--crawl-url={source_url}")
            cmd.append(f"--max-pages={max_pages}")

        # Add other parameters
        cmd.extend([
            f"--max-features={max_features}",
            f"--positive-ratio={positive_ratio}",
            f"--max-depth={max_depth}",
            f"--eta={eta}"
        ])

        # Run the training process
        try:
            logger.info(f"Starting model training with command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                # Training succeeded, load the new model
                new_model_tag = f"{model_name}:latest"
                self.load_model(new_model_tag)

                # Update model information
                self.model_name = model_name
                self.model_version = "latest"
                self.model_tag = new_model_tag

                # Refresh the list of available models
                self.available_models = self.get_available_models()

                # Save to config if requested
                if save_to_config and project_name:
                    # Update project configuration
                    updated_config = {
                        "description": project_config.get("description", f"Project for {model_name}"),
                        "model_name": model_name,
                        "data_source": data_source,
                        "parameters": {
                            "max_depth": max_depth,
                            "eta": eta,
                            "max_features": max_features,
                            "positive_ratio": positive_ratio
                        }
                    }

                    if data_source != 'default':
                        updated_config["source_url"] = source_url

                    if data_source == 'crawl':
                        updated_config["parameters"]["max_pages"] = max_pages

                    # Save the updated configuration
                    config_manager.update_project(project_name, updated_config)

                return {
                    "success": True,
                    "message": f"Successfully trained and loaded model: {new_model_tag}",
                    "model_tag": new_model_tag,
                    "project": self.project_name,
                    "stdout": stdout,
                    "stderr": stderr
                }
            else:
                return {
                    "success": False,
                    "message": f"Model training failed with return code: {process.returncode}",
                    "stdout": stdout,
                    "stderr": stderr
                }
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return {
                "success": False,
                "message": f"Error during model training: {str(e)}"
            }
