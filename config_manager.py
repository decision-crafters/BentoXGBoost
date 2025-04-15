import os
import yaml
import logging
from typing import Dict, Any, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for BentoXGBoost projects.
    Handles loading and managing project configurations from a YAML file.
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.current_project = self._get_current_project()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the YAML file.
        
        Returns:
            Dictionary containing the configuration
        """
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"Configuration file {self.config_file} not found. Using default configuration.")
                return {
                    "default_project": "default",
                    "projects": {
                        "default": {
                            "description": "Default project",
                            "model_name": "cancer",
                            "data_source": "default",
                            "parameters": {
                                "max_depth": 3,
                                "eta": 0.3,
                                "max_features": 1000,
                                "positive_ratio": 0.5
                            }
                        }
                    }
                }
            
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return a default configuration
            return {
                "default_project": "default",
                "projects": {
                    "default": {
                        "description": "Default project",
                        "model_name": "cancer",
                        "data_source": "default",
                        "parameters": {
                            "max_depth": 3,
                            "eta": 0.3,
                            "max_features": 1000,
                            "positive_ratio": 0.5
                        }
                    }
                }
            }
    
    def _get_current_project(self) -> str:
        """
        Get the current project name from environment variable or default.
        
        Returns:
            Current project name
        """
        project = os.getenv("BENTO_PROJECT")
        
        if project and project in self.config["projects"]:
            logger.info(f"Using project from environment variable: {project}")
            return project
        
        default_project = self.config.get("default_project", "default")
        if default_project in self.config["projects"]:
            logger.info(f"Using default project: {default_project}")
            return default_project
        
        # If the default project doesn't exist, use the first available project
        first_project = next(iter(self.config["projects"]))
        logger.warning(f"Default project not found. Using first available project: {first_project}")
        return first_project
    
    def get_project_config(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the configuration for a specific project.
        
        Args:
            project_name: Name of the project. If None, uses the current project.
            
        Returns:
            Project configuration dictionary
        """
        if project_name is None:
            project_name = self.current_project
        
        if project_name not in self.config["projects"]:
            logger.warning(f"Project {project_name} not found. Using current project: {self.current_project}")
            project_name = self.current_project
        
        return self.config["projects"][project_name]
    
    def get_model_name(self, project_name: Optional[str] = None) -> str:
        """
        Get the model name for a specific project.
        
        Args:
            project_name: Name of the project. If None, uses the current project.
            
        Returns:
            Model name
        """
        project_config = self.get_project_config(project_name)
        return project_config.get("model_name", "cancer")
    
    def get_data_source(self, project_name: Optional[str] = None) -> str:
        """
        Get the data source for a specific project.
        
        Args:
            project_name: Name of the project. If None, uses the current project.
            
        Returns:
            Data source type ('default', 'github', 'web', or 'crawl')
        """
        project_config = self.get_project_config(project_name)
        return project_config.get("data_source", "default")
    
    def get_source_url(self, project_name: Optional[str] = None) -> Optional[str]:
        """
        Get the source URL for a specific project.
        
        Args:
            project_name: Name of the project. If None, uses the current project.
            
        Returns:
            Source URL or None if not applicable
        """
        project_config = self.get_project_config(project_name)
        return project_config.get("source_url")
    
    def get_parameters(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the model parameters for a specific project.
        
        Args:
            project_name: Name of the project. If None, uses the current project.
            
        Returns:
            Dictionary of model parameters
        """
        project_config = self.get_project_config(project_name)
        return project_config.get("parameters", {})
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """
        Get information about all available projects.
        
        Returns:
            List of project information dictionaries
        """
        projects = []
        for name, config in self.config["projects"].items():
            projects.append({
                "name": name,
                "description": config.get("description", ""),
                "model_name": config.get("model_name", ""),
                "data_source": config.get("data_source", ""),
                "is_current": name == self.current_project
            })
        return projects
    
    def set_current_project(self, project_name: str) -> bool:
        """
        Set the current project.
        
        Args:
            project_name: Name of the project to set as current
            
        Returns:
            True if successful, False otherwise
        """
        if project_name in self.config["projects"]:
            self.current_project = project_name
            logger.info(f"Current project set to: {project_name}")
            return True
        else:
            logger.warning(f"Project {project_name} not found. Current project unchanged: {self.current_project}")
            return False
    
    def save_config(self) -> bool:
        """
        Save the current configuration to the YAML file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def add_project(self, project_name: str, project_config: Dict[str, Any]) -> bool:
        """
        Add a new project to the configuration.
        
        Args:
            project_name: Name of the new project
            project_config: Configuration for the new project
            
        Returns:
            True if successful, False otherwise
        """
        if project_name in self.config["projects"]:
            logger.warning(f"Project {project_name} already exists. Use update_project to modify it.")
            return False
        
        self.config["projects"][project_name] = project_config
        logger.info(f"Added new project: {project_name}")
        return self.save_config()
    
    def update_project(self, project_name: str, project_config: Dict[str, Any]) -> bool:
        """
        Update an existing project in the configuration.
        
        Args:
            project_name: Name of the project to update
            project_config: New configuration for the project
            
        Returns:
            True if successful, False otherwise
        """
        if project_name not in self.config["projects"]:
            logger.warning(f"Project {project_name} not found. Use add_project to create it.")
            return False
        
        self.config["projects"][project_name] = project_config
        logger.info(f"Updated project: {project_name}")
        return self.save_config()
    
    def remove_project(self, project_name: str) -> bool:
        """
        Remove a project from the configuration.
        
        Args:
            project_name: Name of the project to remove
            
        Returns:
            True if successful, False otherwise
        """
        if project_name not in self.config["projects"]:
            logger.warning(f"Project {project_name} not found.")
            return False
        
        if len(self.config["projects"]) <= 1:
            logger.warning("Cannot remove the last project.")
            return False
        
        del self.config["projects"][project_name]
        logger.info(f"Removed project: {project_name}")
        
        # If the current project was removed, set a new current project
        if project_name == self.current_project:
            self.current_project = next(iter(self.config["projects"]))
            logger.info(f"Current project set to: {self.current_project}")
        
        return self.save_config()

# Create a singleton instance
config_manager = ConfigManager()
