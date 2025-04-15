#!/usr/bin/env python3
"""
Example script demonstrating how to use the project configuration system.
This script shows how to:
1. List all available projects
2. Get information about the current project
3. Switch to a different project
4. Create a new project
5. Update an existing project
6. Train a model using a project configuration
"""

import os
import sys
import json
from typing import Dict, Any

# Add the parent directory to the path so we can import the config_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_manager import config_manager

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def pretty_print_json(data):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2))

def list_projects():
    """List all available projects."""
    print_section("Listing All Projects")
    projects = config_manager.get_all_projects()
    for project in projects:
        current = "(current)" if project["is_current"] else ""
        print(f"{project['name']} {current}")
        print(f"  Description: {project['description']}")
        print(f"  Model: {project['model_name']}")
        print(f"  Data source: {project['data_source']}")
        print("-" * 80)
    return projects

def get_current_project():
    """Get information about the current project."""
    print_section("Current Project Information")
    project_name = config_manager.current_project
    project_config = config_manager.get_project_config()
    
    print(f"Current project: {project_name}")
    print(f"Description: {project_config.get('description', '')}")
    print(f"Model name: {project_config.get('model_name', '')}")
    print(f"Data source: {project_config.get('data_source', '')}")
    
    if 'source_url' in project_config:
        print(f"Source URL: {project_config['source_url']}")
    
    print("\nParameters:")
    parameters = project_config.get('parameters', {})
    for key, value in parameters.items():
        print(f"  {key}: {value}")
    
    return project_config

def switch_project(project_name):
    """Switch to a different project."""
    print_section(f"Switching to Project: {project_name}")
    success = config_manager.set_current_project(project_name)
    
    if success:
        print(f"Successfully switched to project: {project_name}")
        get_current_project()
    else:
        print(f"Failed to switch to project: {project_name}")
    
    return success

def create_project(project_name, project_config):
    """Create a new project."""
    print_section(f"Creating New Project: {project_name}")
    print("Project configuration:")
    pretty_print_json(project_config)
    
    success = config_manager.add_project(project_name, project_config)
    
    if success:
        print(f"Successfully created project: {project_name}")
    else:
        print(f"Failed to create project: {project_name}")
    
    return success

def update_project(project_name, project_config):
    """Update an existing project."""
    print_section(f"Updating Project: {project_name}")
    print("New project configuration:")
    pretty_print_json(project_config)
    
    success = config_manager.update_project(project_name, project_config)
    
    if success:
        print(f"Successfully updated project: {project_name}")
    else:
        print(f"Failed to update project: {project_name}")
    
    return success

def train_model_with_project(project_name):
    """Train a model using a project configuration."""
    print_section(f"Training Model with Project: {project_name}")
    
    # Switch to the project
    if not config_manager.set_current_project(project_name):
        print(f"Failed to switch to project: {project_name}")
        return False
    
    # Get the project configuration
    project_config = config_manager.get_project_config()
    
    # Print the configuration that will be used for training
    print("Training with the following configuration:")
    pretty_print_json(project_config)
    
    # In a real scenario, you would call the save_model.py script with the project name
    print(f"\nTo train the model, run: python save_model.py --project {project_name}")
    
    return True

def main():
    """Main function demonstrating the project configuration system."""
    # List all available projects
    projects = list_projects()
    
    # Get current project information
    current_project = get_current_project()
    
    # If there are multiple projects, switch to a different one
    if len(projects) > 1:
        # Find a project that's different from the current one
        for project in projects:
            if project["name"] != config_manager.current_project:
                switch_project(project["name"])
                break
    
    # Create a new example project
    example_project_config = {
        "description": "Example project created by project_examples.py",
        "model_name": "example_model",
        "data_source": "default",
        "parameters": {
            "max_depth": 4,
            "eta": 0.2,
            "max_features": 1200,
            "positive_ratio": 0.6
        }
    }
    
    create_project("example_project", example_project_config)
    
    # Update the example project
    updated_project_config = example_project_config.copy()
    updated_project_config["description"] = "Updated example project"
    updated_project_config["parameters"]["max_depth"] = 5
    
    update_project("example_project", updated_project_config)
    
    # Train a model using the example project
    train_model_with_project("example_project")
    
    # Switch back to the original project
    switch_project(config_manager.config.get("default_project", "cancer_classification"))

if __name__ == "__main__":
    main()
