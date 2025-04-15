#!/usr/bin/env python3
"""
Example script demonstrating how to interact with the BentoML service API.
This script shows how to:
1. List available models
2. Get information about the current model
3. Switch to a different model
4. Train a new model
5. Make predictions
"""

import requests
import json
import time
import sys
import os

# Service URL
SERVICE_URL = "http://localhost:3000"

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def pretty_print_json(data):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2))

def list_models():
    """List all available models."""
    print_section("Listing Available Models")
    response = requests.get(f"{SERVICE_URL}/models")
    if response.status_code == 200:
        models = response.json()
        pretty_print_json(models)
        return models
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return []

def get_current_model():
    """Get information about the current model."""
    print_section("Current Model Information")
    response = requests.get(f"{SERVICE_URL}/current_model")
    if response.status_code == 200:
        model_info = response.json()
        pretty_print_json(model_info)
        return model_info
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}

def switch_model(model_tag):
    """Switch to a different model."""
    print_section(f"Switching to Model: {model_tag}")
    response = requests.post(
        f"{SERVICE_URL}/switch_model",
        json={"model_tag": model_tag}
    )
    if response.status_code == 200:
        result = response.json()
        pretty_print_json(result)
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}

def train_model(model_name, data_source, source_url=None, max_pages=10, 
                max_features=1000, positive_ratio=0.5, max_depth=3, eta=0.3):
    """Train a new model."""
    print_section(f"Training New Model: {model_name}")
    
    payload = {
        "model_name": model_name,
        "data_source": data_source,
        "max_features": max_features,
        "positive_ratio": positive_ratio,
        "max_depth": max_depth,
        "eta": eta
    }
    
    if source_url:
        payload["source_url"] = source_url
    
    if data_source == "crawl":
        payload["max_pages"] = max_pages
    
    print("Training parameters:")
    pretty_print_json(payload)
    
    print("\nStarting training (this may take a while)...")
    start_time = time.time()
    
    response = requests.post(
        f"{SERVICE_URL}/train_model",
        json=payload
    )
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print("Training result:")
        pretty_print_json(result)
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}

def make_prediction(data):
    """Make a prediction using the current model."""
    print_section("Making Prediction")
    response = requests.post(
        f"{SERVICE_URL}/predict",
        json={"data": data}
    )
    if response.status_code == 200:
        prediction = response.json()
        print("Prediction result:")
        pretty_print_json(prediction)
        return prediction
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}

def main():
    """Main function demonstrating the service API."""
    # Check if the service is running
    try:
        requests.get(SERVICE_URL)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the service at {SERVICE_URL}")
        print("Make sure the service is running with 'make serve' or 'bentoml serve .'")
        sys.exit(1)
    
    # List available models
    models = list_models()
    
    # Get current model information
    current_model = get_current_model()
    
    # If there are multiple models, switch to a different one
    if len(models) > 1:
        # Find a model that's different from the current one
        for model in models:
            if model["tag"] != current_model["model_tag"]:
                switch_model(model["tag"])
                break
    
    # Train a new model with the default dataset
    train_model(
        model_name="example_model",
        data_source="default"
    )
    
    # Make a prediction using the breast cancer dataset sample
    sample_data = [
        [1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
         4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
         1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
         1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
         1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]
    ]
    make_prediction(sample_data)
    
    # List models again to see the newly trained model
    list_models()

if __name__ == "__main__":
    main()
