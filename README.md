<div align="center">
    <h1 align="center">Enhanced XGBoost Model Serving with BentoML</h1>
</div>

This is an enhanced fork of the original BentoML XGBoost example project, which demonstrates how to serve and deploy an [XGBoost](https://xgboost.readthedocs.io/en/stable/) model with BentoML. This enhanced version includes additional features such as:

- **Multiple data sources**: Train models using GitHub repositories, web pages, or crawled websites
- **Project configuration system**: Manage different projects with their own model names, data sources, and parameters
- **Enhanced API**: Switch between models, train new models, and manage projects via API
- **Makefile support**: Simplified setup and execution with Python 3.11 virtual environments
- **AI assistant prompts**: Example prompts for interacting with the project using AI coding assistants

The original BentoML example can be found [here](https://docs.bentoml.com/en/latest/examples/overview.html).

## Installation

This project is a fork of the original BentoML XGBoost example with additional features. To get started, fork this repository and then clone your fork:

```bash
# Fork the repository first at https://github.com/decision-crafters/BentoXGBoost

# Then clone your fork
git clone https://github.com/YOUR-USERNAME/BentoXGBoost.git
cd BentoXGBoost
```

Alternatively, you can clone directly from the decision-crafters repository:

```bash
git clone https://github.com/decision-crafters/BentoXGBoost.git
cd BentoXGBoost
```

### Using Make (Recommended)

This project includes a Makefile to simplify setup and execution. Make sure you have `make` installed on your system.

```bash
# Set up a Python virtual environment and install dependencies
make setup

# For macOS users, this will also attempt to install libomp via Homebrew
# which is required for XGBoost
```

### Manual Installation

Alternatively, you can set up the environment manually:

```bash
# Create a virtual environment (Python 3.11 recommended)
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Unix/macOS
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# For macOS users, you need to install OpenMP runtime for XGBoost
brew install libomp  # On macOS with Homebrew
```

## Project Configuration System

This project includes a configuration system that allows you to define and manage different projects in a `config.yaml` file. Each project can have its own model name, data source, and parameters.

### Configuration File Structure

The `config.yaml` file has the following structure:

```yaml
# Default project to use if none is specified
default_project: "cancer_classification"

# Project configurations
projects:
  # Project name
  cancer_classification:
    description: "Breast cancer classification using the default dataset"
    model_name: "cancer"
    data_source: "default"
    parameters:
      max_depth: 3
      eta: 0.3
      max_features: 1000
      positive_ratio: 0.5

  # Another project
  github_project:
    description: "Model trained on GitHub repository data"
    model_name: "github_model"
    data_source: "github"
    source_url: "https://github.com/KolbySisk/next-supabase-stripe-starter/archive/refs/heads/main.zip"
    parameters:
      max_depth: 4
      eta: 0.2
      max_features: 1500
      positive_ratio: 0.6
```

### Using Projects

You can use projects in several ways:

1. **List all projects**:
   ```bash
   make list-projects
   # or
   python save_model.py --list-projects
   ```

2. **Train a model using a project**:
   ```bash
   make run-project PROJECT=github_project
   # or
   python save_model.py --project github_project
   ```

3. **Start the service with a specific project**:
   ```bash
   make serve-project PROJECT=github_project
   # or
   BENTO_PROJECT=github_project bentoml serve .
   ```

4. **Switch projects via the API**:
   ```bash
   curl -X 'POST' \
     'http://localhost:3000/switch_project' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
     "project_name": "github_project"
   }'
   ```

## Train and save a model

This project now supports multiple data sources for training the XGBoost model:

1. Default breast cancer dataset from scikit-learn
2. GitHub repositories (downloaded as ZIP files)
3. Web pages (converted to markdown)
4. Crawled websites (using firecrawl)

You can train the model using different data sources. The examples below show both the Make commands and the direct Python commands.

### Using the default dataset

Save the model to the BentoML Model Store using the default breast cancer dataset:

```bash
# Using Make
make run-default

# Or directly with Python
python3 save_model.py --use-default-dataset
```

### Using a GitHub repository

Train the model using code from a GitHub repository:

```bash
# Using Make
make run-github

# Or directly with Python
python3 save_model.py --github-repo-url https://github.com/KolbySisk/next-supabase-stripe-starter/archive/refs/heads/main.zip --model-name github_model
```

### Using a web page

Train the model using content from a web page:

```bash
# Using Make
make run-web

# Or directly with Python
python3 save_model.py --web-url https://www.cancer.org/cancer/types/breast-cancer/about/what-is-breast-cancer.html --model-name web_model
```

### Using a crawled website

Train the model by crawling a website:

```bash
# Using Make
make run-crawl

# Or directly with Python
python3 save_model.py --crawl-url https://www.cancer.org/cancer/types/breast-cancer/ --max-pages 5 --model-name crawled_model
```

### Additional options

You can customize the model training with additional parameters:

```bash
python3 save_model.py --help
```

### Make commands

View all available Make commands:

```bash
make help
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. You can start the service using Make or directly with BentoML.

```bash
# Using Make
make serve

# Using Make with a specific model
make serve-model MODEL=github_model

# Or directly with BentoML
bentoml serve .
```

You should see output similar to:

```
2024-06-19T08:37:31+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:ModelService" listening on http://localhost:3000 (Press CTRL+C to quit)
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

### Service API Endpoints

The service provides the following API endpoints:

1. **`/predict`**: Make predictions using the currently loaded model
2. **`/models`**: List all available models in the BentoML model store
3. **`/current_model`**: Get information about the currently loaded model
4. **`/switch_model`**: Switch to a different model by specifying a model tag
5. **`/train_model`**: Train a new model with specified parameters and data source

### Using Environment Variables

You can control which model is loaded by default using environment variables:

```bash
# Load a specific model by name
MODEL_NAME=github_model bentoml serve .

# Load a specific model version
MODEL_VERSION=v1 bentoml serve .

# Load a specific model by name and version
MODEL_NAME=web_model MODEL_VERSION=v2 bentoml serve .
```

### Training Models via API

You can train new models directly through the API:

```bash
curl -X 'POST' \
  'http://localhost:3000/train_model' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_name": "github_model",
  "data_source": "github",
  "source_url": "https://github.com/KolbySisk/next-supabase-stripe-starter/archive/refs/heads/main.zip",
  "max_features": 1000,
  "positive_ratio": 0.5,
  "max_depth": 3,
  "eta": 0.3
}'
```

This will train a new model and automatically load it if training is successful.

### Example Scripts

Example scripts are provided to demonstrate how to interact with the service API and project configuration system:

```bash
# Run the service API example (make sure the service is running first)
python examples/service_examples.py

# Run the project configuration example
python examples/project_examples.py
```

The service API example demonstrates:
1. Listing available models
2. Getting information about the current model
3. Switching between models
4. Training a new model
5. Making predictions

The project configuration example demonstrates:
1. Listing all available projects
2. Getting information about the current project
3. Switching to a different project
4. Creating a new project
5. Updating an existing project
6. Training a model using a project configuration

## AI Coding Assistant Prompts

This project includes a `prompts/` folder with example prompts that you can use when interacting with AI coding assistants like CursorAI, Windsurf, or Cline. These prompts will help you get started with common tasks and understand how to work with the project effectively.

Available prompts include:

- `getting_started.md`: Get started with the project
- `setup_project.md`: Set up a new project in the configuration system
- `train_model.md`: Train a model with custom data sources
- `serve_model.md`: Serve a model and interact with the API
- `create_custom_project.md`: Create a custom project with specific parameters
- `modify_service.md`: Modify the service to add new functionality
- `debug_issues.md`: Debug common issues with the project
- `extend_data_sources.md`: Extend the project with new data sources
- `contributing.md`: Contribute to the project

To use these prompts, simply copy the content of a prompt file, paste it into your AI coding assistant's input field, modify it as needed for your specific use case, and submit it to get assistance with the task.

<details>

<summary>CURL</summary>

```bash
curl -X 'POST' \
    'http://localhost:3000/predict' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "data": [
        [1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
        4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
        1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
        1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
        1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]
      ]
    }'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.predict(
        data=[
            [1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
            4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
            1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
            1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
            1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]
        ],
    )
    print(result)
```

</details>

For detailed explanations, see [the BentoML documentation](https://docs.bentoml.com/en/latest/examples/xgboost.html).

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html).

```bash
bentoml cloud login
```

Deploy it from the project directory.

```bash
bentoml deploy .
```

Once the application is up and running, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).

## Contributing

Contributions to this enhanced BentoXGBoost project are welcome! This project is maintained by [decision-crafters](https://github.com/decision-crafters) and is a fork of the original [BentoML XGBoost example](https://github.com/bentoml/BentoXGBoost).

To contribute:

1. Fork the repository at https://github.com/decision-crafters/BentoXGBoost
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Submit a pull request

You can also use the `prompts/contributing.md` file as a template when working with AI coding assistants to help with your contribution.
