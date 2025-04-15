# Setup Project Prompt

I'm working with the BentoXGBoost project that uses a configuration system to manage different projects. I want to set up a new project in the configuration system with the following details:

- Project name: [PROJECT_NAME]
- Description: [DESCRIPTION]
- Model name: [MODEL_NAME]
- Data source: [DATA_SOURCE] (one of: default, github, web, crawl)
- Source URL: [SOURCE_URL] (if applicable)
- Parameters:
  - max_depth: [MAX_DEPTH]
  - eta: [ETA]
  - max_features: [MAX_FEATURES]
  - positive_ratio: [POSITIVE_RATIO]
  - max_pages: [MAX_PAGES] (if using crawl data source)

Please help me:

1. Create the project configuration in the `config.yaml` file
2. Show me how to train a model using this project configuration
3. Show me how to serve the model using this project configuration

I'm familiar with the project structure, which includes:
- `config.yaml` for project configurations
- `config_manager.py` for managing project configurations
- `save_model.py` for training models
- `service.py` for serving models
- Makefile with commands for working with projects

Thank you!
