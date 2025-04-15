# Create Custom Project Prompt

I'm working with the BentoXGBoost project and I want to create a custom project with specific parameters. I have the following requirements:

- Project name: [PROJECT_NAME]
- Description: [DESCRIPTION]
- Model name: [MODEL_NAME]
- I want to use a custom data source that is not currently supported by the project
- The custom data source is: [CUSTOM_DATA_SOURCE_DESCRIPTION]
- I want to use the following parameters:
  - max_depth: [MAX_DEPTH]
  - eta: [ETA]
  - max_features: [MAX_FEATURES]
  - positive_ratio: [POSITIVE_RATIO]
  - [CUSTOM_PARAMETER_1]: [VALUE_1]
  - [CUSTOM_PARAMETER_2]: [VALUE_2]

Please help me:

1. Extend the `data_loader.py` module to support my custom data source
2. Update the `save_model.py` script to handle the custom data source
3. Create a project configuration in `config.yaml` for my custom project
4. Show me how to train a model using my custom project
5. Show me how to serve the model using my custom project

I'm familiar with the project structure, which includes:
- `data_loader.py` for downloading and processing data from different sources
- `data_processor.py` for processing the data for XGBoost
- `save_model.py` for training models
- `service.py` for serving models
- `config.yaml` for project configurations
- `config_manager.py` for managing project configurations

Thank you!
