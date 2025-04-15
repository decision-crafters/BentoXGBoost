# Train Model Prompt

I'm working with the BentoXGBoost project and I want to train a model using a custom data source. I have the following requirements:

- I want to train an XGBoost model using [DATA_SOURCE_TYPE] (one of: GitHub repository, web page, crawled website)
- The source URL is: [SOURCE_URL]
- I want to save the model with the name: [MODEL_NAME]
- I want to use the following parameters:
  - max_depth: [MAX_DEPTH]
  - eta: [ETA]
  - max_features: [MAX_FEATURES]
  - positive_ratio: [POSITIVE_RATIO]
  - max_pages: [MAX_PAGES] (if using a crawled website)

Please help me:

1. Show me the command to train the model directly using `save_model.py`
2. Show me how to create a project configuration for this in `config.yaml`
3. Show me how to train the model using the project configuration
4. Explain how to verify that the model was trained successfully

I'm familiar with the project structure, which includes:
- `data_loader.py` for downloading and processing data from different sources
- `data_processor.py` for processing the data for XGBoost
- `save_model.py` for training models
- `config.yaml` for project configurations
- `config_manager.py` for managing project configurations

Thank you!
