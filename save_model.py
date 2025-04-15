import typing as t
import argparse
import os
import logging

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch

import bentoml

# Import our custom modules
from data_loader import DataLoader
from data_processor import DataProcessor
from config_manager import config_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from sklearn.utils import Bunch
    from bentoml._internal import external_typing as ext

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and save an XGBoost model with BentoML')

    # Project options
    project_group = parser.add_argument_group('Project Options')
    project_group.add_argument('--project', type=str,
                             help='Project name from config.yaml to use for training')
    project_group.add_argument('--list-projects', action='store_true',
                             help='List all available projects from config.yaml')

    # Data source options
    data_group = parser.add_argument_group('Data Sources')
    data_group.add_argument('--use-default-dataset', action='store_true',
                          help='Use the default breast cancer dataset')
    data_group.add_argument('--github-repo-url', type=str,
                          help='URL to a GitHub repository ZIP file to download and use for training')
    data_group.add_argument('--web-url', type=str,
                          help='URL to a web page to download, convert to markdown, and use for training')
    data_group.add_argument('--crawl-url', type=str,
                          help='URL to crawl using firecrawl and use the content for training')
    data_group.add_argument('--max-pages', type=int, default=10,
                          help='Maximum number of pages to crawl when using --crawl-url')

    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--max-depth', type=int, default=3,
                           help='Maximum depth of the XGBoost trees')
    model_group.add_argument('--eta', type=float, default=0.3,
                           help='Learning rate for XGBoost')
    model_group.add_argument('--model-name', type=str, default='cancer',
                           help='Name to save the model as in BentoML')

    # Text processing parameters
    text_group = parser.add_argument_group('Text Processing Parameters')
    text_group.add_argument('--max-features', type=int, default=1000,
                          help='Maximum number of features to extract from text')
    text_group.add_argument('--positive-ratio', type=float, default=0.5,
                          help='Ratio of positive samples for binary labels')

    args = parser.parse_args()

    # If --list-projects is specified, list all projects and exit
    if args.list_projects:
        projects = config_manager.get_all_projects()
        print("\nAvailable projects:")
        print("-" * 80)
        for project in projects:
            current = "(current)" if project["is_current"] else ""
            print(f"{project['name']} {current}")
            print(f"  Description: {project['description']}")
            print(f"  Model: {project['model_name']}")
            print(f"  Data source: {project['data_source']}")
            print("-" * 80)
        exit(0)

    # If a project is specified, use its configuration
    if args.project:
        if config_manager.set_current_project(args.project):
            project_config = config_manager.get_project_config()

            # Set model name from project config if not explicitly provided
            if not parser.get_default('model_name') == args.model_name:
                args.model_name = project_config.get('model_name', 'cancer')

            # Set data source from project config
            data_source = project_config.get('data_source', 'default')
            if data_source == 'default':
                args.use_default_dataset = True
            elif data_source == 'github':
                args.github_repo_url = project_config.get('source_url')
            elif data_source == 'web':
                args.web_url = project_config.get('source_url')
            elif data_source == 'crawl':
                args.crawl_url = project_config.get('source_url')
                args.max_pages = project_config.get('parameters', {}).get('max_pages', 10)

            # Set model parameters from project config
            parameters = project_config.get('parameters', {})
            if 'max_depth' in parameters:
                args.max_depth = parameters['max_depth']
            if 'eta' in parameters:
                args.eta = parameters['eta']
            if 'max_features' in parameters:
                args.max_features = parameters['max_features']
            if 'positive_ratio' in parameters:
                args.positive_ratio = parameters['positive_ratio']

            logger.info(f"Using configuration from project: {args.project}")

    # Default to using the default dataset if no other data source is specified
    if not (args.github_repo_url or args.web_url or args.crawl_url):
        args.use_default_dataset = True

    return args

def load_default_dataset():
    """Load the default breast cancer dataset."""
    logger.info("Loading default breast cancer dataset")
    cancer: Bunch = t.cast("Bunch", load_breast_cancer())
    cancer_data = t.cast("ext.NpNDArray", cancer.data)
    cancer_target = t.cast("ext.NpNDArray", cancer.target)
    dt = xgb.DMatrix(cancer_data, label=cancer_target)
    return dt

def load_github_repo_data(url, max_features, positive_ratio):
    """Load data from a GitHub repository."""
    logger.info(f"Loading data from GitHub repository: {url}")
    data_loader = DataLoader()
    data_processor = DataProcessor(max_features=max_features)

    # Download and extract the repository
    extract_dir = data_loader.download_and_extract_zip(url)

    # Process the extracted files
    dt = data_processor.process_directory_for_training(
        directory=extract_dir,
        file_pattern="**/*.*",  # Process all files
        positive_ratio=positive_ratio
    )

    return dt

def load_web_url_data(url, max_features, positive_ratio):
    """Load data from a web URL."""
    logger.info(f"Loading data from web URL: {url}")
    data_loader = DataLoader()
    data_processor = DataProcessor(max_features=max_features)

    # Fetch and convert the web page to markdown
    markdown_text = data_loader.fetch_and_convert_to_markdown(url)

    # Process the markdown text
    dt = data_processor.process_markdown_list_for_training(
        markdown_texts=[markdown_text],
        positive_ratio=positive_ratio
    )

    return dt

def load_crawled_data(url, max_pages, max_features, positive_ratio):
    """Load data by crawling a website."""
    logger.info(f"Loading data by crawling website: {url} (max {max_pages} pages)")
    data_loader = DataLoader()
    data_processor = DataProcessor(max_features=max_features)

    # Crawl the website
    page_contents = data_loader.crawl_website(url, max_pages=max_pages)

    # Convert HTML to markdown
    markdown_texts = [data_loader.convert_html_to_markdown(content) for content in page_contents]

    # Process the markdown texts
    dt = data_processor.process_markdown_list_for_training(
        markdown_texts=markdown_texts,
        positive_ratio=positive_ratio
    )

    return dt

def main():
    """Main function to train and save the model."""
    args = parse_arguments()

    # Load data based on the specified source
    if args.use_default_dataset:
        dt = load_default_dataset()
    elif args.github_repo_url:
        dt = load_github_repo_data(args.github_repo_url, args.max_features, args.positive_ratio)
    elif args.web_url:
        dt = load_web_url_data(args.web_url, args.max_features, args.positive_ratio)
    elif args.crawl_url:
        dt = load_crawled_data(args.crawl_url, args.max_pages, args.max_features, args.positive_ratio)
    else:
        raise ValueError("No data source specified")

    # Specify model parameters
    param = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "objective": "multi:softprob",
        "num_class": 2
    }

    # Train the model
    logger.info("Training XGBoost model")
    model = xgb.train(param, dt)

    # Specify the model name and save it
    logger.info(f"Saving model as '{args.model_name}'")
    bentoml.xgboost.save_model(args.model_name, model)
    logger.info(f"Model saved successfully as '{args.model_name}'")

if __name__ == "__main__":
    main()
