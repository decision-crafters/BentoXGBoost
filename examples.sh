#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Example 1: Train with the default dataset
echo "Example 1: Training with the default dataset"
python save_model.py --use-default-dataset --model-name cancer_default

# Example 2: Train with a GitHub repository
echo "Example 2: Training with a GitHub repository"
python save_model.py --github-repo-url https://github.com/KolbySisk/next-supabase-stripe-starter/archive/refs/heads/main.zip --model-name cancer_github

# Example 3: Train with a web URL
echo "Example 3: Training with a web URL"
python save_model.py --web-url https://www.cancer.org/cancer/types/breast-cancer/about/what-is-breast-cancer.html --model-name cancer_web

# Example 4: Train with a crawled website
echo "Example 4: Training with a crawled website"
python save_model.py --crawl-url https://www.cancer.org/cancer/types/breast-cancer/ --max-pages 5 --model-name cancer_crawl

# Run the BentoML service with the default model
echo "Running BentoML service with the default model"
bentoml serve .
