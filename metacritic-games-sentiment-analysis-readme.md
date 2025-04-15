# Metacritic Games Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)

## 📊 Project Overview

This repository contains a comprehensive sentiment analysis project focused on Metacritic game reviews. It leverages a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model to analyze and classify the sentiment of user reviews for video games listed on Metacritic.

The project combines data scraping, natural language processing, and data visualization to provide insights into player sentiment across different games, platforms, and time periods.

## 🎯 Key Features

- **Data Collection**: Integration with a custom Metacritic scraper to gather game reviews
- **Sentiment Analysis**: Fine-tuned BERT model for accurate sentiment classification
- **Visualization**: Interactive charts and graphs to present sentiment trends and patterns
- **Comprehensive Analysis**: Breakdown of sentiment by game title, genre, platform, and release date
- **Performance Metrics**: Evaluation of model accuracy, precision, recall, and F1 score

## 🗂️ Repository Structure

```
metacritic-games-sentiment-analysis/
│
├── data/
│   ├── raw/                       # Raw scraped data from Metacritic
│   ├── processed/                 # Preprocessed review datasets
│   └── sentiment/                 # Reviews with sentiment annotations
│
├── models/
│   ├── bert_finetuned/            # Fine-tuned BERT model files
│   └── evaluation/                # Model performance metrics and results
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Initial data exploration
│   ├── 02_data_preprocessing.ipynb # Text preprocessing steps
│   ├── 03_model_finetuning.ipynb  # BERT model fine-tuning process
│   ├── 04_sentiment_analysis.ipynb # Applying the model to the dataset
│   └── 05_visualization.ipynb     # Data visualization and insights
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py       # Text preprocessing utilities
│   │   └── data_loader.py         # Data loading functionality
│   │
│   ├── model/
│   │   ├── bert_model.py          # BERT model architecture
│   │   └── training.py            # Model training functionality
│   │
│   ├── analysis/
│   │   ├── sentiment.py           # Sentiment analysis functions
│   │   └── evaluation.py          # Model evaluation metrics
│   │
│   └── visualization/
│       ├── plots.py               # Visualization utilities
│       └── dashboard.py           # Interactive dashboard creation
│
├── requirements.txt               # Project dependencies
├── setup.py                       # Package installation script
└── README.md                      # Project documentation
```

## 📊 Dataset

The project uses a dataset of video game reviews from Metacritic, available on [Kaggle](https://www.kaggle.com/datasets/deepcontractor/metacritic-video-game-reviews). The dataset includes:

- User reviews for thousands of video games
- Review text content
- User scores
- Game metadata (title, platform, release date)
- Publication dates for reviews

Additional data was collected using a custom scraper from a [companion repository](https://github.com/davutbayik/metacritic-scraper) to enrich the analysis with more recent reviews.

## 🤖 BERT Fine-tuning Process

The sentiment analysis leverages a pre-trained BERT model from Hugging Face that was fine-tuned on a labeled subset of game reviews. The fine-tuning process involved:

1. **Data Preparation**: Cleaning and preprocessing review text, balancing sentiment classes
2. **Model Selection**: Using the `bert-base-uncased` model as the foundation
3. **Fine-tuning**: Training the model with review text and sentiment labels
4. **Hyperparameter Optimization**: Tuning learning rate, batch size, and training epochs
5. **Evaluation**: Testing model performance on a held-out validation set

The fine-tuned model achieves over 90% accuracy on the test dataset, demonstrating strong performance in classifying game review sentiments.

## 🔍 Sentiment Analysis Results

The sentiment analysis categorizes reviews into three sentiment classes:

- **Positive**: Reviews expressing satisfaction, enjoyment, or praise
- **Neutral**: Reviews with balanced opinions or mixed sentiments
- **Negative**: Reviews expressing disappointment, frustration, or criticism

Key insights from the analysis include:

- Correlation between user sentiment and critic scores
- Sentiment trends over time for major game franchises
- Platform-specific sentiment patterns
- Genre-based sentiment distribution
- Temporal analysis of sentiment shifts following game updates or patches

## 📈 Visualizations

The repository includes various visualizations that illustrate sentiment patterns:

- Sentiment distribution across game genres
- Temporal sentiment trends
- Platform comparison charts
- Word clouds for positive/negative sentiment vocabulary
- Correlation matrices between sentiment and metadata features

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster model training)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/davutbayik/metacritic-games-sentiment-analysis.git
   cd metacritic-games-sentiment-analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

1. **Data Preparation**:
   ```bash
   python src/data/preprocessing.py
   ```

2. **Model Training** (if you want to retrain the model):
   ```bash
   python src/model/training.py
   ```

3. **Sentiment Analysis**:
   ```bash
   python src/analysis/sentiment.py
   ```

4. **Generate Visualizations**:
   ```bash
   python src/visualization/plots.py
   ```

Alternatively, you can follow the step-by-step process in the Jupyter notebooks located in the `notebooks/` directory.

## 📓 Notebooks Guide

The repository includes several Jupyter notebooks that walk through the entire project workflow:

1. **01_data_exploration.ipynb**: Initial data analysis and understanding the structure of Metacritic reviews
2. **02_data_preprocessing.ipynb**: Text cleaning, tokenization, and preparation for model training
3. **03_model_finetuning.ipynb**: Fine-tuning the BERT model on labeled game reviews
4. **04_sentiment_analysis.ipynb**: Applying the fine-tuned model to classify review sentiments
5. **05_visualization.ipynb**: Creating visualizations and extracting insights from the sentiment data

## 🧪 Model Performance

The fine-tuned BERT model achieves the following performance metrics on the test set:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 92.3%  |
| Precision | 91.7%  |
| Recall    | 90.9%  |
| F1 Score  | 91.3%  |

The confusion matrix demonstrates particularly strong performance in distinguishing between positive and negative reviews, with some expected overlap in the neutral category.

## 🔮 Future Work

- Implement aspect-based sentiment analysis to extract opinions about specific game features
- Extend the model to include more fine-grained sentiment categories
- Create an interactive web dashboard for exploring sentiment data
- Develop temporal analysis to track sentiment evolution for game franchises
- Compare sentiment across different gaming platforms
- Analyze the impact of updates and patches on player sentiment

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Metacritic API Documentation](https://www.metacritic.com/about-metacritic)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Davut Bayik - [GitHub](https://github.com/davutbayik)

## 🙏 Acknowledgements

- [Kaggle](https://www.kaggle.com/) for hosting the initial dataset
- [Hugging Face](https://huggingface.co/) for providing pre-trained models and tools
- [Metacritic](https://www.metacritic.com/) for being the source of the review data
