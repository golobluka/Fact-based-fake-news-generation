# Synthetic News Generation for Fake News Classification  

This repository contains the implementation and experiments for our research paper on generating and evaluating synthetic fake news, with an emphasis on fact-based manipulations rather than stylistic changes.

## Project Overview

We focus on generating synthetic fake news by systematically extracting critical facts from real articles, modifying them, and regenerating content that simulates fake news while maintaining coherence. We demonstrate how AI-driven content can be manipulated to alter key factual elements while remaining coherent. Additionally, we evaluate various machine learning models for fake news classification and assess the impact of using synthetic data for training.

## Repository Structure

### 1. Data

This folder contains the datasets used in our study:
- **Synthetic News Articles Dataset**: original and synthetically generated news articles focused on vaccination
- **Synthetic Tweets Dataset**: original and synthetically generated tweets related to COVID-19 vaccines
- **MMCoVar News Dataset**: dataset filled with real and fake news articles


These datasets served as the foundation for our synthetic news generation.

### 2. Synthetic Data Generation

This folder contains our methodology for generating synthetic fake news:
- **Fact Characterization**: Code for identifying different types of facts within articles
- **Fact Extraction**: Implementation for extracting critical elements from original content
- **Fact Manipulation**: Procedures for modifying extracted facts
- **Content Regeneration**: Code for creating synthetic news based on modified facts
- **Evaluation Metrics**: Implementation of our evaluation framework:
  - Coherence: Measures how strongly neighboring sentences relate to each other
  - Dissimilarity: Measures how much the synthetic content differs from the original
  - Correctness: Verifies whether the new facts align correctly with the modified article

We employed Llama3.1:8B as our large language model to generate 338 synthetic articles and 481 synthetic tweets.

### 3. Classification Experiments

This folder contains our fake news classification experiments:
- **Feature Engineering Approaches**:
  - TF-IDF: Standard term frequency-inverse document frequency features
  - Entity Features: Named entity extraction to identify unusual patterns
  - Stylometric Features: Writing style metrics to capture stylistic differences
  - Fact Verification Features: Features designed to detect factual inconsistencies
- **Model Implementations**:
  - Traditional machine learning models (Naive Bayes, Logistic Regression, SVM, Random Forest)
  - Neural network models (MLP, BERT)
- **Experimental Setups**:
  - Different synthetic data ratios experiments
  - Transfer learning analysis
  - Advanced approaches testing

## Key Results

Our experiments yielded several critical insights:

1. **Generation Quality**: We successfully generated coherent synthetic fake news with manipulated facts. Manual evaluation showed 7 out of 10 examples were successful.

2. **Model Performance**: Most traditional models performed worse when trained with synthetic data. However, BERT showed a slight improvement (1.1%) when using a small proportion (10%) of synthetic data.

3. **Feature Effectiveness**: Content-based features outperformed stylistic features for synthetic fake news detection, supporting our methodology's focus on factual manipulation.

4. **Synthetic Data Ratio**: Using small amounts (10%) of synthetic data can slightly improve performance with advanced models like BERT, but higher proportions actively harm transfer learning to real-world examples.

5. **Domain Gap**: There appears to be a fundamental difference between the patterns in our synthetic fake news and those in real-world misinformation, limiting the practical utility of our current approach.

## Citation

If you use our code or findings in your research, please cite our paper:

```
@article{sittar2025synthetic,
  title={Synthetic News Generation for Fake News Classification},
  author={Sittar, Abdul and Golob, Luka and Smiljanic, Mateja},
  journal={IEEE Access},
  year={2025}
}
```

