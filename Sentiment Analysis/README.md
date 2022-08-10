# Sentiment Analysis Model using Pipeline & Transformers

**Sentiment analysis is a natural language processing (NLP) technique used to determine whether data is positive, negative, or neutral. Sentiment analysis is often performed on textual data to help businesses monitor brand and product sentiment in customer feedback, and understand customer needs.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahmudul-Hasan-Shakill/Transformers_Model/blob/main/Sentiment%20Analysis/Transformer_Sentiment.ipynb)

## Installation

```bash
!pip install transformers
```

## Usage

```python
from transformers import pipeline
```

## Model  

```python
classifier = pipeline("sentiment-analysis")
```

## Prediction

```python
def toxicity_level(string):
  new_string = [string]
  pred = classifier(new_string)

  print("{}:".format(string))
  print("{}:".format(pred[0]))
  print()
```
## Input
```python
data = input()
toxicity_level(data)
```
## License
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://choosealicense.com/licenses/mit/)
