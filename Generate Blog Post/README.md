# Generate Blog Posts using GPT2 & Hugging Face Transformers

**A blog post is any article, news piece, or guide published in a website's blog section. A blog post typically covers a specific topic or query, is educational, ranges from 600 to 2,000+ words, and contains other media types such as images, videos, infographics, and interactive charts.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/Mahmudul-Hasan-Shakill/Transformers_Model/blob/main/Generate%20Blog%20Post/Generate_Blog_Post.ipynb]

## Installation

```bash
!pip install transformers
!pip install tensorflow
!pip install gradio
```

## Usage

```python
import tensorflow as tf
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
```

## Pre-trained Model 

```python
model_name = "gpt2-large"

tokenizer = GPT2Tokenizer.from_pretrained(model_name )
model = GPT2LMHeadModel.from_pretrained(model_name , pad_token_id=tokenizer.eos_token_id)
```

## Visualization

```python
def generate_text(input):
  input_ids = tokenizer.encode(input, return_tensors='pt')
  beam_output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
  output = tokenizer.decode(beam_output[0], skip_special_tokens=True)
  return '.'.join(output.split('.')[:-1]) + '.'
```

```python
output_text = gr.outputs.Textbox()
gr.Interface(generate_text, 'textbox', output_text, title='Generate Blog Post',
             description = 'Generate Blog Post with GPT-2').launch(debug=True)
```
## License
[MIT](https://choosealicense.com/licenses/mit/)
