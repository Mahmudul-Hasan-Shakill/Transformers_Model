# Text Generation Model using GPT2 & Hugging Face Transformers

**A Text Generation model, also known as causal language model, can be trained on code from scratch to help the programmers in their repetitive coding tasks.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahmudul-Hasan-Shakill/Transformers_Model/blob/main/Text%20Generation%20Model/TextGeneration_GPT_2.ipynb)

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
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
```

## Pre-trained Model 

```python
model_name = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_name )
model = TFGPT2LMHeadModel.from_pretrained(model_name, pad_token_id = tokenizer.eos_token_id)
```

## Visualization

```python
def generate_text(input):
  input_ids = tokenizer.encode(input, return_tensors = 'tf')
  beam_output = model.generate(input_ids, max_length=400, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
  output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
  return '.'.join(output.split('.')[:-1]) + '.'
```

```python
output_text = gr.outputs.Textbox()
gr.Interface(generate_text, 'textbox', output_text, title='GPT-2',
             description = 'Very first project in GPT-2').launch(debug=True)
```
## License
[MIT](https://choosealicense.com/licenses/mit/)
