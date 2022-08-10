# Text Generation Model using GPT-Neo & Hugging Face Transformers

**A Text Generation model, also known as the causal language model, can be trained on code from scratch to help the programmers in their repetitive coding tasks.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahmudul-Hasan-Shakill/Transformers_Model/blob/main/Text%20Generation%20GPT%20neo/GPT3_GPT_NEO.ipynb)

## Installation

```bash
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
!pip install transformers
!pip install -q gradio
!pip install pynvml
```

## Usage

```python
from transformers import pipeline
import gradio as gr
import tensorflow as tf
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
```

## Model  

```python
model_name = "EleutherAI/gpt-neo-1.3B"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name, pad_token_id = tokenizer.eos_token_id)
```

## GPU 
```python
free_vram = 0.0
if torch.cuda.is_available():
    from pynvml import *
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    free_vram = info.free/1048576000
    print("There is a GPU with " + str(free_vram) + "GB of free VRAM")
```
```python
if model_name == "EleutherAI/gpt-neo-2.7B" and free_vram>13.5:
    use_cuda = True
    model.to("cuda:0")
elif model_name == "EleutherAI/gpt-neo-1.3B" and free_vram>7.5:
    use_cuda = True
    model.to("cuda:0")
else:
    use_cuda = False
```

## Input
```python
def generate_text(input):
  input_ids = tokenizer(input, return_tensors="pt").input_ids
  if use_cuda:
    input_ids = input_ids.cuda()
  gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=500)
  output = tokenizer.batch_decode(gen_tokens)[0]
  return '.'.join(output.split('.')[:-1]) + '.'
```
## Visualization
```
output_text = gr.outputs.Textbox()
out = gr.Interface(generate_text, 'textbox', output_text, title='GPT-Neo',
             description = 'Very first project in GPT-Neo').launch(debug=True)

out.launch()
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
