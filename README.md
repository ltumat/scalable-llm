# Second project of Scalable ML

Authors:
- Jonas Lorenz
- Álvaro Mazcuñán Herreros


## How we finetuned the models

Recently we obtained a lot of Modal credits and decided to use them here. So, we did not use Google Colab's GPUs and finetuned our models with the hardware provided by Modal. As you can see, we have a file named `modal_tune.py` that has two entrypoints: `run` and `push`:

- `run` launches the script where we do the finetune, called `finetune.py`. We created this command so that it can run with multiple GPUs. For the finetuning, we are using `trl.SFTTrainer` (https://huggingface.co/docs/trl/sft_trainer), which is used for supervised finetuning.
- `push` reads our selected checkpoints from a model and push them to HuggingFace.


## Models

We finetuned the following models:

- `Qwen/Qwen3-4B-Instruct-2507` (https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). This model was finetuned using the `mlabonne/FineTome-100k` dataset, as required by the project instructions. The fine-tuned model is accessible at `alvaro-mazcu/Qwen3-4B-Instruct-FineTome` (https://huggingface.co/alvaro-mazcu/Qwen3-4B-Instruct-FineTome). 

- `meta-llama/Llama-3.2-1B-Instruct` (https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). This model was finetuned using a dataset containing 3402 Q&A pairs from LeBron James. The resulted model can be found at `alvaro-mazcu/Llama-3.2-1B-Instruct-LeBron` (https://huggingface.co/alvaro-mazcu/Llama-3.2-1B-Instruct-LeBron).


## Deployment

We deployed both finetuned models in their respective HuggingFace Space using the Gradio framework. You can chat with the models using the chat window. The first call every few hours might take some time as the model checkpoint must be downloaded. Here are the links:

- `alvaro-mazcu/Qwen3-4B-Instruct-FineTome` --> https://huggingface.co/spaces/alvaro-mazcu/Qwen3-4B-Instruct-FineTome-gradio
- `alvaro-mazcu/Llama-3.2-1B-Instruct-LeBron` --> https://huggingface.co/spaces/alvaro-mazcu/LeBron


## Datasets

- `mlabonne/FineTome-100k` is a subset of `arcee-ai/The-Tome`, as described in the official page: https://huggingface.co/datasets/mlabonne/FineTome-100k.

- The LeBron's interwiew dataset was obtained from this web page: https://www.asapsports.com/show_player.php?id=13888. We created a script to read all these interviews and download the transcript, parsing it in the desired format. The resulting dataset can be seen in the `lebron_james` directory of this repository.
