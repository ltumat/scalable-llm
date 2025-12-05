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

- `Qwen/Qwen3-4B-Instruct-2507` (https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). This model was finetuned using the `mlabonne/FineTome-100k` dataset, as required by the project instructions. The fine-tuned model is accessible at `alvaro-mazcu/Qwen3-4B-Instruct-FineTome` (https://huggingface.co/alvaro-mazcu/Qwen3-4B-Instruct-FineTome). This model was finetuned for 8 epochs on 2 H100 GPUs, taking around 12 hours to complete.

- `meta-llama/Llama-3.2-1B-Instruct` (https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). This model was finetuned using a dataset containing 3402 Q&A pairs from LeBron James. The resulted model can be found at `alvaro-mazcu/Llama-3.2-1B-Instruct-LeBron` (https://huggingface.co/alvaro-mazcu/Llama-3.2-1B-Instruct-LeBron). This model was finetunes for a single epoch on 2 A100 GPUs, compleating the finetune in less than 10 minutes.

- `meta-llama/Llama-3.2-1B-Instruct` (https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). Same model again, but using the `mlabonne/FineTome-100k` dataset. We decided to train this model becuase the inference time of `alvaro-mazcu/Qwen3-4B-Instruct-FineTome` took too much time. That model was running super fast in our local CPU, but not in Gradio's. Therefore, we decided to train again the `Llama-3.2-1B-Instruct` with the instruct dataset. The finetuned model can be seen at `alvaro-mazcu/Llama-3.2-1B-Instruct-FineTome` (https://huggingface.co/alvaro-mazcu/Llama-3.2-1B-Instruct-FineTome). It took us 7 minutes to finetune for a single epoch on 8 A100 GPUs (we have a lot of credits in Modal).

## Deployment

We deployed all finetuned models in their respective HuggingFace Space using the Gradio framework. You can chat with the models using the chat window. The first call every few hours might take some time as the model checkpoint must be downloaded. Here are the links:

- `alvaro-mazcu/Qwen3-4B-Instruct-FineTome` --> https://huggingface.co/spaces/alvaro-mazcu/Qwen3-4B-Instruct-FineTome-gradio. Takes around 40 seconds to answer!
- `alvaro-mazcu/FIneTome-1B-Llama3` --> https://huggingface.co/spaces/alvaro-mazcu/FIneTome-1B-Llama3. Fast inference.
- `alvaro-mazcu/Llama-3.2-1B-Instruct-LeBron` --> https://huggingface.co/spaces/alvaro-mazcu/LeBron. As the data that the model has seen is not updated, in this deployment we are getting additional context with Gemini so that the low-parameter model has extra, current information about the user's question. Interact with the tool as if you were a journalist asking random things to LeBron James, who has just played a match.


## Datasets

- `mlabonne/FineTome-100k` is a subset of `arcee-ai/The-Tome`, as described in the official page: https://huggingface.co/datasets/mlabonne/FineTome-100k.

- The LeBron's interwiew dataset was obtained from this web page: https://www.asapsports.com/show_player.php?id=13888. We created a script to read all these interviews and download the transcript, parsing it in the desired format. The resulting dataset can be seen in the `lebron_james` directory of this repository.

## Notes

* `meta-llama/Llama-3.2-1B-Instruct` (https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) is a gated model, which means that if you were to use it, you would need to be approved by Meta to use the weights. Our request was approved some time ago, and we access the model with a special HF token that we have generated.

* We are using `gemini-2.5-flash` to get additional context in the LeBron James inference deployment. When the user asks a question, we use Gemini to add a couple of sentences that contextualize the user's demands. We noted a significant increase in the performance of the model, mostly at identifying real players and past NBA Champions. We are not paying for Gemini, as it is free for students to generate an API key.
