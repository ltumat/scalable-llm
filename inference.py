from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
import json

def infer_response(model, tokenizer, messages: json):

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    FastLanguageModel.for_inference(model)

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                            temperature = 1.5, min_p = 0.1)
    
    response = tokenizer.batch_decode(outputs)

    return response