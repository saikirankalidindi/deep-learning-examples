import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Example prompt
prompt = "Once upon a time in a"

# Encode the prompt to tensor
input_ids = tokenizer.encode(prompt, return_tensors='tf')

# Generate text with GPT-2
output = gpt2_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)