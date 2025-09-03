from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the pad_token_id to the eos_token_id to avoid warnings
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_response(prompt, max_length=50):
    # Tokenize the input prompt and include attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    #Generate a response using the model
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"], # Pass the attention mask
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode the response and return it
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):] #Only return the generated text

def chat():
    print("Welcome to the GPT-2 Chatbot! Type `quit` to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        if not user_input.strip(): # Check if input is empty or only whitespace
            print("Chatbot: Please enter something.")
            continue # Skip to the next iteration if input is empty

        response = generate_response(user_input)
        print("Chatbot:", response)

chat()

# As using a large model and may run out of memory,so free up GPU memory. 
del model
del tokenizer
torch.cuda.empty_cache()