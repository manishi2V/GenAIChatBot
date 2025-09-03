# GenAIChatBot

GenAIChatBot is a simple AI-powered chatbot built in Python, utilizing Hugging Face's Transformers library. It uses the GPT-2 language model (`GPT2LMHeadModel`) and tokenizer (`GPT2Tokenizer`) to generate conversational responses, demonstrating the basics of creating a chatbot with generative AI.

## Features

- Uses GPT-2 model for generating human-like text responses
- Simple Python interface for running chatbot conversations
- Easily extensible for more advanced applications

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- [transformers library](https://huggingface.co/docs/transformers/index)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manishi2V/GenAIChatBot.git
   cd GenAIChatBot
   ```

2. **Install dependencies:**
   ```bash
   pip install transformers torch
   ```

### Usage

Here's an example of how to use `GPT2LMHeadModel` and `GPT2Tokenizer` for a basic chatbot:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # instead of gpt2, "gpt2-medium", "gpt2-large" can also be used
model = GPT2LMHeadModel.from_pretrained("gpt2")

def chat():
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        output = model.generate(
            input_ids,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
```

You can save this script as `main.py` and run it with:

```bash
python main.py
```

## Project Structure

```
GenAIChatBot/
├── main.py             # Example chatbot implementation
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
```

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.
