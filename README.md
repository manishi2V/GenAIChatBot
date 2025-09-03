# GenAIChatBot

GenAIChatBot is a simple AI-powered chatbot built in Python using Hugging Face’s Transformers library. It leverages the GPT-2 language model (`GPT2LMHeadModel`) and tokenizer (`GPT2Tokenizer`) to generate conversational responses — a practical demonstration of creating a chatbot with generative AI.

---

## 🛠️ Chatbot Development Flow

**STEP 1: Initialize Project**  
- **Description:** Set up Colab and import libraries  
- **Tools used:** Google Colab, Hugging Face  

**STEP 2: Load Model**  
- **Description:** Utilize Transformers to load GPT-2 model  
- **Tools used:** Hugging Face Transformers  

**STEP 3: Tokenize Input**  
- **Description:** Prepare user input for the model  
- **Tools used:** Hugging Face Transformers  

**STEP 4: Generate Response**  
- **Description:** Model predicts and generates a response  
- **Tools used:** GPT-2 via PyTorch  

---

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
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
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
