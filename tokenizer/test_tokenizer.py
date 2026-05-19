from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(".")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

test_text = "Hello, how are you? I am testing the tokenizer."

tokens_1 = tokenizer(test_text, max_length=100, truncation=True, padding="max_length")
tokens_2 = tokenizer.encode(test_text)

print("Tokens 1:", tokens_1)
print("Tokens 2:", tokens_2)

print(tokenizer.eos_token_id)
print(tokenizer.eos_token)

tokens = tokenizer(
    test_text, add_special_tokens=True, padding="max_length", max_length=100
)
print("Tokens with special tokens:", tokens)
