import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# NOTE: try: srun -w dgx2-4 -p dgx2 -t 1-00:00:00 --gres=gpu:2 --mem=100G --pty bash
# Still doesn't work on the hpc but will run in colab on an A100. Still need to debug.

def initialize_model(model_name):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # cache_dir="/nfs/hpc/share/zontosj/.cache/huggingface",
        torch_dtype=torch.bfloat16,
        rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
    )

    # Check if GPUs are available and set the model to use them
    if torch.cuda.is_available():
        model = model.to('cuda')

    # Explicitly set pad_token_id if it's not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def interactive_chat(model, tokenizer):
    print("Starting interactive chat with the model. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Generate response
        with torch.no_grad():
            inputs = tokenizer.encode(user_input, return_tensors='pt').to(model.device)
            reply_ids = model.generate(inputs, max_length=float('inf'))
            reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        
        print(f"Model: {reply}")

def main():
    torch.cuda.empty_cache()
    # model_name = 'gpt2'
    model_name = 'meta-llama/Llama-2-7b-chat-hf'
    model, tokenizer = initialize_model(model_name)
    
    # Start interactive chat
    interactive_chat(model, tokenizer)

if __name__ == "__main__":
    main()
