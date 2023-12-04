from transformers import pipeline
import csv

def generate_responses(model_name, prompts):
    # Initialize the model pipeline
    generator = pipeline('text-generation', model=model_name)

    # Generate responses for each prompt
    responses = []
    for prompt in prompts:
        response = generator(prompt, max_length=50)
        responses.append(response[0]['generated_text'])

    return responses

def save_to_csv(filename, prompts, responses):
    # Save the prompts and responses to a CSV file
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Prompt', 'Response'])

        for prompt, response in zip(prompts, responses):
            writer.writerow([prompt, response])

def main():
    model_name = 'gpt2'  # Example: using GPT-2 model

    prompts = [
        "What is the future of AI?",
        "Explain the concept of machine learning.",
        "Describe the use of Python in data science."
    ]

    responses = generate_responses(model_name, prompts)
    save_to_csv('dataset.csv', prompts, responses)

    print("Dataset generated successfully.")

if __name__ == "__main__":
    main()
