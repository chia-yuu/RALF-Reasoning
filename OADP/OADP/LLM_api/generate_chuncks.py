import openai
import os
import time
from tqdm import tqdm

def clean_output(text):
    lines = text.strip().split('\n')
    cleaned = []
    for line in lines:
        # Remove leading numbers and symbols like "1.", "2)", etc.
        # line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
        # Remove trailing periods if it's a short phrase
        line = line[3:].rstrip(".")
        cleaned.append(line)
    return "\n".join(cleaned)

def main():
    start_time = time.time()

    # === Set your OpenAI API key ===
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or "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"
    )

    # === Clear the output file at the start ===
    open("output_synonym.txt", "w", encoding="utf-8").close()

    # === Load vocabulary words ===
    with open("input.txt", "r", encoding="utf-8") as file:
        vocab_list = [line.strip() for line in file if line.strip()]

    # === Load and flatten prompt template ===
    with open("gen_synonym.txt", "r", encoding="utf-8") as file:
        prompt_template_1 = " ".join(line.strip() for line in file)
    
    with open("gen_difference.txt", "r", encoding="utf-8") as file:
        prompt_template_2 = " ".join(line.strip() for line in file)

    # === Process each vocabulary with progress bar ===
    with open("output_synonym.txt", "a", encoding="utf-8") as output_file:
        for vocab in tqdm(vocab_list, desc="Processing words"):
            prompt = vocab + prompt_template_1

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            result = response.choices[0].message.content.strip()
            result = clean_output(result)

            # output_file.write(f"\n{vocab}:\n{result}\n")

            result_list = result.split("\n")
            for synonym in result_list:
                prompt = vocab + " vs " + synonym + prompt_template_2

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )

                result = response.choices[0].message.content.strip()
                result = clean_output(result)

                output_file.write(f"{result}\n")

                prompt = synonym + " vs " + vocab + prompt_template_2

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )

                result = response.choices[0].message.content.strip()
                result = clean_output(result)

                output_file.write(f"{result}\n")

            # output_file.write(f"{clean_output(result)}\n")

    end_time = time.time()
    print(f"\nTotal run time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
