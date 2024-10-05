import openai
import os
import json
import tiktoken
from PyPDF2 import PdfReader
import pandas as pd
import requests
import time

openai.api_key = 'replace-with-your-api-key'

# Define model and token limits
tokenizer = tiktoken.get_encoding("o200k_base")
MAX_TOKENS = 4096
MAX_RESPONSE_TOKENS = 1000
MAX_INPUT_TOKENS = MAX_TOKENS - MAX_RESPONSE_TOKENS

# Define your tailored prompts
parameters = {
    "Research Question": "Summarize the primary research question of the paper. Provide exactly three questions in clear question format. Stop after the third question and do not include any additional text or commentary.",
    "Key Findings": "Summarize the key findings of the paper in three exact sentences. Ensure all three sentences are concise, focused, and do not exceed 50 words when combined.",
    "Data Sources": "List up to 5 of the most relevant data sources used in the paper, in bullet-point format. Each point should briefly describe the data source and its relevance.",
    "Innovation Measures": "List the top 5 most relevant measures used to quantify innovation in the paper. Each measure should be a bullet point, starting with **, followed by a brief description in one sentence."
}

def extract_text_from_pdf(file_path):
    """Extract text from a PDF."""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''.join([page.extract_text() or '' for page in reader.pages])
    return text

def split_text_by_tokens(text, max_tokens=MAX_INPUT_TOKENS):
    """Split text into chunks based on token limit."""
    tokens = tokenizer.encode(text)
    chunks = [tokenizer.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks

def upload_jsonl_file(jsonl_file):
    """Uploads the JSONL file to OpenAI's Files API."""
    try:
        with open(jsonl_file, 'rb') as f:
            batch_input_file = openai.File.create(
                file=f,
                purpose='batch'  # Changed from 'fine-tune' to 'batch'
            )
        file_id = batch_input_file['id']

        # Verify that the file_id starts with 'file-'
        if not file_id.startswith('file-'):
            raise ValueError(f"Invalid file ID: {file_id}. Expected an ID that starts with 'file-'.")

        print(f"Batch input file uploaded with ID: {file_id}")
        return file_id

    except Exception as e:
        print(f"Error during file upload: {e}")
        return None


# Prepare batch input for each PDF and save it to a .jsonl file
def prepare_batch_input_for_batch_api(folder_path, output_jsonl_file):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    batch_input = []

    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        text = extract_text_from_pdf(file_path)
        chunks = split_text_by_tokens(text)

        for i, chunk in enumerate(chunks):
            for question, prompt in parameters.items():
                # # Use exact keys from parameters for custom_id
                # custom_id = f"{pdf_file}_chunk_{i+1}_{question.replace(' ', '_')}"
                # Create custom_id without modifying the key's spaces
                custom_id = f"{pdf_file}_chunk_{i+1}_{question}"
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are an experienced research assistant with PhD in Economics at Harvard University. Your role is to conduct a literature review on given papers. For each paper, you will summarize the papers. More specifically, the purpose is to get at the all the measures that papers use to quantify innovation."},
                            {"role": "user", "content": f"{prompt}\n\n{chunk}"}
                        ],
                        "max_tokens": MAX_RESPONSE_TOKENS
                    }
                }
                batch_input.append(request)

    with open(output_jsonl_file, 'w') as f:
        for entry in batch_input:
            f.write(json.dumps(entry) + '\n')

    print(f"Batch input file '{output_jsonl_file}' created successfully.")


def submit_batch_job(file_id):
    headers = {
        'Authorization': f'Bearer {openai.api_key}',
        'Content-Type': 'application/json',
    }

    data = {
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"description": "Summarize research papers batch job"}
    }

    response = requests.post('https://api.openai.com/v1/batches', headers=headers, json=data)
    batch_job = response.json()

    # Check if 'id' exists before trying to access it
    if 'id' in batch_job:
        print(f"Batch job created with ID: {batch_job['id']}")
        return batch_job['id']
    else:
        print(f"Error in batch job creation: {batch_job}")
        return None

def monitor_batch_completion(batch_id):
    """Monitors the batch until it is completed."""
    while True:
        status, batch_status = check_batch_status(batch_id)
        print(f"Batch status: {status}")  # Log the status each time
        if status == 'completed':
            print("Batch job completed.")
            return batch_status['output_file_id']
        elif status == 'failed':
            print("Batch job failed.")
            return None
        else:
            print(f"Waiting for batch completion...")
            time.sleep(600)  # Check every 60 seconds


def check_batch_status(batch_id):
    """Check the status of the batch job."""
    headers = {
        'Authorization': f'Bearer {openai.api_key}',
        'Content-Type': 'application/json',
    }

    try:
        response = requests.get(f'https://api.openai.com/v1/batches/{batch_id}', headers=headers)
        batch_status = response.json()

        # Check if 'status' is in the response
        if 'status' in batch_status:
            return batch_status['status'], batch_status
        else:
            print(f"Error: 'status' key not found in response: {batch_status}")
            return None, batch_status

    except Exception as e:
        print(f"Exception occurred while checking batch status: {e}")
        return None, None

def process_batch_results(output_file_id):
    """Processes the batch output from OpenAI."""
    headers = {
        'Authorization': f'Bearer {openai.api_key}',
        'Content-Type': 'application/json',
    }

    # Retrieve the batch output
    response = requests.get(f'https://api.openai.com/v1/files/{output_file_id}/content', headers=headers)

    # Save results locally
    with open('batch_output.jsonl', 'w') as f:
        f.write(response.text)
    print("Batch results saved to 'batch_output.jsonl'.")

    results = []
    with open('batch_output.jsonl', 'r') as f:
        batch_output = [json.loads(line) for line in f]

    file_summaries = {}

    for entry in batch_output:
        custom_id = entry['custom_id']
        print(f"Processing custom_id: {custom_id}")

        try:
            # Extract filename and question directly without replacing underscores
            filename, chunk_info = custom_id.split("_chunk_", 1)
            chunk_number, question = chunk_info.rsplit("_", 1)
            question = question.strip()  # Just strip any potential whitespace
        except IndexError:
            print(f"Error processing custom_id: {custom_id}. Skipping...")
            continue

        # Initialize file summary if not already present
        if filename not in file_summaries:
            file_summaries[filename] = {key: "" for key in parameters.keys()}

        # Check if the extracted question matches the parameters key directly
        if question not in parameters:
            print(f"Invalid question key: {question}. Skipping...")
            continue

        # Get the content of the response
        if 'response' in entry and 'body' in entry['response'] and 'choices' in entry['response']['body']:
            chunk_summary = entry['response']['body']['choices'][0]['message']['content']
        else:
            chunk_summary = f"No 'choices' found for {custom_id}"
            print(f"Warning: No 'choices' found for {custom_id}")

        # Append the response to the appropriate section
        file_summaries[filename][question] += chunk_summary + " "


    # Prepare results for Excel
    for filename, summary in file_summaries.items():
        results.append({
            'Filename': filename,
            'Research Question': summary['Research Question'].strip(),
            'Key Findings': summary['Key Findings'].strip(),
            'Data Sources': summary['Data Sources'].strip(),
            'Innovation Measures': summary['Innovation Measures'].strip()
        })

    return results



def save_to_excel(summary_data, output_path):
    """Save summarized data to an Excel file."""
    if not summary_data:
        print("No data to write to Excel.")
        return
    df = pd.DataFrame(summary_data)
    print(f"Writing the following data to Excel:\n{df}")  # Debugging print to check data
    df.to_excel(output_path, index=False)
    print(f"Summary data saved to {output_path}")


def main():
    folder_path = "./detection"  # Adjust this to your folder
    jsonl_file = "batch_input.jsonl"
    
    # Step 1: Prepare the batch input
    prepare_batch_input_for_batch_api(folder_path, jsonl_file)
    
    # Step 2: Upload the batch file
    file_id = upload_jsonl_file(jsonl_file)
    if not file_id:
        print("Error: File upload failed. Exiting.")
        return

    # Step 3: Submit the batch job
    batch_id = submit_batch_job(file_id)
    if not batch_id:
        print("Error: Batch job creation failed. Exiting.")
        return

    # Step 4: Monitor batch completion
    output_file_id = monitor_batch_completion(batch_id)
    if not output_file_id:
        print("Error: Batch job did not complete successfully. Exiting.")
        return

    # Step 5: Process and save results
    results = process_batch_results(output_file_id)
    save_to_excel(results, "summarized_papers.xlsx")

if __name__ == "__main__":
    main()
