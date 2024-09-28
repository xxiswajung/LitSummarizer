import pandas as pd
import datetime, os, re, json
from datetime import datetime
from openai import OpenAI
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def clean_text_for_excel(text):
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)
    
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = text.replace('“', '"').replace('”', '"') 
    text = text.replace('’', "'").replace('‘', "'")  
    text = text.replace('–', '-')  
    text = text.replace('—', '-')  
    
    return text


# def extract_metadata(file_path: str) -> dict:
#     from pdfminer.pdfparser import PDFParser
#     from pdfminer.pdfdocument import PDFDocument

#     metadata = {}
#     try:
#         with open(file_path, 'rb') as f:
#             parser = PDFParser(f)
#             doc = PDFDocument(parser)
#             meta = doc.info[0]  # Metadata is a list of dictionaries
#             metadata = {key: value.decode('utf-8', errors='ignore') if isinstance(value, bytes) else value 
#                         for key, value in meta.items()}
#     except Exception as e:
#         print(f"Metadata extraction failed: {e}")
#     return metadata

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdfs(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            with open(os.path.join(folder_path, filename), 'rb') as file:
                reader = PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() or ''
                texts.append(text)
    return texts

def extract_text_first_page(file_path):
    try:
        text = extract_text(file_path, page_numbers=[0])
        return text
    except Exception as e:
        print(f"Text extraction failed: {e}")
        return ""

def ask_chatgpt_for_metadata(text):
    try:
        prompt = (
            "You are an expert in academic papers. Given the following text from the first page of a PDF, "
            "identify the title of the paper, the authors, and the publication year:\n\n"
            f"{text}\n\n"
            "Please respond with the title followed by 'Title:', the authors followed by 'Authors:', and the year "
            "formatted as 'Year: YYYY'."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in academic paper analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response.choices[0].message.content
        
        title_match = re.search(r'Title:\s*(.+)', response_text)
        authors_match = re.search(r'Authors:\s*(.+)', response_text)
        year_match = re.search(r'Year:\s*(\d{4})', response_text)

        title = title_match.group(1).strip() if title_match else "Unknown Title"
        authors = authors_match.group(1).strip() if authors_match else "Unknown Authors"
        year = int(year_match.group(1).strip()) if year_match else "Unknown Year"

        return title, authors, year

    except Exception as e:
        print(f"Error asking ChatGPT for metadata: {e}")
        return "Unknown Title", "Unknown Authors", "Unknown Year"

def process_paper(file_path):
    """Process the PDF to extract title, authors, and publication year using ChatGPT."""
    text = extract_text_first_page(file_path)
    
    title, authors, year = ask_chatgpt_for_metadata(text)
    formatted_authors = f"{authors} ({year})"

    return title, formatted_authors, year

def process_folder(folder_path, output_filename):
    data_records = []
    folder_summary = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return
    
    for idx, pdf_file in enumerate(pdf_files, start=1):
        print(f"Processing file {idx}/{len(pdf_files)}: {pdf_file}")
        file_path = os.path.join(folder_path, pdf_file)
        
        title, authors, year = process_paper(file_path)
        try:
            full_text = extract_text(file_path)
            full_text_clean = clean_text(full_text)
        except Exception as e:
            print(f"Failed to extract text from {pdf_file}: {e}")
            continue 
        
        summaries = summarize_paper(full_text_clean)
        
        record = {
            "Index": idx,
            "Paper Name": title,
            "Paper Authors": authors,
            "Publication Year": year,
            **summaries
        }
        data_records.append(record)
        folder_summary[title] = summaries
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(data_records)
    df.to_excel(output_filename, index=False)
    print(f"\nAll papers processed. Summary saved to '{output_filename}'.")

    return folder_summary

def create_comprehensive_review(folder_summary, broad_topic):
    review = f"Objective: Create a comprehensive literature review focusing on {broad_topic}, considering various perspectives and key determinants.\n\n"
    
    review += "Sections:\n\n"
    # Introduction
    review += f"Introduction: The importance of studying {broad_topic} is significant, as it provides insights into key determinants and varying perspectives. Key contributors include...\n\n"

    # Major Thematic Sections
    review += "Major Thematic Section 1: \n"
    review += "Discuss aspect 1, breaking it down into relevant sub-categories.\n\n"

    review += "Major Thematic Section 2: \n"
    review += "Explore aspect 2, breaking it down into relevant sub-categories.\n\n"

    review += "Major Thematic Section 3: \n"
    review += "Analyze aspect 3, breaking it down into relevant sub-categories.\n\n"

    # Additional Thematic Sections if needed
    review += "Additional Thematic Sections: Add more sections as needed to cover all relevant aspects of the topic.\n\n"

    for filename, summary in folder_summary.items():
        review += f"\nPaper: {filename}\n"
        for key, value in summary.items():
            review += f"{key}: {value}\n"

    # Add more detailed sections based on the summaries here
    review += "\nIntegration:\n"
    review += "Synthesize findings, highlighting common themes and differences. Use visual aids (diagrams/frameworks/tables) to summarize the literature\n"
    
    review += "\nCritical Analysis:\n"
    review += "Assess strengths, weaknesses, and gaps in the literature. Suggest future research directions.\n"
    
    review += "\nMethodology:\n"
    review += "Discuss methodological approaches and their limitations.\n"
    
    review += "\nConclusion:\n"
    review += "Summarize key findings and their implications.\n"
    
    return review
        
def summarize_paper(text):
    parameters = {
        "RQ": "What is the research question of the paper?",
        "Main Findings": "What are the main findings of the paper?",
        "Contributions": "What are the main contributions of the paper?",
        "Data": "What data does the paper use?",
        "Methods": "What method does it use and is there any issue with endogeneity?",
        "Key Variables": "What are the key variables of the paper?",
        "Limitation and Future Directions": "What are the limitations of the paper and what could future research look like?"
    }
    summaries = {}
    for key, prompt in parameters.items():
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert assistant. Please provide concise and informative responses. Each response should be no longer than 2-3 sentences."},
                {"role": "user", "content": f"{prompt}\n\n{text}"}
            ]
        )
        summary_text = response.choices[0].message.content
        summaries[key] = clean_text_for_excel(summary_text)  # Clean the summary text before saving
    return summaries

# def summarize_folder(folder_path):
#     texts = extract_text_from_pdfs(folder_path)
#     folder_summary = {}
#     for i, text in enumerate(texts):
#         filename = f"Paper_{i+1}"  # Assuming papers are named sequentially
#         print(f"Summarizing {filename}...")
#         paper_summary = summarize_paper(text)
#         folder_summary[filename] = paper_summary
#     return folder_summary

def ask_chatgpt(question,context):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        # The system message is optional and can be used to set the behavior of the assistant
        # The user messages provide requests or comments for the assistant to respond to
        # Assistant messages store previous assistant responses, but can also be written by you to give examples of desired behavior 
        messages=[
            {"role": "system", "content": "You are an expert assistant, skilled in comparing research topics."},
            {"role": "user", "content": f"{context}\n\n{question}"}
        ]
    )
    return response.choices[0].message.content

def load_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {"questions": [], "answers": []}

def save_history(file_path, history):
    with open(file_path, 'w') as file:
        json.dump(history, file, indent=4)

def update_history(file_path, question, answer):
    history = load_history(file_path)
    history["questions"].append(question)
    history["answers"].append(answer)
    save_history(file_path, history)

def save_answer_to_file(filename, answer):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chatgpt_response_{timestamp}.txt"
    with open(filename, 'w') as file:
        file.write(answer)
    print(f"Answer saved to {filename}")

def main():
    history_file = "chat_history.json"

    num_folders = int(input('How many folders do you want to analyze? '))
    folder_info = {}
    for i in range(num_folders):
        folder_path = input(f'Enter the path for folder {i + 1}: ')
        label = input(f'Enter the label for folder {i + 1}: ')
        folder_info[label] = folder_path

    all_reviews = []

    while True:
        history = load_history(history_file)
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(history["questions"], history["answers"])])

        question = input('Ask a question (or type "exit" to quit): ').strip()
        if question.lower() in ['exit', 'quit', 'end']:
            print("Exiting the program.")
            break

        print("\nWorking...\n")

        for label, folder_path in folder_info.items():
            # Summarize papers in the folder
            output_file = f"{label}_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            folder_summary = process_folder(folder_path, output_file)

            broad_topic = f"{label} research"
            review = create_comprehensive_review(folder_summary, broad_topic)
            all_reviews.append(f"\nComprehensive Review for {label}:\n{review}\n")    
        
        combined_reviews = "\n".join(all_reviews)
        message = f"{combined_reviews}\n\nQuestion:\n{question}"
        
        answer = ask_chatgpt(message, context)
        save_answer_to_file("chatgpt_response.txt", answer)
        update_history(history_file, question, answer)

if __name__ == "__main__":
    main()