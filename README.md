# LitSummarizer 📚

**LitSummarizer** is a tool designed to streamline the process of summarizing and analyzing research papers. It leverages the OpenAI API for generating insights from PDF documents and includes utilities to clean and prepare extracted text for further analysis or export, such as to Excel.

### Key Features:
1. **Text Extraction & Cleaning:**
   - Cleans and formats text by removing control characters and replacing special symbols (e.g., quotes, dashes) for compatibility with data tools like Excel.

2. **OpenAI Integration:**
   - Utilizes OpenAI’s language model for generating comprehensive summaries and insights from research papers.

3. **Planned Metadata Extraction:**
   - A placeholder for future functionality to extract metadata from PDFs using `pdfminer`.

### How It Works:
- You can specify your local folder which contains a number of papers, and the tool will process the files, clean the text, and leverage the OpenAI API to generate summaries of the papers' content.


#### LitSummarizer.py 
- Input the location of your folder and the prompts directly in the terminal.

#### LitSummarizer_batchmode.py
- Concerned about the cost of using OpenAI? This script offers the same quality at half the price!
- Update the folder location directly in the code to use this version.
