import sys
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import PyPDF2

def load_document(file_path):
    file_path = Path(file_path)
    if file_path.is_file():
        if file_path.suffix.lower() == '.pdf':
            return load_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return [Document(text=content)]
    elif file_path.is_dir():
        return SimpleDirectoryReader(input_dir=str(file_path)).load_data()
    else:
        raise ValueError(f"The path {file_path} is neither a file nor a directory.")

def load_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        content = ""
        for page in reader.pages:
            content += page.extract_text() + "\n"
    return [Document(text=content)]

def main(input_path):
    # Load the document(s)
    documents = load_document(input_path)

    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # ollama
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("Write 3-4 minute script for a person to read that summarizes the data. Then generate 3 questions that might be asked on the text and answers those questions.")
    print(response)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_file_or_directory>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    main(input_path)