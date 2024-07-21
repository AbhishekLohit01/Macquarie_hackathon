from docx import Document as DocxDocument
from PyPDF2 import PdfReader
from dotenv import dotenv_values
import openai
import numpy as np
import pandas as pd
from scipy import spatial
import tiktoken
import ast
import logging as logger
import os
from multiprocessing import Pool, cpu_count

def pipeline_flow(file_path):
    """Pipeline flow for extracting and embedding paragraphs from a document."""
    print("pipeline_flow", file_path)
    obj = GenerateEmbeddings()
    paragraphs = obj.extract_paragraphs_from_text(file_path)
    para_chunks = obj.extract_para_chunks(paragraphs)
    return file_path, para_chunks

class GenerateEmbeddings:
    def __init__(self):
        # Load API key from environment file
        self.api_key = dotenv_values(os.path.join(os.path.dirname(__file__), "..", "config", "api_data.env"))
        openai.api_key = self.api_key['OPEN_AI_KEY']
        self.client = openai
        
        # Set embedding and GPT models
        self.EMBEDDING_MODEL = "text-embedding-3-small"
        self.GPT_MODEL = "gpt-3.5-turbo"
        # Configure logging
        self.logger = logger
        self.logger.basicConfig(level=logger.DEBUG, filename=os.path.join(os.path.dirname(__file__), "..", "log_file", "logs_data.log"), filemode='w+')        
        # Load and preprocess embedding data if exists
        self.df = None
        embedding_file_path = os.path.join(os.path.dirname(__file__), "..","..", "input_data", "embedding_output", "embedded_Macquarie_Group_announces_$A3.csv")
        if os.path.exists(embedding_file_path):
            self.df = pd.read_csv(embedding_file_path).drop(columns=['Unnamed: 0'])
            self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)

    def extract_paragraphs_from_text(self, file_path):
        """Extract paragraphs and headings from a document based on its file type."""
        print("Inside extract_paragraphs_from_text")
        paragraphs = []
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == '.docx':
            document = DocxDocument(file_path)
            for paragraph in document.paragraphs:
                paragraphs.append(paragraph.text)
        elif file_extension.lower() == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    paragraphs.append(page.extract_text())
        elif file_extension.lower() in ['.txt']:
            with open(file_path, 'r') as file:
                paragraphs = file.readlines()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return paragraphs

    def extract_para_chunks(self, paragraphs):
        """Extract chunks of paragraphs."""
        print("Inside extract_para_chunks")
        chunk_size = 5
        return [' '.join(paragraphs[i:i + chunk_size]) for i in range(0, len(paragraphs), chunk_size)]

    def create_embeddings(self, para_chunks):
        """Create embeddings for a list of paragraph chunks."""
        print("Inside create_embeddings")
        embeddings_dict = {}
        embeddings = []

        for batch in range(0, len(para_chunks)):
            response = self.client.embeddings.create(model=self.EMBEDDING_MODEL, input=para_chunks[batch])
            for i, be in enumerate(response.data):
                assert i == be.index  # double check embeddings are in same order as input
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
            embeddings_dict[para_chunks[batch]] = batch_embeddings

        embedded_data = pd.DataFrame(list(embeddings_dict.items()), columns=['text', 'embedding'])
        embedded_data.to_csv(os.path.join(os.path.dirname(__file__), "..","..", "input_data", "embedding_output", "embedding_Macquarie_Group_announces_$A3_2.csv"), index=False)
        self.df = embedded_data

    def strings_ranked_by_relatedness(self, query, relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y), top_n=100):
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        try:
            self.logger.info("pre embedding call")
            query_embedding_response = self.client.Embedding.create(
                model=self.EMBEDDING_MODEL,
                input=query,
            )
            self.logger.info("post embedding call")
            query_embedding = query_embedding_response['data'][0]['embedding']
            strings_and_relatednesses = [
                (row["text"], relatedness_fn(query_embedding, row["embedding"]))
                for i, row in self.df.iterrows()
            ]
            strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
            strings, relatednesses = zip(*strings_and_relatednesses)
            self.logger.info("post1 embedding call")
            return strings[:top_n], relatednesses[:top_n]
        except Exception as e:
            self.logger.error("Error ranking strings by relatedness: %s", e)
            return [], []

    def process_files_in_parallel(self, file_paths, num_workers=None):
        print("inside process_files_in_parallel")
        if num_workers is None:
            num_workers = cpu_count()
        
        para_chunks_final = []
        with Pool(num_workers) as pool:
            results = pool.map(pipeline_flow, file_paths)
        print("check",results)
        for file_path, para_chunks in results:
            para_chunks_final.extend(para_chunks)
        
        self.create_embeddings(para_chunks_final)

    def num_tokens(self, text, model1):
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(model1)
        return len(encoding.encode(text))
    
    def query_message(self, query, token_budget):
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        try:
            self.logger.info("Inside query_message function")
            strings, relatednesses = self.strings_ranked_by_relatedness(query)
            introduction = 'Use the below articles on the SOW of a Company to answer questions"'
            question = f"\n\nQuestion: {query}"
            message = introduction

            for string in strings:
                next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
                text = message + next_article + question
                encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
                if len(encoding.encode(text)) > token_budget:
                    break
                else:
                    message += next_article
            self.logger.info("message+q: %s", message + question)
            return message + question
        except Exception as e:
            self.logger.error("Error creating query message: %s", e)
            return ""
    
    def ask(self, query, token_budget=500, print_message=False):
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        self.logger.info("Inside ask function")
        message = self.query_message(query, token_budget=token_budget)
        if print_message:
            print(message)
            self.logger.info("message: %s", message)
        messages = [
            {"role": "system", "content": "You answer questions about the macquarie annual report news."},
            {"role": "user", "content": message},
        ]
        response = self.client.chat.completions.create(
            model=self.GPT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=50
        )
        response_message = response.choices[0].message.content
        return response_message

if __name__ == '__main__':
    obj = GenerateEmbeddings()
    # file_paths = [
    #     os.path.join(os.path.dirname(__file__), "..","..", "input_data", "embedding_input", "Macquarie Group announces $A3.docx"),
    #     os.path.join(os.path.dirname(__file__), "..","..", "input_data", "embedding_input", "RENT_RECIPT_JAN24_MAR24.pdf"),
    # ]
    directory_path = r'C:\Users\ACER\Downloads\Macquarie_hackathon\Macquarie_hackathon\Hackathon\input_data\embedding_input'
    file_paths = [os.path.join(r'C:\Users\ACER\Downloads\Macquarie_hackathon\Macquarie_hackathon\Hackathon\input_data\embedding_input', file) for file in os.listdir(directory_path)]
    obj.process_files_in_parallel(file_paths)
