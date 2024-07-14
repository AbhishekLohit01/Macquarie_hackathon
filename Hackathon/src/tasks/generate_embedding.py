import os
from spire.doc import Document
from dotenv import dotenv_values
import openai
import numpy as np
import pandas as pd
from scipy import spatial
import tiktoken
import ast
import logging as logger

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
        """Extract paragraphs and headings from a Word document."""
        print("Inside extract_paragraphs_from_text")
        paragraphs = []
        headings = []
        document = Document()
        document.LoadFromFile(file_path)

        # Use a for loop with range to avoid enumerator issues
        for section_index in range(document.Sections.Count):
            section = document.Sections[section_index]
            for paragraph_index in range(section.Paragraphs.Count):
                paragraph = section.Paragraphs[paragraph_index]
                if paragraph.StyleName and "Heading" in paragraph.StyleName:
                    headings.append(paragraph.Text)
                else:
                    paragraphs.append(paragraph.Text)

        document.Close()
        return paragraphs

    def extract_para_chunks(self, paragraphs):
        """Extract chunks of paragraphs."""
        print("Inside extract_para_chunks")
        chunk_size = 5
        return [' '.join(paragraphs[i:i + chunk_size]) for i in range(0, len(paragraphs), chunk_size)]

    def create_embeddings(self, para_chunks):
        """Create embeddings for a list of paragraph chunks."""
        print("Inside extract_embeddings")
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

    def pipeline_flow(self, file_path):
        """Pipeline flow for extracting and embedding paragraphs from a document."""
        print("pipeline_flow", file_path)
        paragraphs = self.extract_paragraphs_from_text(file_path)
        para_chunks = self.extract_para_chunks(paragraphs)
        self.create_embeddings(para_chunks)

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
        response = self.client.ChatCompletion.create(
            model=self.GPT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=50
        )
        response_message = response['choices'][0]['message']['content']
        return response_message

if __name__ == '__main__':
    obj = GenerateEmbeddings()
    obj.pipeline_flow(os.path.join(os.path.dirname(__file__), "..","..", "input_data", "embedding_input", "Macquarie Group announces $A3.docx"))
