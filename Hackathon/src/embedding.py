from spire.doc import *
from spire.doc.common import *
from dotenv import dotenv_values
import openai
from openai import OpenAI
import numpy as np
import pandas as pd


class embedding:
    def __init__(self):
        api_key = dotenv_values("../api_data.env")
        openai.api_key = api_key['OPEN_AI_KEY']
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", openai.api_key))
        # self.document = Document()
        self.EMBEDDING_MODEL = "text-embedding-3-small"
        
    

    def extractParagraphsFromText(self,filePath):
        # Create a list to store the extracted headings
        print("Inside extractParagraphsFromText")
        headings = []
        para = []
        heading_flag = 0
        sectionlst= []
        document = Document()
        document.LoadFromFile(filePath)
        # Iterate through all sections in the document
        for i in range(document.Sections.Count):
            section = document.Sections[i]
            heading_flag = 0
            headng=''
            # Iterate through all paragraphs in each section
            for j in range(section.Paragraphs.Count):
                paragraph = section.Paragraphs[j]
                # Check if the style name of the paragraph contains Heading
                if paragraph.StyleName is not None and "Heading" in paragraph.StyleName:
                    # Get the text of the paragraph and append it to the list
                    headings.append(paragraph.Text)
                else:
                    #if len(paragraph.Text) > 0:
                    para.append(paragraph.Text)
                        
        document.Close()
        return para
    
    def extractParaChucks(self,para):
        print("Inside extractParaChucks")
        # Create a list to store the extracted headings
        
        # Step 1: Define your list
        pars_lst = []

        # Step 2: Define a function to extract chunks
        def extract_chunks(lst, chunk_size):
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]
        
        # Step 3: Extract chunks
        chunk_size = 5
        chunks = list(extract_chunks(para, chunk_size))

        # Step 4: Convert chunks to DataFrames

        for chu in chunks:
            pars_lst.append(' '.join(chu))
        return pars_lst

    def extractEmbeddings(self,pars_lst):
        print("Inside extractEmbeddings")
        embeddings = []
        embeddings_dic ={}
        for batch in range(0, len(pars_lst)):
            response = self.client.embeddings.create(model=self.EMBEDDING_MODEL, input=pars_lst[batch])
            
            for i, be in enumerate(response.data):
                assert i == be.index  # double check embeddings are in same order as input
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
            embeddings_dic[pars_lst[batch]] = batch_embeddings
        print("embeddings_dic:",embeddings_dic)
        df = pd.DataFrame(embeddings_dic.items()).rename(columns = {0:'text',1:'embedding'})
        df.to_csv("embedding_Macquarie_Group_announces_$A3_2.csv")

    def pipelineFlow(self,filePath):
        paras      = self.extractParagraphsFromText(filePath)
        paras_chunks = self.extractParaChucks(paras)
        self.extractEmbeddings(paras_chunks)


if __name__ == '__main__':
    obj = embedding()
    obj.pipelineFlow("../input_data/embedding_files/Macquarie Group announces $A3.docx")