# https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
# https://cookbook.openai.com/examples/using_tool_required_for_customer_service
from spire.doc import *
from spire.doc.common import *
from dotenv import dotenv_values
import openai
from openai import OpenAI
import numpy as np
import os
import sqlite3
import ast 
import pandas as pd
import json
import tiktoken
import sqlite3
from scipy import spatial
import PySimpleGUI as sg
import logging as logger



class initiation:
    def __init__(self):
        self.api_key = dotenv_values("../api_data.env")
        openai.api_key = self.api_key['OPEN_AI_KEY']
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", openai.api_key))

        # models
        self.EMBEDDING_MODEL = "text-embedding-3-small"
        self.GPT_MODEL = "gpt-3.5-turbo"
        self.logger = logger
        self.logger.basicConfig(level=logger.DEBUG,filename=f'log_file/logs_data.log',filemode='w+')

class generate_embeddings(initiation):
    def __init__(self):
        super().__init__()


    df = pd.read_csv('../input_data/embedding_files/embedded_Macquarie_Group_announces_$A3.csv').drop(columns = ['Unnamed: 0'])
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    
    def strings_ranked_by_relatedness(
        self,
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) -> tuple:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        try:
            self.logger.info("pre embedding call")
            query_embedding_response = self.client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=query,
            )
            self.logger.info("post embedding call")
            query_embedding = query_embedding_response.data[0].embedding
            strings_and_relatednesses = [
                (row["text"], relatedness_fn(query_embedding, row["embedding"]))
                for i, row in df.iterrows()
            ]
            strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
            strings, relatednesses = zip(*strings_and_relatednesses)
            self.logger.info("post1 embedding call")
            return strings[:top_n], relatednesses[:top_n]
        except Exception as e:
            self.logger.info("Exception: ",e)

    def num_tokens(text: str, model1: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(model1)
        return len(encoding.encode(text))
    
    def query_message(
        self,
        query: str,
        df: pd.DataFrame,
        token_budget: int
    ) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        try:
            self.logger.info("Inside query_message function")
            strings, relatednesses = self.strings_ranked_by_relatedness(query, df)
            introduction = 'Use the below articles on the SOW of a Company to answer questions"'
            question = f"\n\nQuestion: {query}"
            message = introduction
            # self.logger.info("strings:",strings)

            for string in strings:
                next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
                # self.logger.info("message + next_article + question:",message + next_article + question)
                text = message + next_article + question
                encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
                len_str=  len(encoding.encode(text))
                if (len_str > token_budget):
                    break
                else:
                    message += next_article
            self.logger.info("message+q: %s", message + question)
            return message + question
        except Exception as e:
            self.logger.info("exp:%s",e)
    def ask(
        self,
        query: str,
        df: pd.DataFrame = df,
        token_budget: int = 500,
        print_message: bool = False,
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        self.logger.info("Inside ask function")
        message = self.query_message(query, df, token_budget=token_budget)
        if print_message:
            print(message)
            self.logger.info("message: ",message)
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


class function_calling(initiation):
    def __init__(self):
        super().__init__()
        self.conn = sqlite3.connect("alerting_db")
        print("Opened database successfully")

    def get_table_names(self,conn):
        """Return a list of table names."""
        table_names = []
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for table in tables.fetchall():
            table_names.append(table[0])
        return table_names

    def get_column_names(self,conn, table_name):
        """Return a list of column names."""
        column_names = []
        columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
        for col in columns:
            column_names.append(col[1])
        return column_names

    def get_database_info(self,conn):
        """Return a list of dicts containing the table name and columns for each table in the database."""
        table_dicts = []
        for table_name in self.get_table_names(conn):
            columns_names = self.get_column_names(conn, table_name)
            table_dicts.append({"table_name": table_name, "column_names": columns_names})
        return table_dicts
    
    def database_schema_string(self):
        database_schema_dict = self.get_database_info(self.conn)
        database_schema_string = "\n".join(
            [
                f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
                for table in database_schema_dict
            ]
        )
        return database_schema_string

    def ask_database(conn, query):
        """Function to query SQLite database with a provided SQL query."""
        try:
            results = str(conn.execute(query).fetchall())
        except Exception as e:
            results = f"query failed with error: {e}"
        return results


class chat(function_calling,generate_embeddings):
    # The tools our customer service LLM will use to communicate
    def __init__(self):
        super().__init__()
        self.tools = [
            {
            "type": "function",
            "function": {
                "name": "Answer_generic_questions",
                "description": """Use this to speak to the user to give them information using the ongoing conversation and a apt response.
                                Read the whole previous conversation till the lastest user query and see if you answer.
                                User might ask to summarize any document, do spell check or write a mailer for him/her""",
                "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                    "type": "string",
                    "description": "Text of message to send to user. Can cover multiple topics."
                    }
                },
                "required": ["message"]
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "get_information",
                "description": "Used to get instructions to deal with the user's problem.",
                "parameters": {
                "type": "object",
                "properties": {
                    "information": {
                    "type": "string",
                    "description": """The user wants to know information about the Macquarie annual report.
                                        Use the embedding search functionality to answer properly. """
                    }
                },
                "required": [
                    "information"
                ]
                }
            }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_database",
                    "description": "Use this function to answer user questions about Alerts and Deals in trading data. Input should be a fully formed SQL query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": f"""
                                        SQL query extracting info to answer the user's question.
                                        SQL should be written using this database schema:
                                        {self.database_schema_string}
                                        The query should be returned in plain text, not in JSON.
                                        """,
                            }
                        },
                        "required": ["query"],
                    },
                }
            }
        ]
        self.assistant_system_prompt = """You are a user service assistant. Your role is to answer user questions politely and competently.
        You should follow these instructions to solve the case:
        - Understand their problem and get the relevant instructions.
        - Follow the instructions to solve the user's problem.
        - Help them with any other problems or close the case.

        Only call a tool once in a single message.
        If you need to fetch a piece of information from a system or document that you don't have access to, give a clear, confident answer with some dummy values."""
   
    
    def execute_function(self,function_calls,messages):
        """Wrapper function to execute the tool calls"""
        self.logger.info("inside execute_function")
        for function_call in function_calls.tool_calls:
            function_id = function_call.id
            function_name = function_call.function.name
            function_arguments = json.loads(function_call.function.arguments)
        
            if function_name == 'get_information':
                respond = True
                self.logger.info("inside get_information")
                # instruction_name will have the user query:
                instruction_name = function_arguments['information']
                
                response_message_embedding = self.ask(instruction_name)
                messages.append({
                            "role":"assistant", 
                            "content": response_message_embedding
                            })
                
                print(f"Assistant: {response_message_embedding}")
                print("\n\n")             
            elif function_name == 'ask_database':
                respond = True
                instruction_name = function_arguments['query']
                messages.append({
                            "role":"assistant", 
                            "content": instruction_name
                            })
                
                print(f"Assistant: {instruction_name}")
                print("\n\n")       
            elif function_name == 'Answer_generic_questions':
                respond = True
                instruction_name = function_arguments['message']
                messages.append({
                        "role":"assistant", 
                        "content": instruction_name
                        })
                print(f"Assistant: {instruction_name}")
                print("\n\n") 
        return (respond, messages)
    
    def chat_block(self):
        sg.theme('GreenTan') # give our window a spiffy set of colors

        layout = [[sg.Text('Your output will go here', size=(40, 1))],
                [sg.Output(size=(110, 20), font=('Helvetica 10'))],
                [sg.Multiline(size=(70, 5), enter_submits=True, key='-QUERY-', do_not_clear=False),
                sg.Button('SEND', button_color=(sg.YELLOWS[0], sg.BLUES[0]), bind_return_key=True),
                sg.Button('EXIT', button_color=(sg.YELLOWS[0], sg.GREENS[0]))]]

        window = sg.Window('Chat window', layout, font=('Helvetica', ' 13'), default_button_element_size=(8,2), use_default_focus=False)

        #conversation_messages = []
        messages = [
            {
                        "role": "system",
                        "content": self.assistant_system_prompt
            }
        ]
        try:
            while True:     # The Event Loop
                event, values = window.read()
                if event in (sg.WIN_CLOSED, 'EXIT'):            # quit if exit button or X
                    break
                    #window.close()
                if event == 'SEND':
                    #print("inside send")
                    # Initiate a respond object. This will be set to True by our functions when a response is required
                    
                    user_question = values['-QUERY-'].rstrip()
                    user_message = {"role":"user","content": user_question}
                    messages.append(user_message)

                    print(f"User: {user_question}")

                    # Make the ChatCompletion call with tool_choice='required' so we can guarantee tools will be used
                    response = self.client.chat.completions.create(model=self.GPT_MODEL
                                                            ,messages=messages
                                                            ,temperature=0
                                                            ,tools=self.tools
                                                            ,tool_choice='required'
                                                            )
                    
                    # self.logger.info("Response: ",response)
                    #print("conversation_messages in submit_user_message: ",messages)
                    respond, messages = self.execute_function(response.choices[0].message,messages)
                    print("\n\n")
                    
        except Exception as e:
            print("Exception occured :",e)
            #window.close()
            
        #window.close()
        

if __name__ == '__main__':
    obj = chat()
    obj.chat_block()