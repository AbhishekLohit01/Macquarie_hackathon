import json
import PySimpleGUI as sg
from function_calling import FunctionCalling
from generate_embeddings import GenerateEmbeddings

os.environ['TK_SILENCE_DEPRECATION'] = '1'


class Chat(FunctionCalling, GenerateEmbeddings):
    def __init__(self):
        super().__init__()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "Answer_generic_questions",
                    "description": """Use this to speak to the user to give them information using the ongoing conversation and a apt response.
                                    Read the whole previous conversation till the latest user query and see if you can answer.
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
   
    def execute_function(self, function_calls, messages):
        """Wrapper function to execute the tool calls"""
        self.logger.info("inside execute_function")
        for function_call in function_calls.tool_calls:
            function_id = function_call.id
            function_name = function_call.function.name
            function_arguments = json.loads(function_call.function.arguments)
        
            if function_name == 'get_information':
                respond = True
                self.logger.info("inside get_information")
                instruction_name = function_arguments['information']
                
                response_message_embedding = GenerateEmbeddings.ask(instruction_name)
                messages.append({
                    "role":"assistant", 
                    "content": response_message_embedding
                })
                
                print(f"Assistant: {response_message_embedding}")
                print("\n\n")             
            elif function_name == 'ask_database':
                respond = True
                instruction_name = function_arguments['query']
                # response_message = FunctionCalling.ask_database(instruction_name)
                messages.append({
                    "role":"assistant", 
                    "content": instruction_name
                })
                
                print(f"Assistant: {response_message}")
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
    
    def create_window(self):
        sg.theme('GreenTan')  # Set theme for the window
        
        layout = [[sg.Text('Your output will go here', size=(40, 1))],
                [sg.Output(size=(110, 20), font=('Helvetica 10'))],
                [sg.Multiline(size=(70, 5), enter_submits=True, key='-QUERY-', do_not_clear=False),
                sg.Button('SEND', button_color=(sg.YELLOWS[0], sg.BLUES[0]), bind_return_key=True),
                sg.Button('EXIT', button_color=(sg.YELLOWS[0], sg.GREENS[0]))]]
        
        return sg.Window('Chat window', layout, font=('Helvetica', ' 13'), default_button_element_size=(8,2), use_default_focus=False)


    def chat_block(self):
        window = self.create_window()
        messages = [
            {
                "role": "system",
                "content": self.assistant_system_prompt
            }
        ]
        try:
            while True:     
                event, values = window.read()
                if event in (sg.WIN_CLOSED, 'EXIT'):            
                    break
                if event == 'SEND':
                    user_question = values['-QUERY-'].rstrip()
                    user_message = {"role":"user","content": user_question}
                    messages.append(user_message)

                    print(f"User: {user_question}")

                    response = self.client.chat.completions.create(model=self.GPT_MODEL,
                        messages=messages,
                        temperature=0,
                        tools=self.tools,
                        tool_choice='required'
                    )
                    
                    respond, messages = self.execute_function(response.choices[0].message, messages)
                    print("\n\n")
                    
        except Exception as e:
            print("Exception occurred: %s", e)
            
        window.close()
