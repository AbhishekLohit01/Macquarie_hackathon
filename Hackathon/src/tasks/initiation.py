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


class Initiation:
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
        self.logger.basicConfig(level=logger.DEBUG, filename=os.path.join(os.path.dirname(__file__), "..", "log_file", "logs_data_v1.log"), filemode='w+')        