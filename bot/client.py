import os
from dotenv import load_dotenv

load_dotenv()

def get_client():
    mock_mode = os.getenv("MOCK_MODE", "False") == "True"

    if mock_mode:
        return None  
    else:
        from binance.client import Client
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")

        client = Client(api_key, api_secret)
        client.FUTURES_URL = os.getenv("BASE_URL")
        return client