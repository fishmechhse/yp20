import os
from dotenv import load_dotenv

load_dotenv()

BOT = os.getenv("BOT_TOKEN")
API_URL = os.getenv("API_URL")