import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
MISTRAL_API_KEY =  os.getenv("MISTRAL_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR")

PGHOST = os.getenv("PGHOST")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPORT = os.getenv("PGPORT")
PGPASSWORD = os.getenv("PGPASSWORD")

