from dotenv import load_dotenv
load_dotenv()


import os
from supabase import create_client

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

data = supabase.table("Attendence Management").select("id, Name").execute()
print(data)