import os
import io
import pdfplumber
import openpyxl
import pandas as pd
import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from typing import List
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS settings (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Updated Hugging Face endpoint
HF_API_URL = "https://api-inference.huggingface.co/embeddings"

def get_embedding(text: str) -> List[float]:
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": [text],
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    print("👉 Sending payload to Hugging Face:", payload)

    response = httpx.post(HF_API_URL, json=payload, headers=headers)

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        print("❌ Hugging Face response error:", e.response.text)
        raise

    print("✅ Hugging Face returned:", response.json())
    return response.json()["embeddings"][0]

def parse_pdf(file_bytes: bytes) -> pd.DataFrame:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    lines = text.strip().split("\n")
    data = [line.split() for line in lines if line]
    df = pd.DataFrame(data)
    return df

def parse_excel(file_bytes: bytes) -> pd.DataFrame:
    wb = openpyxl.load_workbook(filename=io.BytesIO(file_bytes), data_only=True)
    ws = wb.active
    data = ws.values
    columns = next(data)
    df = pd.DataFrame(data, columns=columns)
    return df

@app.post("/match")
async def match_file(file: UploadFile = File(...)):
    file_bytes = await file.read()

    if file.filename.endswith(".pdf"):
        df = parse_pdf(file_bytes)
    elif file.filename.endswith((".xls", ".xlsx")):
        df = parse_excel(file_bytes)
    else:
        return {"error": "Unsupported file type."}

    # Identify the description column
    description_col = next((col for col in df.columns if "description" in col.lower()), None)
    if not description_col:
        return {"error": "No description column found."}

    results = []
    for description in df[description_col].dropna():
        try:
            embed = get_embedding(description)

            # Your Supabase RPC function call
            response = supabase.rpc("match_products", {"embedding": embed}).execute()
            top_matches = response.data if hasattr(response, "data") else []

            results.append({
                "original": description,
                "matches": top_matches
            })
        except Exception as e:
            results.append({
                "original": description,
                "error": str(e)
            })

    return {"results": results}
