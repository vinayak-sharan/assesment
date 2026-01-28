import os
import asyncio
from typing import TypedDict, List, Optional, Union
from pydantic import BaseModel, Field
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


from langchain_google_genai import ChatGoogleGenerativeAI

file_write_lock = asyncio.Lock()



# --- Data Models ---
class ExtractField(BaseModel):
    # Allow float/int for amounts, str for text
    value: Union[str, float, int, None] = Field(None)
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")

class GermanInvoice(BaseModel):
    company_name: ExtractField = Field(..., description="Name of the company issuing the invoice")
    invoice_number: ExtractField = Field(..., description="The unique identifier for the invoice")
    invoice_date: ExtractField = Field(..., description="Date of the invoice in YYYY-MM-DD format")
    due_date: ExtractField = Field(..., description="Date payment is due in YYYY-MM-DD format")
    total_amount: ExtractField = Field(..., description="Total amount including tax as a float")
    bank_name: ExtractField = Field(..., description="Name of the Bank")
    iban: ExtractField = Field(..., description="International Bank Account Number")


# --- State Definition ---
class AgentState(TypedDict):
    image_bytes: bytes
    image_path: str
    extraction: Optional[GermanInvoice]
    safety_check: str  # "pass" or "flagged"
    messages: List[str]


# --- Model Setup ---
# Gemini 2.0 Flash is excellent for vision + structured data
llm = ChatGoogleGenerativeAI(
    model="gemini-3-pro-image-preview",
    temperature=0.0, # Low temp for extraction tasks
)
structured_llm = llm.with_structured_output(GermanInvoice)

# import os
# import google.generativeai as genai
#
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# print("Available Models:")
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(f"- {m.name}")
"""   """