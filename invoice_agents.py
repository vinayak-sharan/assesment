import base64
import os
import asyncio
import json
from langchain_core.messages import HumanMessage
from orchestrator import AgentState, structured_llm
import logging
logger = logging.getLogger(__name__)

file_write_lock = asyncio.Lock()


async def extract_node(state: AgentState):
    logger.info(f"Extracting data for {state['image_path']}...")

    if not state['image_bytes']:
        logger.error("No image bytes found.")
        return {"extraction": None, "safety_check": "flagged"}

    image_b64 = base64.b64encode(state['image_bytes']).decode("utf-8")

    system_instructions = """
    ### ROLE
    You are a specialized AI assistant for German Accounting (DACH region). Your task is to extract structured data from invoice images with high precision.

    ### CONTEXT
    The document provided is a German invoice ("Rechnung"). 
    - Language: German
    - Currency: Usually EUR (€)
    - Number Format: German standard (uses ',' as decimal separator and '.' as thousands separator).
    
     ### FIELD DEFINITIONS & FORMATTING RULES
    1. **invoice_number**: Look for "Rechnungsnummer", "Beleg-Nr.", "Rech-Nr".
    2. **date**: Look for "Rechnungsdatum", "Datum". **Convert all dates to ISO format (YYYY-MM-DD).**
    3. **total_amount**: Look for "Gesamtbetrag", "Bruttobetrag", "Zahlbetrag". **Convert to a standard float (e.g., transform "1.050,50" to 1050.50).**
    4. **vendor_name**: The company issuing the invoice.
    5. **iban**: Look for "IBAN" in the footer.

    ### INSTRUCTIONS
    1. Extract the specific fields listed into the JSON schema provided.
    2. For "total_amount", convert the German format (e.g., "1.050,50") to a standard float (1050.50).
    3. For dates, convert to ISO format (YYYY-MM-DD).
    4. If a field is not found, return null for value and 0.0 for confidence.
    """

    msg = HumanMessage(content=[
        {"type": "text", "text": system_instructions},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
    ])

    try:
        result = await structured_llm.ainvoke([msg])
        return {"extraction": result}
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {"extraction": None, "safety_check": "flagged"}


async def audit_node(state: AgentState):
    data = state.get('extraction')

    if not data:
        logger.warning("No data extracted. Flagging.")
        return {"safety_check": "flagged"}

    critical_fields = [data.invoice_number, data.total_amount]

    if any(f.confidence < 0.85 for f in critical_fields):
        logger.info(f"DECISION: Low confidence detected for {state.get('image_path')}. Routing to Human.")
        return {"safety_check": "flagged"}

    if data.total_amount.value is None or data.invoice_number.value is None:
        logger.info(f"DECISION: Missing critical values for {state.get('image_path')}. Routing to Human.")
        return {"safety_check": "flagged"}

    logger.info(f"DECISION: High confidence for {state.get('image_path')}. Auto-approving.")
    return {"safety_check": "pass"}


async def human_review_node(state: AgentState):
    """
    Adds the filename to review_queue.json safely using a Lock.
    """
    filename = state.get('image_path', 'unknown_invoice.jpg')
    review_file = "review_queue.json"

    logger.info(f"Flagging file: {filename} for human review.")

    async with file_write_lock:
        if os.path.exists(review_file):
            try:
                with open(review_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {"invoice_needs_review": []}
        else:
            data = {"invoice_needs_review": []}

        if filename not in data["invoice_needs_review"]:
            data["invoice_needs_review"].append(filename)

            with open(review_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("File write complete.")
        else:
            logger.info(f"File {filename} is already in the queue.")

    return {"messages": [f"Flagged {filename}"]}

