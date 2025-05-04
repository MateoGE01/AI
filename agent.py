# filepath: /Users/cristhianac/Documents/mcpserver/AI/main.py
import os
import io
import json
import requests
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Literal, Optional, List
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes 

#Mucho codigo
# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BACKEND_ENDPOINT = os.getenv("BACKEND_ENDPOINT")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
if not BACKEND_ENDPOINT:
    raise ValueError("BACKEND_ENDPOINT not found in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or another suitable model

# --- Pydantic Models ---
class TransactionData(BaseModel):
    tipo: Literal["ingreso", "gasto", "transferencia"]
    monto: float
    cuentaId: int
    descripcion: Optional[str] = None
    fecha: Optional[str] = None # Consider using date type if format is consistent
    cuentaDestinoId: Optional[int] = None

    @field_validator('monto')
    def monto_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('El monto debe ser positivo')
        return v

class ErrorResponse(BaseModel):
    error: str
    missing_fields: Optional[List[str]] = None
    details: Optional[str] = None

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Texto extraido del speech-to-text")

# --- FastAPI App ---
app = FastAPI(title="MCP AI Document Processor")

# --- Helper Functions ---
def extract_text_from_image(image_bytes: bytes) -> str:
    """Extracts text from image bytes using Tesseract OCR."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen con OCR: {e}")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes."""

    try:
        images = convert_from_bytes(pdf_bytes)
        full_text = ""
        for img in images:
            full_text += pytesseract.image_to_string(img) + "\n"
        return full_text
    except Exception as e:
        print(f"Error processing PDF with pdf2image/tesseract: {e}")
         # Fallback or raise error
        raise HTTPException(status_code=500, detail=f"Error al procesar PDF: {e}")



def extract_info_with_gemini(text: str) -> dict:
    """Uses Gemini to extract transaction details from text."""
    prompt = f"""
    Extrae la siguiente información del texto de la factura o recibo. Responde únicamente con un objeto JSON válido.
    Los campos requeridos son 'tipo', 'monto' y 'cuentaId'.
    Si no encuentras un campo opcional, omítelo del JSON.
    Si no puedes determinar un campo requerido, incluye "null" como valor para ese campo en el JSON.

    Campos a extraer:
    - tipo: (string) El tipo de transacción. Debe ser "ingreso", "gasto" o "transferencia". Infiere si no está explícito.
    - monto: (number) El monto total de la transacción. Debe ser un número positivo.
    - cuentaId: (number) El ID de la cuenta origen o principal. Infiere si es posible, si no usa 1 como default.
    - descripcion: (string, opcional) Una breve descripción de la transacción.
    - fecha: (string, opcional) La fecha de la transacción (ej. "YYYY-MM-DD").
    - cuentaDestinoId: (number, opcional) El ID de la cuenta destino (solo para tipo "transferencia").

    Texto:
    ---
    {text}
    ---

    JSON extraído:
    """
    try:
        response = gemini_model.generate_content(prompt)
        # Clean the response to get only the JSON part
        json_text = response.text.strip().lstrip('```json').rstrip('```').strip()
        extracted_data = json.loads(json_text)
        return extracted_data
    except json.JSONDecodeError as e:
        print(f"Error decoding Gemini JSON response: {e}")
        print(f"Raw Gemini response: {response.text}")
        raise HTTPException(status_code=500, detail="La IA no devolvió un JSON válido.")
    except Exception as e:
        print(f"Error interacting with Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Error al comunicarse con la IA: {e}")

def validate_extracted_data(data: dict) -> (TransactionData | List[str]):
    """Validates the data extracted by Gemini against the Pydantic model."""
    required_fields = {"tipo", "monto", "cuentaId"}
    missing_fields = []

    # Check for nulls explicitly marked by Gemini for required fields
    for field in required_fields:
        if data.get(field) is None:
            missing_fields.append(field)

    if missing_fields:
        return missing_fields # Return list of missing fields

    try:
        # Attempt to parse and validate with Pydantic
        transaction = TransactionData(**data)
        return transaction # Return validated data object
    except ValidationError as e:
        # Extract missing fields from Pydantic's error details
        missing = [err['loc'][0] for err in e.errors() if err['type'] == 'missing']
        # Extract fields with other validation errors (e.g., wrong type, value error)
        invalid = [err['loc'][0] for err in e.errors() if err['type'] != 'missing']

        error_fields = list(set(missing + invalid + missing_fields)) # Combine all problematic fields
        print(f"Validation Error: {e.errors()}") # Log detailed errors
        return error_fields # Return list of problematic fields

def send_to_backend(data: TransactionData) -> bool:
    """Sends the validated transaction data to the backend."""
    try:
        response = requests.post(
            BACKEND_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=data.model_dump(exclude_none=True) # Send validated data, excluding None values
        )
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        print(f"Backend response: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to backend: {e}")
        # Consider more specific error handling based on status code if needed
        raise HTTPException(status_code=502, detail=f"Error al comunicarse con el backend: {e}")


# --- API Endpoint ---
@app.post("/process-document",
          response_model=TransactionData, # On success, expect TransactionData
          responses={
              400: {"model": ErrorResponse, "description": "Error de cliente (archivo inválido, datos faltantes)"},
              500: {"model": ErrorResponse, "description": "Error interno del servidor (OCR, IA, Validación)"},
              502: {"model": ErrorResponse, "description": "Error de comunicación con el backend"}
          })
async def process_document_endpoint(file: UploadFile = File(...)):
    """
    Recibe un archivo (imagen o PDF), extrae texto, usa IA para obtener
    datos de transacción, valida y envía al backend si los datos requeridos están completos.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No se envió ningún archivo.")

    contents = await file.read()
    filename = file.filename or ""
    content_type = file.content_type

    print(f"Processing file: {filename}, type: {content_type}")

    # --- Text Extraction ---
    extracted_text = ""
    if content_type in ["image/jpeg", "image/png", "image/gif"]:
        extracted_text = extract_text_from_image(contents)
    elif content_type == "application/pdf":
         extracted_text = extract_text_from_pdf(contents)
         # If using pdf2image, call that function instead
         # extracted_text = extract_text_from_pdf_using_images(contents)
    else:
        raise HTTPException(status_code=400, detail="Tipo de archivo no soportado. Use JPG, PNG, GIF o PDF.")

    if not extracted_text.strip():
         raise HTTPException(status_code=400, detail="No se pudo extraer texto del documento.")

    print(f"Extracted Text (first 200 chars): {extracted_text[:200]}...")

    # --- AI Information Extraction ---
    try:
        ai_data = extract_info_with_gemini(extracted_text)
        print(f"Data extracted by AI: {ai_data}")
    except HTTPException as e: # Catch exceptions from Gemini interaction
         return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e: # Catch unexpected errors
         print(f"Unexpected error during AI extraction: {e}")
         return JSONResponse(status_code=500, content={"error": "Error inesperado durante la extracción con IA."})


    # --- Validation ---
    validation_result = validate_extracted_data(ai_data)

    if isinstance(validation_result, list): # Validation failed, result is list of missing/invalid fields
        print(f"Validation failed. Missing/Invalid fields: {validation_result}")
        return JSONResponse(
            status_code=400,
            content={"error": "Faltan campos requeridos o son inválidos.", "missing_fields": validation_result}
        )

    # If validation passes, result is a TransactionData object
    validated_data: TransactionData = validation_result
    print(f"Validation successful: {validated_data}")

    """
    # --- Backend Integration ---
    try:
        success = send_to_backend(validated_data)
        if success:
            # Return the validated data upon successful backend post
            return validated_data
        else:
            # send_to_backend now raises HTTPException on failure
            # This part might not be reached if raise_for_status() is used effectively
             return JSONResponse(status_code=502, content={"error": "El backend no pudo procesar la solicitud."})
    except HTTPException as e: # Catch exceptions from backend communication
         return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e: # Catch unexpected errors during backend call
         print(f"Unexpected error during backend communication: {e}")
         return JSONResponse(status_code=500, content={"error": "Error inesperado durante la comunicación con el backend."})
    """
@app.post("/process-text"
          , response_model=TransactionData,
          responses={
              400: {"model": ErrorResponse, "description": "Error de cliente (texto inválido, datos faltantes)"},
              500: {"model": ErrorResponse, "description": "Error interno del servidor (IA, Validación)"},
              502: {"model": ErrorResponse, "description": "Error de comunicación con el backend"}
          })
async def process_text_endpoint(input_data: TextInput):
    """
    Recibe un texto, usa IA para obtener datos de transacción, valida y envía al backend
    si los datos requeridos están completos.
    """
    extracted_text = input_data.text
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No se envió texto o el texto está vacío.")

    # --- AI Information Extraction ---
    try:
        ai_data = extract_info_with_gemini(extracted_text)
        print(f"Data extracted by AI: {ai_data}")
    except HTTPException as e:
         return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
         print(f"Unexpected error during AI extraction: {e}")
         return JSONResponse(status_code=500, content={"error": "Error inesperado durante la extracción con IA."})

    # --- Validation ---
    validation_result = validate_extracted_data(ai_data)
    if isinstance(validation_result, list):
        print(f"Validation failed. Missing/Invalid fields: {validation_result}")
        return JSONResponse(
            status_code=400,
            content={"error": "Faltan campos requeridos o son inválidos.", "missing_fields": validation_result}
        )
    # If validation passes, result is a TransactionData object
    validated_data: TransactionData = validation_result
    print(f"Validation successful: {validated_data}")
    
    """
    # --- Backend Integration ---
    try:
        success = send_to_backend(validated_data)
        if success:
            # Return the validated data upon successful backend post
            return validated_data
        else:
            # send_to_backend now raises HTTPException on failure
            # This part might not be reached if raise_for_status() is used effectively
             return JSONResponse(status_code=502, content={"error": "El backend no pudo procesar la solicitud."})
    except HTTPException as e:
         return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e: # Catch unexpected errors during backend call
         print(f"Unexpected error during backend communication: {e}")
         return JSONResponse(status_code=500, content={"error": "Error inesperado durante la comunicación con el backend."})
    """
# --- Root endpoint for health check ---
@app.get("/")
def read_root():
    return {"message": "MCP AI Document Processor is running."}

# --- Run instruction (for local development) ---
# Use: uvicorn main:app --reload --port 8000