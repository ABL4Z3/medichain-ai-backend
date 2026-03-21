from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import List

app = FastAPI()
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env into the system
# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')
# demo 2
# Data structure matching your Firebase model
class PatientData(BaseModel):
    name: str
    age: int
    bloodGroup: str
    allergies: List[str]
    currentMedications: List[str]
    diagnoses: List[str]

@app.post("/summarize")
async def get_summary(patient: PatientData):
    prompt = f"""
    You are a clinical decision support AI for Indian doctors.
    Analyze this patient profile and give a structured summary.
    
    Patient: {patient.name}, Age: {patient.age}
    Blood Group: {patient.bloodGroup}
    Known Allergies: {", ".join(patient.allergies)}
    Current Medications: {", ".join(patient.currentMedications)}
    Active Diagnoses: {", ".join(patient.diagnoses)}
    
    Respond with EXACTLY 5 points:
    1. CRITICAL ALERTS (allergies or dangerous drug combinations)
    2. Current Health Status (1 sentence)
    3. Drug Interaction Warning (check current medications)
    4. Overdue Preventive Tests (based on diagnoses and age)
    5. Recommended Focus for This Consultation
    
    Use simple medical language. Flag serious risks in CAPS.
    """
    
    try:
        response = model.generate_content(prompt)
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "MediChain AI Engine Online"}


if __name__ == "__main__":
    import uvicorn
    import os
    # Railway provides the PORT as an environment variable
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port)
