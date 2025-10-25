# app/schemas.py

from pydantic import BaseModel, Field

class HeartInput(BaseModel):
    # Numeric-ish features (coerce types automatically)
    age: float = Field(..., description="Age in years")
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., description="Serum cholesterol (mg/dl)")
    thalach: float = Field(..., description="Maximum heart rate achieved")
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")

    # Categorical/integer-coded features
    sex: int = Field(..., description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., description="Chest pain type (0-3)")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg: int = Field(..., description="Resting electrocardiographic results (0-2)")
    exang: int = Field(..., description="Exercise induced angina (1 = yes; 0 = no)")
    slope: int = Field(..., description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: int = Field(..., description="Thalassemia (1/2/3 or dataset-specific coding)")

class PredictionOutput(BaseModel):
    predicted_class: str  # "presence" or "absence"
    probability: float    # probability of "presence" class (1)
