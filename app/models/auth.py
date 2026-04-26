from pydantic import BaseModel, EmailStr
from typing import Optional, List

class SignupRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str

class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str

class ResendOTPRequest(BaseModel):
    email: EmailStr

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

# Premium UX Onboarding Schema
class OnboardingRequest(BaseModel):
    email: EmailStr  # To identify the user updating their profile
    role: str        # 'patient' or 'doctor'
    avatar_url: Optional[str] = None
    
    # Doctor Specific Fields
    specialty: Optional[str] = None
    license_number: Optional[str] = None
    clinic_name: Optional[str] = None
    
    # Patient Specific Fields
    age: Optional[int] = None
    gender: Optional[str] = None
    pre_existing_conditions: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    emergency_contact: Optional[str] = None
    health_goals: Optional[str] = None