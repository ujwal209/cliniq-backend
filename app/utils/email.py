import os
import random
import string
import aiosmtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

async def generate_otp() -> str:
    """Generates a secure 6-digit OTP."""
    return ''.join(random.choices(string.digits, k=6))

async def send_otp_email(email: str, otp: str) -> bool:
    """
    Sends a real HTML email using Google SMTP.
    """
    sender_email = os.getenv("SMTP_USERNAME")
    sender_password = os.getenv("SMTP_PASSWORD")

    # Safety check: If credentials aren't set, fallback to printing in terminal
    if not sender_email or not sender_password:
        print("⚠️ SMTP credentials missing in .env! Printing OTP to terminal instead:")
        print(f"[{email}] OTP: {otp}")
        return True

    # Build the email
    message = EmailMessage()
    message["From"] = f"HealthSync AI <{sender_email}>"
    message["To"] = email
    message["Subject"] = "Your HealthSync Verification Code"

    # Premium HTML Email Template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f9fafb; padding: 40px 0;">
        <div style="max-w: 500px; margin: 0 auto; background-color: #ffffff; padding: 40px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;">
            <h2 style="color: #2563eb; margin-top: 0; font-size: 24px;">HealthSync</h2>
            <p style="color: #4b5563; font-size: 16px; line-height: 1.5; text-align: left;">
                Hello,
            </p>
            <p style="color: #4b5563; font-size: 16px; line-height: 1.5; text-align: left;">
                Use the verification code below to securely access your account. This code will expire in 10 minutes.
            </p>
            
            <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; padding: 20px; font-size: 32px; font-weight: bold; letter-spacing: 8px; border-radius: 12px; margin: 30px 0;">
                {otp}
            </div>
            
            <p style="color: #6b7280; font-size: 14px; text-align: left; border-top: 1px solid #e5e7eb; padding-top: 20px;">
                If you didn't request this email, you can safely ignore it.
            </p>
        </div>
    </body>
    </html>
    """

    # Set plain text fallback, then attach HTML
    message.set_content(f"Your HealthSync verification code is: {otp}")
    message.add_alternative(html_content, subtype="html")

    try:
        # Connect to Google SMTP and send
        await aiosmtplib.send(
            message,
            hostname="smtp.gmail.com",
            port=587,
            start_tls=True,
            username=sender_email,
            password=sender_password,
        )
        print(f"✅ Real OTP email successfully sent to {email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email to {email}. Error: {str(e)}")
        return False