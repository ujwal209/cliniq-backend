import bcrypt
import jwt
from datetime import datetime, timedelta
import os

SECRET_KEY = os.getenv("SECRET_KEY", "fallback_dev_key_change_me_later")
ALGORITHM = "HS256"

def get_password_hash(password: str) -> str:
    """Hashes a password using pure bcrypt."""
    # bcrypt requires bytes, so we encode the string
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(pwd_bytes, salt)
    
    # Decode back to a normal string so MongoDB can store it easily
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the hashed version."""
    password_bytes = plain_password.encode('utf-8')
    hashed_password_bytes = hashed_password.encode('utf-8')
    
    return bcrypt.checkpw(password_bytes, hashed_password_bytes)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(days=7)):
    """Creates the JWT session token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)