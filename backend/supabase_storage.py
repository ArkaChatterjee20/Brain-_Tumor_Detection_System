import os

from dotenv import load_dotenv
from supabase import create_client


# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SUPABASE_BUCKET = os.getenv(
    "SUPABASE_BUCKET",
    "prediction-files"
)
print("SUPABASE_URL:", repr(os.getenv("SUPABASE_URL")))
print("SUPABASE_BUCKET:", repr(os.getenv("SUPABASE_BUCKET")))
import socket

try:
    hostname = SUPABASE_URL.replace("https://", "").split("/")[0]
    print("Testing Supabase hostname:", hostname)
    print("Resolved IP:", socket.gethostbyname(hostname))
except Exception as e:
    print("SUPABASE DNS ERROR:", repr(e))


# ----------------------------------------------------
# Validate Configuration
# ----------------------------------------------------

if not SUPABASE_URL:
    raise RuntimeError(
        "SUPABASE_URL is not configured."
    )

if not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_KEY is not configured."
    )


# ----------------------------------------------------
# Create Supabase Client
# ----------------------------------------------------

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ----------------------------------------------------
# Upload File
# ----------------------------------------------------

def upload_file(
    local_file_path,
    storage_path,
    content_type
):

    if not os.path.exists(local_file_path):

        raise FileNotFoundError(
            f"File not found: {local_file_path}"
        )

    with open(
        local_file_path,
        "rb"
    ) as file:

        file_data = file.read()

    response = (
        supabase.storage
        .from_(SUPABASE_BUCKET)
        .upload(
            storage_path,
            file_data,
            {
                "content-type": content_type,
                "upsert": "true"
            }
        )
    )

    print(
        "SUPABASE UPLOAD RESPONSE:",
        response
    )

    print(
        "SUPABASE STORAGE PATH:",
        storage_path
    )

    return storage_path


# ----------------------------------------------------
# Create Signed URL
# ----------------------------------------------------

def create_signed_url(
    storage_path,
    expires_in=3600
):

    if not storage_path:

        print(
            "WARNING: Empty Supabase storage path."
        )

        return None

    print(
        "Creating signed URL for:",
        storage_path
    )

    response = (
        supabase.storage
        .from_(SUPABASE_BUCKET)
        .create_signed_url(
            storage_path,
            expires_in
        )
    )

    print(
        "SUPABASE SIGNED URL RESPONSE:",
        response
    )


    # ------------------------------------------------
    # Dictionary response
    # ------------------------------------------------

    if isinstance(response, dict):

        if response.get("signedURL"):
            return response["signedURL"]

        if response.get("signedUrl"):
            return response["signedUrl"]

        if response.get("signed_url"):
            return response["signed_url"]


        data = response.get("data")

        if isinstance(data, dict):

            if data.get("signedURL"):
                return data["signedURL"]

            if data.get("signedUrl"):
                return data["signedUrl"]

            if data.get("signed_url"):
                return data["signed_url"]


    # ------------------------------------------------
    # Object response
    # ------------------------------------------------

    if hasattr(response, "signed_url"):

        return response.signed_url

    if hasattr(response, "signedURL"):

        return response.signedURL

    if hasattr(response, "signedUrl"):

        return response.signedUrl


    raise RuntimeError(
        f"Unable to create signed URL: {response}"
    )