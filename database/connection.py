import os
import time

from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set.")

# Railway provides mysql://, SQLAlchemy needs mysql+pymysql://
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "mysql://",
        "mysql+pymysql://",
        1
    )

# ----------------------------------------------------
# Wait for Database (Docker)
# ----------------------------------------------------

MAX_RETRIES = 30
RETRY_DELAY = 2  # seconds

engine = None

for attempt in range(MAX_RETRIES):

    try:

        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True
        )

        connection = engine.connect()
        connection.close()

        print("✅ Database connected successfully.")

        break

    except OperationalError:

        print(
            f"⏳ Waiting for database... ({attempt + 1}/{MAX_RETRIES})"
        )

        time.sleep(RETRY_DELAY)

if engine is None:
    raise RuntimeError("❌ Could not connect to the database.")

# ----------------------------------------------------
# Session
# ----------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# ----------------------------------------------------
# Base
# ----------------------------------------------------

Base = declarative_base()