from database.connection import (
    Base,
    engine
)

from database.user_model import User
from database.prediction_model import Prediction

print("Creating database tables...")

Base.metadata.create_all(
    bind=engine
)

print("Tables created successfully.")