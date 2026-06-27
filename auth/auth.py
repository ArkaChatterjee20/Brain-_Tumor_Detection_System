from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from sqlalchemy.orm import Session

from database.connection import SessionLocal

from database.user_service import (
    create_user,
    get_user_by_email
)

from auth.schemas import (
    UserCreate,
    UserLogin
)

from auth.hashing import (
    hash_password,
    verify_password
)

from auth.token import (
    create_access_token
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()


@router.post("/register")
def register(
    user: UserCreate,
    db: Session = Depends(get_db)
):

    try:

        existing_user = get_user_by_email(
            db,
            user.email
        )

        if existing_user:

            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )

        hashed_password = hash_password(
            user.password
        )

        new_user = create_user(

            db=db,

            username=user.username,

            email=user.email,

            password=hashed_password

        )

        return {

            "message":
            "User registered successfully",

            "user_id":
            new_user.id

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/login")
def login(
    user: UserLogin,
    db: Session = Depends(get_db)
):

    try:

        print("\n========== LOGIN ATTEMPT ==========")
        print("Email entered:", user.email)

        db_user = get_user_by_email(
            db,
            user.email
        )

        print("Database user:", db_user)

        if db_user is None:

            raise HTTPException(

                status_code=401,

                detail="Invalid email"

            )

        password_verified = verify_password(

            user.password,

            db_user.password

        )

        print("Password verified:", password_verified)

        if not password_verified:

            raise HTTPException(

                status_code=401,

                detail="Invalid password"

            )

        access_token = create_access_token(

            data={
                "sub": db_user.email
            }

        )

        print("Login successful")

        return {

            "access_token":
            access_token,

            "token_type":
            "bearer"

        }

    except HTTPException:

        raise

    except Exception as e:

        print("LOGIN ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )