from sqlalchemy.orm import Session

from database.user_model import User


def create_user(
    db: Session,
    username: str,
    email: str,
    password: str
):
    user = User(
        username=username,
        email=email,
        password=password
    )

    db.add(user)

    db.commit()

    db.refresh(user)

    return user


def get_user_by_email(
    db: Session,
    email: str
):
    return (
        db.query(User)
        .filter(User.email == email)
        .first()
    )


def get_user_by_username(
    db: Session,
    username: str
):
    return (
        db.query(User)
        .filter(User.username == username)
        .first()
    )


def get_user_by_id(
    db: Session,
    user_id: int
):
    return (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )


def get_all_users(
    db: Session
):
    return (
        db.query(User)
        .all()
    )


def delete_user(
    db: Session,
    user_id: int
):
    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if user:
        db.delete(user)
        db.commit()

    return user