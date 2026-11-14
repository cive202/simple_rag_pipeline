from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import Booking
from app.repositories.booking import create_booking_record

_EMAIL_REGEX = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def create_booking(
    db: Session, full_name: str, email: str, date: str, time: str
) -> Booking:
    full_name_clean = full_name.strip()
    email_clean = email.strip().lower()
    date_clean = date.strip()
    time_clean = time.strip()

    if len(full_name_clean.split()) < 2:
        raise ValueError("full_name must include first and last name")
    if not _EMAIL_REGEX.match(email_clean):
        raise ValueError("email is not valid")
    try:
        datetime.strptime(date_clean, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("date must be in YYYY-MM-DD format") from exc
    try:
        datetime.strptime(time_clean, "%H:%M")
    except ValueError as exc:
        raise ValueError("time must be in HH:MM format") from exc

    return create_booking_record(
        db,
        full_name=full_name_clean,
        email=email_clean,
        date=date_clean,
        time=time_clean,
    )
