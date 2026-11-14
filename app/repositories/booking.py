from sqlalchemy.orm import Session

from app.db.models import Booking


def create_booking_record(
    db: Session, *, full_name: str, email: str, date: str, time: str
) -> Booking:
    booking = Booking(full_name=full_name, email=email, date=date, time=time)
    db.add(booking)
    db.commit()
    db.refresh(booking)
    return booking
