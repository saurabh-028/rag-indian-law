"""
firebase_auth.py — Verifies Firebase ID tokens sent from the frontend.

The frontend signs in with Firebase (Google or email/password), then sends
the resulting ID token as `Authorization: Bearer <token>` on requests that
should be tied to a user. Verification happens against Firebase's own
public keys via the Admin SDK — this app never sees or stores a password.

Environment variables:
    FIREBASE_SERVICE_ACCOUNT_PATH — path to the service account JSON
        (default: ./firebase-service-account.json)
"""

import logging
import os

import firebase_admin
from fastapi import Header, HTTPException
from firebase_admin import auth as firebase_auth_sdk
from firebase_admin import credentials

logger = logging.getLogger(__name__)

_SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH", "./firebase-service-account.json"
)

_app = None
if os.path.exists(_SERVICE_ACCOUNT_PATH):
    _cred = credentials.Certificate(_SERVICE_ACCOUNT_PATH)
    _app = firebase_admin.initialize_app(_cred)
    logger.info("Firebase Admin SDK initialized from %s", _SERVICE_ACCOUNT_PATH)
else:
    logger.warning(
        "Firebase service account not found at %s — auth endpoints will reject all requests.",
        _SERVICE_ACCOUNT_PATH,
    )


class FirebaseUser:
    def __init__(self, uid: str, email: str | None):
        self.uid = uid
        self.email = email


async def get_current_user(authorization: str | None = Header(default=None)) -> FirebaseUser | None:
    """Optional-auth dependency: returns the verified user if a valid
    `Authorization: Bearer <token>` header is present, otherwise None.
    Never raises for a *missing* header — most of this app's endpoints work
    fine anonymously; this only identifies who's asking when they're signed in."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    if _app is None:
        return None

    token = authorization.removeprefix("Bearer ").strip()
    try:
        decoded = firebase_auth_sdk.verify_id_token(token, app=_app)
    except Exception as exc:  # noqa: BLE001 — any verification failure just means "not signed in"
        logger.warning("Firebase token verification failed: %s", exc)
        return None

    return FirebaseUser(uid=decoded["uid"], email=decoded.get("email"))


async def require_current_user(authorization: str | None = Header(default=None)) -> FirebaseUser:
    """Required-auth dependency, for endpoints that only make sense signed in
    (e.g. syncing the user record). Raises 401 if there's no valid token."""
    user = await get_current_user(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Sign-in required.")
    return user
