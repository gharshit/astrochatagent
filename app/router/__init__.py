"""
Router module for all API endpoints.

Exports all routers for easy import in main.py:
- chat_router: Chat endpoints with session management
- kundali_router: Kundali generation endpoints
- session_router: Session retrieval endpoints
"""

from .chat_router import chat_router
from .kundali_router import kundali_router
from .session_router import session_router

__all__ = [
    "chat_router",
    "kundali_router",
    "session_router",
]

