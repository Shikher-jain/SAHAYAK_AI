"""Shared slowapi Limiter instance.

Defined here (not in main.py) so routers can import `limiter` for
@limiter.limit(...) decorators without a circular import — main.py imports
routers, so routers can't import back from main.py.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)