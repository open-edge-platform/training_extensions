import logging

from fastapi import APIRouter

from app.services import SystemService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/system/metrics/memory")
async def get_memory() -> dict:
    """Returns the used memory in MB and total available memory in MB."""
    used, total = SystemService().get_memory_usage()
    return {"used": int(used), "total": int(total)}
