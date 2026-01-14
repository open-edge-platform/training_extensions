# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""WebRTC API Endpoints"""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from loguru import logger

from app.api.dependencies import get_ice_servers
from app.api.dependencies import get_webrtc_manager as get_webrtc
from app.api.schemas import WebRTCConfigResponse, WebRTCIceServer
from app.models.webrtc import Answer, InputData, Offer
from app.webrtc.manager import WebRTCManager

router = APIRouter(prefix="/api/webrtc", tags=["WebRTC"])


@router.post(
    "/offer",
    response_model=Answer,
    responses={status.HTTP_200_OK: {"description": "WebRTC Answer"}},
)
async def create_webrtc_offer(offer: Offer, webrtc_manager: Annotated[WebRTCManager, Depends(get_webrtc)]) -> Answer:
    """Create a WebRTC offer"""
    try:
        return await webrtc_manager.handle_offer(offer)
    except Exception as e:
        logger.exception("Error processing WebRTC offer")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/input_hook",
    responses={
        status.HTTP_200_OK: {"description": "WebRTC input data updated"},
    },
)
async def webrtc_input_hook(data: InputData, webrtc_manager: Annotated[WebRTCManager, Depends(get_webrtc)]) -> None:
    """Update webrtc input with user data"""
    webrtc_manager.set_input(data)


@router.get(
    path="/config",
    response_model=WebRTCConfigResponse,
    responses={status.HTTP_200_OK: {"description": "WebRTC configuration"}},
)
async def get_webrtc_config(ice_servers: Annotated[list[dict], Depends(get_ice_servers)]) -> WebRTCConfigResponse:
    """Get WebRTC configuration including ICE servers"""
    servers = [WebRTCIceServer(**server) for server in ice_servers]
    return WebRTCConfigResponse(iceServers=servers)
