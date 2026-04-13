# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from typing import Any

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from loguru import logger

from app.models.webrtc import Answer, InputData, Offer

from .broadcaster import FrameBroadcaster
from .sdp_handler import SDPHandler
from .stream import InferenceVideoStreamTrack


@dataclass(frozen=True)
class WebRTCSettings:
    config: RTCConfiguration
    advertise_ip: str | None = None


class WebRTCManager:
    """Manager for handling WebRTC connections."""

    def __init__(self, frame_broadcaster: FrameBroadcaster, settings: WebRTCSettings, sdp_handler: SDPHandler) -> None:
        self._pcs: dict[str, RTCPeerConnection] = {}
        self._input_data: dict[str, Any] = {}
        self._frame_broadcaster = frame_broadcaster
        self._settings = settings
        self._sdp_handler = sdp_handler
        self._lock = asyncio.Lock()

    async def handle_offer(self, offer: Offer) -> Answer:
        """Create an SDP offer for a new WebRTC connection."""
        pc = RTCPeerConnection(configuration=self._settings.config)

        async with self._lock:
            self._pcs[offer.webrtc_id] = pc

        # Add video track
        stream_queue = self._frame_broadcaster.register(webrtc_id=offer.webrtc_id)
        track = InferenceVideoStreamTrack(stream_queue=stream_queue)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def connection_state_change() -> None:
            if pc.connectionState in ["failed", "closed"]:
                await self.cleanup_connection(offer.webrtc_id)

        # Set remote description from client's offer
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Mangle SDP if public IP is configured
        sdp = pc.localDescription.sdp
        if self._settings.advertise_ip:
            sdp = await self._sdp_handler.mangle_sdp(sdp, self._settings.advertise_ip)

        return Answer(sdp=sdp, type=pc.localDescription.type)

    def set_input(self, data: InputData) -> None:
        """Set input data for specific WebRTC connection"""
        self._input_data[data.webrtc_id] = {
            "conf_threshold": data.conf_threshold,
            "updated_at": asyncio.get_event_loop().time(),
        }

    async def cleanup_connection(self, webrtc_id: str) -> None:
        """Clean up a specific WebRTC connection by its ID."""
        async with self._lock:
            if webrtc_id in self._pcs:
                logger.debug("Cleaning up connection: {}", webrtc_id)
                pc = self._pcs.pop(webrtc_id)
                await pc.close()
                logger.debug("Connection {} successfully closed.", webrtc_id)
                self._input_data.pop(webrtc_id, None)
            self._frame_broadcaster.unregister(webrtc_id=webrtc_id)

    async def cleanup(self) -> None:
        """Clean up all connections"""
        async with self._lock:
            for webrtc_id, pc in self._pcs.items():
                await pc.close()
                self._frame_broadcaster.unregister(webrtc_id=webrtc_id)
            self._pcs.clear()
            self._input_data.clear()
            self._frame_broadcaster.cleanup()
