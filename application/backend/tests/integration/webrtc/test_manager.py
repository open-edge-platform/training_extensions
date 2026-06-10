# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import socket

import numpy as np
import pytest
from aiortc import RTCConfiguration

from app.models.webrtc import InputData, Offer
from app.webrtc import SDPHandler
from app.webrtc.broadcaster import FrameBroadcaster
from app.webrtc.manager import WebRTCManager, WebRTCSettings

VALID_SDP = (
    "v=0\n"
    "o=- 0 0 IN IP4 127.0.0.1\n"
    "s=-\n"
    "t=0 0\n"
    "m=video 9 UDP/TLS/RTP/SAVPF 96\n"
    "c=IN IP4 0.0.0.0\n"
    "a=rtpmap:96 VP8/90000\n"
    "a=ice-ufrag:someufrag\n"
    "a=ice-pwd:somepassword\n"
    "a=setup:actpass\n"
    "a=mid:0\n"
    "a=rtcp-mux\n"
    "a=recvonly\n"
)


@pytest.fixture
def fxt_frame_broadcaster():
    return FrameBroadcaster[np.ndarray]()


@pytest.fixture
def fxt_settings():
    return WebRTCSettings(config=RTCConfiguration(iceServers=[]))


@pytest.fixture
def fxt_sdp_handler():
    return SDPHandler()


@pytest.fixture
def fxt_manager(fxt_frame_broadcaster, fxt_settings, fxt_sdp_handler):
    return WebRTCManager(fxt_frame_broadcaster, fxt_settings, fxt_sdp_handler)


@pytest.fixture
def fxt_offer():
    return Offer(webrtc_id="test_id", sdp=VALID_SDP, type="offer")


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class TestWebRTCManager:
    @pytest.mark.asyncio
    async def test_handle_offer_creates_connection(self, fxt_manager, fxt_offer):
        answer = await fxt_manager.handle_offer(fxt_offer)
        assert get_local_ip() in answer.sdp
        assert answer.type == "answer"
        assert "test_id" in fxt_manager._pcs
        assert fxt_manager._frame_broadcaster.is_registered("test_id")

    @pytest.mark.asyncio
    async def test_handle_offer_with_host_resolution(self, fxt_frame_broadcaster, fxt_sdp_handler, fxt_offer):
        settings = WebRTCSettings(config=RTCConfiguration(iceServers=[]), advertise_ip="localhost")
        manager = WebRTCManager(fxt_frame_broadcaster, settings, fxt_sdp_handler)
        answer = await manager.handle_offer(fxt_offer)
        assert "127.0.0.1" in answer.sdp
        assert answer.type == "answer"
        assert "test_id" in manager._pcs
        assert manager._frame_broadcaster.is_registered("test_id")

    @pytest.mark.asyncio
    async def test_cleanup_connection_removes_pc(self, fxt_manager, fxt_offer):
        await fxt_manager.handle_offer(fxt_offer)
        assert "test_id" in fxt_manager._pcs
        assert fxt_manager._frame_broadcaster.is_registered("test_id")
        await fxt_manager.cleanup_connection("test_id")
        assert "test_id" not in fxt_manager._pcs
        assert not fxt_manager._frame_broadcaster.is_registered("test_id")

    @pytest.mark.asyncio
    async def test_reoffer_same_id_keeps_new_connection_registered(self, fxt_manager, fxt_offer):
        """Re-offering with the same webrtc_id must keep the *new* connection's consumer registered.

        A stale "closed" handler from the replaced connection must not unregister the
        broadcaster consumer of its successor (which would freeze the stream).
        """
        await fxt_manager.handle_offer(fxt_offer)
        first_pc = fxt_manager._pcs["test_id"]

        # Re-offer with the same id -> replaces the previous connection
        await fxt_manager.handle_offer(fxt_offer)
        second_pc = fxt_manager._pcs["test_id"]
        assert second_pc is not first_pc

        # Simulate the replaced connection's delayed "closed" handler firing.
        await fxt_manager.cleanup_connection("test_id", pc=first_pc)

        # The new connection and its consumer must still be alive.
        assert fxt_manager._pcs.get("test_id") is second_pc
        assert fxt_manager._frame_broadcaster.is_registered("test_id")

    def test_set_input_stores_data(self, fxt_manager):
        data = InputData(webrtc_id="test_id", conf_threshold=0.5)
        fxt_manager.set_input(data)
        assert fxt_manager._input_data["test_id"]["conf_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_cleanup_removes_all(self, fxt_manager, fxt_offer):
        await fxt_manager.handle_offer(fxt_offer)
        await fxt_manager.handle_offer(Offer(webrtc_id="id2", sdp=VALID_SDP, type="offer"))
        await fxt_manager.cleanup()
        assert not fxt_manager._pcs
        assert not fxt_manager._input_data
        assert fxt_manager._frame_broadcaster.consumer_count() == 0
