# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import ctypes
import os
from ctypes import wintypes

# Constants
WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY = 4
WINHTTP_NO_PROXY_NAME = None
WINHTTP_NO_PROXY_BYPASS = None
WINHTTP_FLAG_SYNC = 0x00000000

WINHTTP_AUTOPROXY_AUTO_DETECT = 0x00000001
WINHTTP_AUTOPROXY_CONFIG_URL = 0x00000002
WINHTTP_AUTO_DETECT_TYPE_DHCP = 0x00000001
WINHTTP_AUTO_DETECT_TYPE_DNS_A = 0x00000002

# Hard cap (milliseconds) for WPAD auto-detection. Without this, WinHTTP uses the
# OS default which can block for 10-30s on a corporate VPN while it broadcasts
# DHCP option 252 / resolves wpad.<domain> / downloads and evaluates the PAC file.
WPAD_TIMEOUT_MS = 2000


class WINHTTP_AUTOPROXY_OPTIONS(ctypes.Structure):
    _fields_ = [
        ("dwFlags", wintypes.DWORD),
        ("dwAutoDetectFlags", wintypes.DWORD),
        ("lpszAutoConfigUrl", wintypes.LPWSTR),
        ("lpvReserved", wintypes.LPVOID),
        ("dwReserved", wintypes.DWORD),
        ("fAutoLogonIfChallenged", wintypes.BOOL),
    ]


class WINHTTP_PROXY_INFO(ctypes.Structure):
    _fields_ = [
        ("dwAccessType", wintypes.DWORD),
        ("lpszProxy", wintypes.LPWSTR),
        ("lpszProxyBypass", wintypes.LPWSTR),
    ]


class WINHTTP_CURRENT_USER_IE_PROXY_CONFIG(ctypes.Structure):
    _fields_ = [
        ("fAutoDetect", wintypes.BOOL),
        ("lpszAutoConfigUrl", wintypes.LPWSTR),
        ("lpszProxy", wintypes.LPWSTR),
        ("lpszProxyBypass", wintypes.LPWSTR),
    ]


WinHttpOpen = ctypes.windll.winhttp.WinHttpOpen  # type: ignore[attr-defined]
WinHttpOpen.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
WinHttpOpen.restype = wintypes.HANDLE

WinHttpCloseHandle = ctypes.windll.winhttp.WinHttpCloseHandle  # type: ignore[attr-defined]
WinHttpCloseHandle.argtypes = [wintypes.HANDLE]
WinHttpCloseHandle.restype = wintypes.BOOL

WinHttpSetTimeouts = ctypes.windll.winhttp.WinHttpSetTimeouts  # type: ignore[attr-defined]
WinHttpSetTimeouts.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
WinHttpSetTimeouts.restype = wintypes.BOOL

WinHttpGetProxyForUrl = ctypes.windll.winhttp.WinHttpGetProxyForUrl  # type: ignore[attr-defined]
WinHttpGetProxyForUrl.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCWSTR,
    ctypes.POINTER(WINHTTP_AUTOPROXY_OPTIONS),
    ctypes.POINTER(WINHTTP_PROXY_INFO),
]
WinHttpGetProxyForUrl.restype = wintypes.DWORD

WinHttpGetIEProxyConfigForCurrentUser = ctypes.windll.winhttp.WinHttpGetIEProxyConfigForCurrentUser  # type: ignore[attr-defined]
WinHttpGetIEProxyConfigForCurrentUser.argtypes = [ctypes.POINTER(WINHTTP_CURRENT_USER_IE_PROXY_CONFIG)]
WinHttpGetIEProxyConfigForCurrentUser.restype = wintypes.BOOL


def _autoproxy_resolve(url: str, auto_detect: bool, config_url: str | None) -> str | None:
    """Resolve a proxy via WPAD auto-detection and/or a PAC config URL, with a timeout."""
    hSession = WinHttpOpen(
        "PythonProxyResolver",
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        WINHTTP_FLAG_SYNC,
    )
    if not hSession:
        return None

    try:
        # Cap how long WPAD / PAC retrieval may block.
        WinHttpSetTimeouts(hSession, WPAD_TIMEOUT_MS, WPAD_TIMEOUT_MS, WPAD_TIMEOUT_MS, WPAD_TIMEOUT_MS)

        options = WINHTTP_AUTOPROXY_OPTIONS()
        options.dwFlags = 0
        if auto_detect:
            options.dwFlags |= WINHTTP_AUTOPROXY_AUTO_DETECT
            options.dwAutoDetectFlags = WINHTTP_AUTO_DETECT_TYPE_DHCP | WINHTTP_AUTO_DETECT_TYPE_DNS_A
        if config_url:
            options.dwFlags |= WINHTTP_AUTOPROXY_CONFIG_URL
            options.lpszAutoConfigUrl = config_url
        options.fAutoLogonIfChallenged = True

        proxy_info = WINHTTP_PROXY_INFO()
        result = WinHttpGetProxyForUrl(hSession, url, ctypes.byref(options), ctypes.byref(proxy_info))
        return proxy_info.lpszProxy if result == 1 else None
    finally:
        WinHttpCloseHandle(hSession)


def _detect_proxy(url: str) -> str | None:
    """Detect the proxy for ``url``.

    Prefers the user's already-configured proxy (static proxy or PAC URL) to avoid
    expensive WPAD discovery, and only falls back to timed auto-detection when the
    system is configured for it.
    """
    ie_config = WINHTTP_CURRENT_USER_IE_PROXY_CONFIG()
    has_ie_config = bool(WinHttpGetIEProxyConfigForCurrentUser(ctypes.byref(ie_config)))

    if has_ie_config:
        # A statically configured proxy needs no network round-trips.
        if ie_config.lpszProxy:
            return ie_config.lpszProxy
        # Honour an explicit PAC URL and/or auto-detect, but only when enabled.
        if ie_config.lpszAutoConfigUrl or ie_config.fAutoDetect:
            return _autoproxy_resolve(
                url,
                auto_detect=bool(ie_config.fAutoDetect),
                config_url=ie_config.lpszAutoConfigUrl,
            )
        # IE config present but no proxy configured at all: nothing to do.
        return None

    # No IE/system config available: best-effort timed auto-detection.
    return _autoproxy_resolve(url, auto_detect=True, config_url=None)


print("Setup Hook: Detecting proxy")
# Skip detection entirely when a proxy is already configured in the environment.
# This covers both user-provided HTTP(S)_PROXY and child/worker processes that
# inherit the value resolved by the parent, avoiding repeated WPAD discovery.
if os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"):
    print("Setup Hook: Proxy already set in environment, skipping detection")
else:
    proxy = _detect_proxy("https://huggingface.co")
    print("Setup Hook: Detected proxy: ", proxy)
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
