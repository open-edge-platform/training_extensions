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
    # lpszProxy / lpszProxyBypass are allocated by WinHttpGetProxyForUrl and must be released with
    # GlobalFree by the caller. They are kept as raw pointers (c_void_p) so the underlying buffer
    # can be freed after its value is copied into a Python string.
    _fields_ = [
        ("dwAccessType", wintypes.DWORD),
        ("lpszProxy", ctypes.c_void_p),
        ("lpszProxyBypass", ctypes.c_void_p),
    ]


class WINHTTP_CURRENT_USER_IE_PROXY_CONFIG(ctypes.Structure):
    # lpszAutoConfigUrl / lpszProxy / lpszProxyBypass are allocated by
    # WinHttpGetIEProxyConfigForCurrentUser and must likewise be released with GlobalFree.
    _fields_ = [
        ("fAutoDetect", wintypes.BOOL),
        ("lpszAutoConfigUrl", ctypes.c_void_p),
        ("lpszProxy", ctypes.c_void_p),
        ("lpszProxyBypass", ctypes.c_void_p),
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

GlobalFree = ctypes.windll.kernel32.GlobalFree  # type: ignore[attr-defined]
GlobalFree.argtypes = [wintypes.HGLOBAL]
GlobalFree.restype = wintypes.HGLOBAL


def _take_str(ptr: int | None) -> str | None:
    """Copy a wide string from a WinHTTP-allocated pointer and free it with GlobalFree.

    WinHTTP allocates the string fields of its proxy structures with GlobalAlloc; the caller
    owns them and must release them. This reads the value (if any) and always frees the buffer.

    Args:
        ptr: Address of the WinHTTP-allocated wide string, or ``None``/0 when not set.

    Returns:
        The decoded string, or ``None`` when the pointer is null.
    """
    if not ptr:
        return None
    try:
        return ctypes.wstring_at(ptr)
    finally:
        GlobalFree(ctypes.c_void_p(ptr))


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
        if result != 1:
            return None
        # Copy out the proxy and free both allocated buffers (bypass list is unused).
        proxy = _take_str(proxy_info.lpszProxy)
        _take_str(proxy_info.lpszProxyBypass)
        return proxy
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
        # Copy out and free all allocated strings up front to avoid leaking the WinHTTP buffers.
        auto_detect = bool(ie_config.fAutoDetect)
        ie_proxy = _take_str(ie_config.lpszProxy)
        ie_config_url = _take_str(ie_config.lpszAutoConfigUrl)
        _take_str(ie_config.lpszProxyBypass)

        # A statically configured proxy needs no network round-trips.
        if ie_proxy:
            return ie_proxy
        # Honour an explicit PAC URL and/or auto-detect, but only when enabled.
        if ie_config_url or auto_detect:
            return _autoproxy_resolve(url, auto_detect=auto_detect, config_url=ie_config_url)
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
