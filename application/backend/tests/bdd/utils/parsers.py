# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator

from requests import Response


def parse_sse_events(response: Response) -> Iterator[dict]:
    """Parse Server-Sent Events from streaming response."""
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            data = line[6:]
            yield json.loads(data)
