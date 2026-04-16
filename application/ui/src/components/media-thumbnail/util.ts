// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import dayjs from 'dayjs';
import durationPlugin from 'dayjs/plugin/duration';

dayjs.extend(durationPlugin);

export const formatCompactDuration = (seconds: number) => {
    const handler = dayjs.duration(seconds, 'seconds');
    const totalHours = Math.floor(handler.asHours());

    if (totalHours === 0) {
        return handler.format('mm:ss');
    }

    const mm = String(handler.minutes()).padStart(2, '0');
    const ss = String(handler.seconds()).padStart(2, '0');
    return `${String(totalHours).padStart(2, '0')}:${mm}:${ss}`;
};
