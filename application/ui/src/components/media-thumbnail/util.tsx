// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import dayjs from 'dayjs';
import durationPlugin from 'dayjs/plugin/duration';

dayjs.extend(durationPlugin);

export const formatCompactDuration = (seconds: number) => {
    const handler = dayjs.duration(seconds, 'seconds');
    const format = handler.hours() > 0 ? 'HH:mm:ss' : 'mm:ss';

    return handler.format(format);
};
