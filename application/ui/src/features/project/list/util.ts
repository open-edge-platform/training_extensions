// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import dayjs from 'dayjs';

export const formatCreationDate = (creationDate: string) => {
    return dayjs(creationDate).format('D MMMM YYYY | h:mm A');
};
