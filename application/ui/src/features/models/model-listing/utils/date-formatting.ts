// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import dayjs from 'dayjs';

export const formatTrainingDateTime = (dateString: string | null | undefined): string => {
    if (!dateString) return '-';

    try {
        const date = dayjs(dateString);

        if (!date.isValid()) return '-';

        //      01 Oct 2025
        //      11:07 AM
        return date.format('DD MMM YYYY\nhh:mm A');
    } catch {
        return '-';
    }
};
