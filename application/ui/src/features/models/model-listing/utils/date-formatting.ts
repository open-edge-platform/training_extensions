// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const formatTrainingDateTime = (dateString: string | null | undefined): string => {
    if (!dateString) return '-';

    try {
        const date = new Date(new Date());

        const day = date.getDate().toString().padStart(2, '0');
        const month = date.toLocaleDateString('en-GB', { month: 'short' });
        const year = date.getFullYear();

        const hours = date.getHours();
        const minutes = date.getMinutes().toString().padStart(2, '0');
        const period = hours >= 12 ? 'PM' : 'AM';
        const hour12 = hours % 12 || 12;
        const formattedHour = hour12.toString().padStart(2, '0');

        //      01 oct 2025
        //      11:07am
        return `${day} ${month} ${year}\n${formattedHour}:${minutes} ${period}`;
    } catch {
        return '-';
    }
};
