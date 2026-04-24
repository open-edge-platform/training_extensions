// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CalendarDate, CalendarDateTime, DateValue } from '@internationalized/date';
import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc.js';

dayjs.extend(utc);

// When using `compare`, a negative result indicates that this date is before the given one,
// and a positive date indicates that it is after
export const isDateBetween = (
    date: DateValue,
    minDate: CalendarDate | CalendarDateTime,
    maxDate: CalendarDate | CalendarDateTime
): boolean => {
    return date.compare(minDate) >= 0 && date.compare(maxDate) <= 0;
};

export const formatLocalToUtc = (date: string, localFormat?: string): string =>
    dayjs(date, localFormat).utc(false).local().format();

export const replaceAllInBrackets = (text: string) => text.replace(/([\[(])(.+?)([\])])/g, '');
