// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, DatePicker, Dialog, DialogTrigger, Flex, PressableElement, Text } from '@geti-ui/ui';
import { getLocalTimeZone, parseAbsoluteToLocal, type DateValue } from '@internationalized/date';
import dayjs from 'dayjs';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { isEmpty } from 'lodash-es';

import { formatFilterDate } from '../../../../../shared/date-utils';

import classes from './date-filter.module.scss';

const MIN_DATE = parseAbsoluteToLocal(dayjs('2020-01-30').startOf('d').toISOString());
const MAX_DATE = parseAbsoluteToLocal(dayjs('9999-11-30').endOf('d').toISOString());

export const DateFilter = () => {
    const { startDate, endDate, setStartDate, setEndDate } = useDatasetFiltersSearchParams();

    const dates = [
        ...(startDate ? [{ id: 'startDate', name: `Start: ${formatFilterDate(startDate)}` }] : []),
        ...(endDate ? [{ id: 'endDate', name: `End: ${formatFilterDate(endDate)}` }] : []),
    ];

    const handleStartDateChange = (date: DateValue | null) => {
        if (date === null) {
            return;
        }

        setStartDate(date.toDate(getLocalTimeZone()).toISOString());
    };

    const handleEndDateChange = (date: DateValue | null) => {
        if (date === null) {
            return;
        }

        setEndDate(date.toDate(getLocalTimeZone()).toISOString());
    };

    return (
        <DialogTrigger hideArrow type='popover'>
            <PressableElement>
                <div role='button' aria-label='Filter by date'>
                    <Flex
                        gap={'size-75'}
                        wrap={'wrap'}
                        minWidth={'size-2400'}
                        maxWidth={'size-5000'}
                        height={'size-400'}
                        alignItems={'center'}
                        UNSAFE_className={classes.filterContainer}
                    >
                        {isEmpty(dates) ? (
                            <Text UNSAFE_className={classes.searchPlaceholder}>Filter by upload date</Text>
                        ) : (
                            <Text>{dates.map(({ name }) => name).join(', ')}</Text>
                        )}
                    </Flex>
                </div>
            </PressableElement>

            <Dialog maxWidth={'size-3600'} UNSAFE_className={classes.dialog} aria-label='Filter media items'>
                <Content>
                    <Flex direction='column' gap='size-200'>
                        <DatePicker
                            granularity={'second'}
                            width={'100%'}
                            label='Start date'
                            hourCycle={24}
                            minValue={MIN_DATE}
                            maxValue={endDate === null ? MAX_DATE : parseAbsoluteToLocal(endDate)}
                            value={startDate === null ? null : parseAbsoluteToLocal(startDate)}
                            onChange={handleStartDateChange}
                        />
                        <DatePicker
                            granularity={'second'}
                            width={'100%'}
                            label='End date'
                            hourCycle={24}
                            minValue={startDate === null ? MIN_DATE : parseAbsoluteToLocal(startDate)}
                            maxValue={MAX_DATE}
                            value={endDate === null ? null : parseAbsoluteToLocal(endDate)}
                            onChange={handleEndDateChange}
                        />
                    </Flex>
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
