// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, DatePicker, Dialog, DialogTrigger, Flex, PressableElement, Text } from '@geti/ui';
import { DateValue, getLocalTimeZone, parseAbsoluteToLocal } from '@internationalized/date';
import dayjs from 'dayjs';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { isEmpty } from 'lodash-es';

import { FilterChips } from '../filter-chips/filter-chips.component';

import classes from './date-filter.module.scss';

const MIN_DATE = parseAbsoluteToLocal(dayjs('2020-01-30').startOf('d').toISOString());
const MAX_DATE = parseAbsoluteToLocal(dayjs('9999-11-30').endOf('d').toISOString());

const formatToLocalDate = (date: string) => dayjs(date).format('YYYY-MM-DD HH:mm:ss');

export const DateFilter = () => {
    const { startDate, endDate, setStartDate, setEndDate } = useDatasetFiltersSearchParams();

    const dates = [
        ...(startDate ? [{ id: 'startDate', name: `Start: ${formatToLocalDate(startDate)}` }] : []),
        ...(endDate ? [{ id: 'endDate', name: `End: ${formatToLocalDate(endDate)}` }] : []),
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

    const handleRemoveFilter = (id: string) => {
        if (id === 'startDate') {
            setStartDate(null);
        }
        if (id === 'endDate') {
            setEndDate(null);
        }
    };

    return (
        <DialogTrigger hideArrow type='popover'>
            <PressableElement aria-label='Filter by date'>
                <Flex
                    gap={'size-75'}
                    wrap={'wrap'}
                    minWidth={'size-2400'}
                    maxWidth={'size-5000'}
                    height={'size-400'}
                    alignItems={'center'}
                    UNSAFE_className={classes.filterContainer}
                >
                    {dates.map((date) => (
                        <FilterChips key={date.id} name={date.name} onClose={() => handleRemoveFilter(date.id)} />
                    ))}

                    {isEmpty(dates) && <Text UNSAFE_className={classes.searchPlaceholder}>Filter by upload date</Text>}
                </Flex>
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
                            defaultValue={startDate === null ? null : parseAbsoluteToLocal(startDate)}
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
