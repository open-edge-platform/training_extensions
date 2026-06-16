// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, DatePicker, Dialog, DialogTrigger, Flex, PressableElement, Text } from '@geti/ui';
import { DateValue, parseDate } from '@internationalized/date';
import dayjs from 'dayjs';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { isEmpty } from 'lodash-es';

import { FilterChips } from '../filter-chips/filter-chips.component';

import classes from './date-filter.module.scss';

const MIN_DATE = parseDate('2020-01-30');
const MAX_DATE = parseDate('9999-11-30');

const formatToLocalDate = (date: string) => dayjs(date).format('DD/MM/YYYY');
const toCalendarDate = (date: string) => parseDate(dayjs(date).format('YYYY-MM-DD'));

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

        setStartDate(date.toString());
    };

    const handleEndDateChange = (date: DateValue | null) => {
        if (date === null) {
            return;
        }
        const endOfDay = dayjs(date.toString()).endOf('day').format('YYYY-MM-DDTHH:mm:ss');

        setEndDate(endOfDay);
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
                    width={'size-3400'}
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
                            width={'100%'}
                            label='Start date'
                            minValue={MIN_DATE}
                            maxValue={endDate === null ? MAX_DATE : toCalendarDate(endDate)}
                            defaultValue={startDate === null ? null : toCalendarDate(startDate)}
                            onChange={handleStartDateChange}
                        />
                        <DatePicker
                            width={'100%'}
                            label='End date'
                            minValue={startDate === null ? MIN_DATE : toCalendarDate(startDate)}
                            maxValue={MAX_DATE}
                            value={endDate === null ? null : toCalendarDate(endDate)}
                            onChange={handleEndDateChange}
                        />
                    </Flex>
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
