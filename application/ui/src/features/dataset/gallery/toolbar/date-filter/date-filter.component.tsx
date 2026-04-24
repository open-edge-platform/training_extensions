// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, DatePicker, Dialog, DialogTrigger, Flex, PressableElement, Text } from '@geti/ui';
import { DateValue, parseDateTime } from '@internationalized/date';
import dayjs from 'dayjs';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { isEmpty } from 'lodash-es';

import { FilterChips } from '../filter-chips/filter-chips.component';
import { formatLocalToUtc, isDateBetween, replaceAllInBrackets } from './util';

import classes from './date-filter.module.scss';

const MIN_DATE = parseDateTime('2020-01-30');
const MAX_DATE = parseDateTime('9999-11-30');

const formatToLocalDate = (date: string) => dayjs(date).format('DD MMMM YYYY');

export const DateFilter = () => {
    const { startDate, endDate, setStartDate, setEndDate } = useDatasetFiltersSearchParams();

    const filteredLabels = [
        ...(startDate ? [{ id: 'startDate', name: `Start date: ${formatToLocalDate(startDate)}` }] : []),
        ...(endDate ? [{ id: 'endDate', name: `End date: ${formatToLocalDate(endDate)}` }] : []),
    ];

    const onValidDateChange = (callback: (date: string) => void) => (newValue: DateValue | null) => {
        if (newValue === null) {
            return;
        }

        const isValidDate = isDateBetween(newValue, MIN_DATE, MAX_DATE);
        isValidDate && callback(formatLocalToUtc(replaceAllInBrackets(newValue.toString())));
    };

    const onRemoveFilter = (id: string) => {
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
                    gap={'size-40'}
                    wrap={'wrap'}
                    width={'size-3000'}
                    height={'size-400'}
                    alignItems={'center'}
                    UNSAFE_className={classes.filterContainer}
                >
                    {filteredLabels.map((label) => (
                        <FilterChips key={label.id} name={label.name} onClose={() => onRemoveFilter(label.id)} />
                    ))}

                    {isEmpty(filteredLabels) && (
                        <Text UNSAFE_className={classes.searchPlaceholder}>Filter by upload date</Text>
                    )}
                </Flex>
            </PressableElement>

            <Dialog maxWidth={'size-3600'} UNSAFE_className={classes.dialog} aria-label='Filter media items'>
                <Content>
                    <Flex direction='column' gap='size-200'>
                        <DatePicker width={'100%'} label='Start date' onChange={onValidDateChange(setStartDate)} />
                        <DatePicker width={'100%'} label='End date' onChange={onValidDateChange(setEndDate)} />
                    </Flex>
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
