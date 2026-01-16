// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Flex, SearchField, TextFieldRef } from '@geti/ui';
import { Search } from '@geti/ui/icons';
import { clsx } from 'clsx';

import classes from './expandable-search.module.scss';

interface ExpandableSearchProps {
    value: string;
    onChange: (value: string) => void;
}

const focusInputRef = (ref: TextFieldRef<HTMLInputElement> | null) => {
    if (ref === null) return;

    ref.focus();
};

export const ExpandableSearch = ({ value, onChange }: ExpandableSearchProps) => {
    const [isExpanded, setIsExpanded] = useState(false);

    const handleToggle = () => {
        if (isExpanded && value) {
            onChange('');
        }

        setIsExpanded(!isExpanded);
    };

    const handleBlur = () => {
        if (!value) {
            setIsExpanded(false);
        }
    };

    return (
        <Flex UNSAFE_className={clsx(classes.container, { [classes.expanded]: isExpanded })}>
            {isExpanded ? (
                <SearchField
                    value={value}
                    ref={focusInputRef}
                    onChange={onChange}
                    onBlur={handleBlur}
                    placeholder='Search models...'
                    aria-label={'Search models'}
                    UNSAFE_className={classes.searchField}
                    width={'size-2400'}
                />
            ) : (
                <ActionButton isQuiet onPress={handleToggle} aria-label={'Search'}>
                    <Search />
                </ActionButton>
            )}
        </Flex>
    );
};
