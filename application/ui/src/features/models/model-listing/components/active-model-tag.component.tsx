// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Tag } from '@geti/ui';

import { ReactComponent as StartIcon } from '../../../../assets/icons/start.svg';

export const ActiveModelTag = () => {
    return (
        <Tag
            prefix={<StartIcon />}
            style={{
                backgroundColor: 'var(--energy-blue)',
                color: 'var(--spectrum-global-color-gray-50)',
                borderRadius: dimensionValue('size-50'),
                padding: `${dimensionValue('size-25')} ${dimensionValue('size-50')}`,
                fontWeight: 'normal',
            }}
            text={'Active'}
        />
    );
};
