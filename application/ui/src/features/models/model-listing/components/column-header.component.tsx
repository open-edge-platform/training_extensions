// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';
import { SortDown } from '@geti/ui/icons';

export const ColumnHeader = ({ label, isSorted }: { label: string; isSorted?: boolean }) => (
    <Flex alignItems='center' gap='size-50'>
        <Text>{label}</Text>
        {isSorted && <SortDown width={16} height={16} />}
    </Flex>
);
