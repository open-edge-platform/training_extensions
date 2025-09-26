// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ReactComponent as Image } from '../../../assets/icons/images-folder.svg';
import { DisclosureGroup } from '../sources/disclosure-group.component';
import { Folder } from './folder.component';
import { Webhook } from './webhook.component';

const inputs = [
    { label: 'Folder', value: 'folder', content: <Folder />, icon: <Image width={'24px'} /> },
    { label: 'Webhook', value: 'webhook', content: <Webhook />, icon: <Image width={'24px'} /> },
];

export const SinkOptions = () => {
    const [activeInput, setActiveInput] = useState<string | null>(null);

    const handleActiveInputChange = (value: string) => {
        setActiveInput((prevValue) => (value !== prevValue ? value : null));
    };

    return <DisclosureGroup items={inputs} value={activeInput} onChange={handleActiveInputChange} />;
};
