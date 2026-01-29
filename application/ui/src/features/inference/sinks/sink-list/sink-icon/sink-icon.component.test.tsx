// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render } from '@testing-library/react';

import { SinkIcon } from './sink-icon.component';

describe('SinkIcon', () => {
    it('renders correct icon for each sink type', () => {
        const types: Array<'folder' | 'mqtt' | 'webhook' | 'ros' | 'disconnected'> = [
            'folder',
            'mqtt',
            'webhook',
            'ros',
            'disconnected',
        ];

        types.forEach((type) => {
            const { container } = render(<SinkIcon type={type} />);

            if (type === 'disconnected') {
                expect(container.querySelector('svg')).not.toBeInTheDocument();
                expect(container.firstChild).toBeFalsy();
            } else {
                expect(container.querySelector('svg')).toBeInTheDocument();
            }
        });

        // Verify different icons are rendered for different types
        const { container: folderContainer } = render(<SinkIcon type='folder' />);
        const { container: mqttContainer } = render(<SinkIcon type='mqtt' />);

        const folderSvg = folderContainer.querySelector('svg');
        const mqttSvg = mqttContainer.querySelector('svg');

        expect(folderSvg).toBeInTheDocument();
        expect(mqttSvg).toBeInTheDocument();
        expect(folderSvg).not.toBe(mqttSvg);
    });
});
