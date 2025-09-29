// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitForElementToBeRemoved } from '@test-utils/render';

import { SelectedDataProvider, useSelectedData } from './provider';

describe('SelectedDataProvider', () => {
    it('toggles a selected key and displays it', async () => {
        const testKey = 'test-key';
        const App = () => {
            const { selectedKeys, toggleSelectedKeys } = useSelectedData();

            return (
                <div>
                    <p>{selectedKeys === 'all' ? 'all' : [...selectedKeys.values()].join(', ')}</p>
                    <button onClick={() => toggleSelectedKeys([testKey])}>toggle key</button>
                </div>
            );
        };

        render(
            <SelectedDataProvider>
                <App />
            </SelectedDataProvider>
        );

        expect(screen.queryByText(testKey)).not.toBeInTheDocument();

        screen.getByRole('button', { name: /toggle key/i }).click();
        expect(await screen.findByText(testKey)).toBeVisible();

        screen.getByRole('button', { name: /toggle key/i }).click();
        await waitForElementToBeRemoved(() => screen.queryByText(testKey));
        expect(screen.queryByText(testKey)).not.toBeInTheDocument();
    });
});
