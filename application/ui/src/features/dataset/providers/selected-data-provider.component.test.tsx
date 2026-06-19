// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitForElementToBeRemoved } from '@testing-library/react';
import { render } from 'test-utils/render';

import { SelectedDataProvider, useSelectedData } from './selected-data-provider.component';

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

describe('isSelected', () => {
    it('returns false for a key not in the selection', () => {
        const App = () => {
            const { isSelected } = useSelectedData();
            return <p>{isSelected('absent-key') ? 'selected' : 'not selected'}</p>;
        };

        render(
            <SelectedDataProvider>
                <App />
            </SelectedDataProvider>
        );

        expect(screen.getByText('not selected')).toBeInTheDocument();
    });

    it('returns true after a key is added via toggleSelectedKeys', async () => {
        const testKey = 'key-abc';
        const App = () => {
            const { isSelected, toggleSelectedKeys } = useSelectedData();
            return (
                <div>
                    <p>{isSelected(testKey) ? 'selected' : 'not selected'}</p>
                    <button onClick={() => toggleSelectedKeys([testKey])}>toggle</button>
                </div>
            );
        };

        render(
            <SelectedDataProvider>
                <App />
            </SelectedDataProvider>
        );

        expect(screen.getByText('not selected')).toBeInTheDocument();
        screen.getByRole('button', { name: 'toggle' }).click();
        expect(await screen.findByText('selected')).toBeInTheDocument();
    });

    it('returns false after a key is removed via toggleSelectedKeys', async () => {
        const testKey = 'key-to-remove';
        const App = () => {
            const { isSelected, toggleSelectedKeys } = useSelectedData();
            return (
                <div>
                    <p>{isSelected(testKey) ? 'selected' : 'not selected'}</p>
                    <button onClick={() => toggleSelectedKeys([testKey])}>toggle</button>
                </div>
            );
        };

        render(
            <SelectedDataProvider>
                <App />
            </SelectedDataProvider>
        );

        screen.getByRole('button', { name: 'toggle' }).click();
        await screen.findByText('selected');

        screen.getByRole('button', { name: 'toggle' }).click();
        await screen.findByText('not selected');
    });
});
