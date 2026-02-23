// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import { CircularProgress } from './circular-progress.component';

const renderApp = (percentage: number) => {
    return render(<CircularProgress percentage={percentage} />);
};

describe('CircularProgress', () => {
    it('negative number', () => {
        renderApp(-10);
        expect(screen.getByText('0%')).toBeVisible();
    });

    it('int number', () => {
        renderApp(10);
        expect(screen.getByText('10%')).toBeVisible();
    });

    it('multiple decimals', () => {
        renderApp(21.6666666);
        expect(screen.getByText('21%')).toBeVisible();
    });

    it('single decimal', () => {
        renderApp(20.0);
        expect(screen.getByText('20%')).toBeVisible();
    });

    it('render "0"', () => {
        renderApp(0.5);
        expect(screen.getByText('0%')).toBeVisible();
    });
});
