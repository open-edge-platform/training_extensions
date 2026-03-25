// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { DatasetStatistics } from './dataset-statistics.component';

describe('DatasetStatistics', () => {
    it('displays correct statistics when all items are annotated', async () => {
        render(<DatasetStatistics label='images' totalMediaItems={100} totalAnnotatedItems={33} />);

        expect(await screen.findByText('100')).toBeVisible();

        expect(screen.getByText('Annotated')).toBeVisible();
        expect(screen.getByText('33%')).toBeVisible();
        expect(screen.getByText('33 images')).toBeVisible();

        expect(screen.getByText('Unannotated')).toBeVisible();
        expect(screen.getByText('67%')).toBeVisible();
        expect(screen.getByText('67 images')).toBeVisible();
    });
});
