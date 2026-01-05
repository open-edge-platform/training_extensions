// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, Suspense } from 'react';

import { IntelBrandedLoading, Toast } from '@geti/ui';
import { Outlet } from 'react-router';

import { $api } from '../../api/client';

const HealthCheckup = ({ children }: { children: ReactNode }) => {
    const { data } = $api.useQuery('get', '/health', undefined, {
        refetchInterval: (query) => {
            return query.state.data?.status === 'ok' ? false : 2000;
        },
    });

    if (data?.status === 'ok') {
        return children;
    }

    return <IntelBrandedLoading />;
};

export const RootLayout = () => {
    return (
        <Suspense fallback={<IntelBrandedLoading />}>
            <HealthCheckup>
                <Outlet />
                <Toast />
            </HealthCheckup>
        </Suspense>
    );
};
