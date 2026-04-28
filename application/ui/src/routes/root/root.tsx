// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, Suspense } from 'react';

import { Flex, Heading, IntelBrandedLoading } from '@geti/ui';
import { Outlet } from 'react-router';

import { $api } from '../../api/client';
import { License } from '../../features/license/license.component';
import { ServerErrorFallback } from './server-error-fallback.component';

const REFETCH_INTERVAL = 5000;
const MAX_RETRIES = 30;
const retryDelay = (attempt: number) => Math.min(1000 * 2 ** attempt, 5000);

const HealthCheck = ({ children }: { children: ReactNode }) => {
    const { data, isPending, isError } = $api.useQuery('get', '/health', undefined, {
        retry: MAX_RETRIES,
        retryDelay,
        refetchInterval: (query) => {
            return query.state.data?.status === 'ok' ? false : REFETCH_INTERVAL;
        },
    });

    if (isPending) {
        return (
            <Flex direction={'column'} justifyContent={'center'} alignItems={'center'} height={'100vh'}>
                <IntelBrandedLoading height={'auto'} />
                <Heading bottom={'size-4600'} level={2}>
                    Loading...Please wait.
                </Heading>
            </Flex>
        );
    }

    if (isError) {
        return <ServerErrorFallback />;
    }

    if (data?.status === 'ok') {
        return children;
    }

    return <IntelBrandedLoading />;
};

const LicenseCheck = ({ children }: { children: ReactNode }) => {
    const { data, isPending, isError } = $api.useQuery('get', '/api/system/info', undefined, {
        retry: 2,
        refetchInterval: (query) => {
            return query.state.data?.license_accepted ? false : REFETCH_INTERVAL;
        },
    });

    if (isPending) {
        return <IntelBrandedLoading />;
    }

    if (isError) {
        return <ServerErrorFallback />;
    }

    if (data && !data.license_accepted) {
        return <License platform={data.platform} />;
    }

    return children;
};

export const RootLayout = () => {
    return (
        <Suspense fallback={<IntelBrandedLoading />}>
            <HealthCheck>
                <LicenseCheck>
                    <Outlet />
                </LicenseCheck>
            </HealthCheck>
        </Suspense>
    );
};
