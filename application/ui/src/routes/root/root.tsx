// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, Suspense } from 'react';

import { Button, Heading, IllustratedMessage, IntelBrandedLoading, View } from '@geti/ui';
import { CloudErrorIcon } from '@geti/ui/icons';
import { Outlet } from 'react-router';

import { $api } from '../../api/client';
import { paths } from '../../constants/paths';
import { License } from '../../features/license/license.component';
import { redirectTo } from '../utils';

const REFETCH_INTERVAL = 5000;

const HealthCheck = ({ children }: { children: ReactNode }) => {
    const { data, isPending, isError } = $api.useQuery('get', '/health', undefined, {
        retry: 2,
        refetchInterval: (query) => {
            return query.state.data?.status === 'ok' ? false : REFETCH_INTERVAL;
        },
    });

    if (isPending) {
        return <IntelBrandedLoading />;
    }

    if (isError) {
        return (
            <View height={'100vh'}>
                <IllustratedMessage>
                    <CloudErrorIcon size='XXL' />
                    <Heading>Server Error</Heading>

                    <Button
                        variant={'accent'}
                        marginTop={'size-200'}
                        onPress={() => {
                            redirectTo(paths.root({}));
                        }}
                    >
                        Refresh
                    </Button>
                </IllustratedMessage>
            </View>
        );
    }

    if (data?.status === 'ok') {
        return children;
    }

    return <IntelBrandedLoading />;
};

const LicenseGate = ({ children }: { children: ReactNode }) => {
    const { data, isPending } = $api.useQuery('get', '/api/system/info', undefined, {
        refetchInterval: (query) => {
            return query.state.data?.license_accepted ? false : REFETCH_INTERVAL;
        },
    });

    if (isPending) {
        return <IntelBrandedLoading />;
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
                <LicenseGate>
                    <Outlet />
                </LicenseGate>
            </HealthCheck>
        </Suspense>
    );
};
