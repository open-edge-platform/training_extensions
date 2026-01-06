// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, Suspense } from 'react';

import { Button, Heading, IllustratedMessage, IntelBrandedLoading, Toast, View } from '@geti/ui';
import { CloudErrorIcon } from '@geti/ui/icons';
import { Outlet } from 'react-router';

import { $api } from '../../api/client';
import { paths } from '../../constants/paths';
import { redirectTo } from '../utils';

const HealthCheck = ({ children }: { children: ReactNode }) => {
    const { data, error } = $api.useQuery('get', '/health', undefined, {
        retry: 2,
        refetchInterval: (query) => {
            return query.state.data?.status === 'ok' ? false : 2000;
        },
    });

    if (error) {
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

export const RootLayout = () => {
    return (
        <Suspense fallback={<IntelBrandedLoading />}>
            <HealthCheck>
                <Outlet />
                <Toast />
            </HealthCheck>
        </Suspense>
    );
};
