// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Tauri-only override for IntelBrandedLoading (resolved by the `.tauri.*`
// extension list in rsbuild.config.ts). Replaces the animated WebP with a
// pure-CSS spinner so the embedded webview doesn't decode a frame per ~16ms
// while the main thread is busy bootstrapping WASM / OpenCV / SAM workers.

import { Flex, Loading } from '@geti/ui';

interface IntelBrandedLoadingProps {
    height?: string;
}

export const IntelBrandedLoading = ({ height = '100vh' }: IntelBrandedLoadingProps) => {
    return (
        <Flex justifyContent='center' alignItems='center' height={height} direction='column' marginBottom={'size-400'}>
            <Loading mode={'inline'} size={'L'} aria-label={'Loading'} />
        </Flex>
    );
};
