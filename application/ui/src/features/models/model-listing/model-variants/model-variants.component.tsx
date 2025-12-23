// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Disclosure, DisclosurePanel, DisclosureTitle } from '@geti/ui';

import { ModelVariantsTabs } from './model-variant-tabs.component';

export const ModelVariants = () => {
    return (
        <Disclosure
            isQuiet
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
                paddingLeft: 'var(--spectrum-global-dimension-size-250)',
                paddingTop: 'var(--spectrum-global-dimension-size-250)',
                paddingBottom: 'var(--spectrum-global-dimension-size-250)',
                paddingRight: 'var(--spectrum-global-dimension-size-250)',
            }}
        >
            <DisclosureTitle>Model variants</DisclosureTitle>
            <DisclosurePanel>
                <ModelVariantsTabs />
            </DisclosurePanel>
        </Disclosure>
    );
};
