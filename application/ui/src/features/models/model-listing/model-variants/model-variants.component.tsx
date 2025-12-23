// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Disclosure, DisclosurePanel, DisclosureTitle } from '@geti/ui';

import { ModelVariantsTabs } from './model-variant-tabs.component';

import classes from './model-variants.module.scss';

export const ModelVariants = () => {
    return (
        <Disclosure isQuiet UNSAFE_className={classes.disclosureVariant}>
            <DisclosureTitle>Model variants</DisclosureTitle>
            <DisclosurePanel>
                <ModelVariantsTabs />
            </DisclosurePanel>
        </Disclosure>
    );
};
