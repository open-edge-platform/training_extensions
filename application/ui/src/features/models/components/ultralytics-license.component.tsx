// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Link } from '../../../platform/components/link.component';

const ULTRALYTICS_LICENSE_TEXT = 'Follow Ultralytics guidance for the license usage';

export const UltralyticsLicense = () => {
    return (
        <Link
            href={'https://www.ultralytics.com/legal/agpl-3-0-software-license'}
            target={'_blank'}
            rel={'noopener noreferrer'}
        >
            {ULTRALYTICS_LICENSE_TEXT}
        </Link>
    );
};
