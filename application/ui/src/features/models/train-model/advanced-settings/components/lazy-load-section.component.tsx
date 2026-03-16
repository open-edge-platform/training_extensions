// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useRef, useState } from 'react';

import { useIsVisible } from 'hooks/use-is-visible.hook';

type LazyLoadSectionProps = {
    children: ReactNode;
};

export const LazyLoadSection = ({ children }: LazyLoadSectionProps) => {
    const [container, setContainer] = useState<HTMLDivElement | null>(null);
    const rootRef = useRef<Element | null>(null);

    if (rootRef.current === null) {
        rootRef.current = document.querySelector('[data-testid="advanced-settings-scroll-container"]');
    }

    const isVisible = useIsVisible({
        element: container,
        options: {
            threshold: 0,
            root: rootRef.current,
            rootMargin: '150px',
        },
    });

    return (
        // It's safe to assume that min height is at least 50px. This value is used by useIsVisible to trigger
        // element rendering.
        <div ref={setContainer} style={{ minHeight: '50px' }}>
            {isVisible && children}
        </div>
    );
};
