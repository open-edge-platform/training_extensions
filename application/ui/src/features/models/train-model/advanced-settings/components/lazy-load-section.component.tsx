// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, RefObject, useState } from 'react';

import { useIsVisible } from 'hooks/use-is-visible.hook';

type LazyLoadSectionProps = {
    children: ReactNode;
    rootRef: RefObject<HTMLDivElement | null>;
};

export const LazyLoadSection = ({ children, rootRef }: LazyLoadSectionProps) => {
    const [container, setContainer] = useState<HTMLDivElement | null>(null);

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
