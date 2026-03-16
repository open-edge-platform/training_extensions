// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

export const useIsVisible = ({
    element,
    options,
}: {
    element: HTMLElement | undefined | null;
    options?: IntersectionObserverInit;
}): boolean => {
    const [isVisible, setIsVisible] = useState<boolean>(false);

    useEffect(() => {
        if (!element) return;

        const observer = new IntersectionObserver(([entry]) => {
            if (entry.isIntersecting) {
                setIsVisible(true);
                observer.disconnect();
            }
        }, options);

        observer.observe(element);

        return () => {
            observer.disconnect();
        };
    }, [element, options]);

    return isVisible;
};
