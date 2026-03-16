// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

export const useIsVisible = ({
    element,
    options,
}: {
    element: HTMLElement | undefined | null;
    options?: IntersectionObserverInit;
}): boolean => {
    const [isVisible, setIsVisible] = useState<boolean>(false);

    const optionsRef = useRef(options);

    useEffect(() => {
        optionsRef.current = options;
    }, [options]);

    useEffect(() => {
        if (!element || isVisible) return;

        const observer = new IntersectionObserver(([entry]) => {
            if (entry.isIntersecting) {
                setIsVisible(true);
                observer.disconnect();
            }
        }, optionsRef.current);

        observer.observe(element);

        return () => {
            observer.disconnect();
        };
    }, [element, isVisible]);

    return isVisible;
};
