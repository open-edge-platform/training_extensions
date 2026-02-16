// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedLocalStorage } from './utils';

const STAGING_DATASET_KEY = (projectId: string) => `staging-dataset-${projectId}`;

export const useStageDataset = () => {
    const projectId = useProjectIdentifier();

    const [_lsStagingProject, setLsStagingId] = useLocalStorage<string[]>(
        STAGING_DATASET_KEY(projectId),
        () => getParsedLocalStorage(STAGING_DATASET_KEY(projectId)) ?? []
    );

    const getLsStagingIds = (): string[] => {
        return getParsedLocalStorage<string[]>(STAGING_DATASET_KEY(projectId)) ?? [];
    };

    const addLsStagingId = (jobId: string) => {
        return setLsStagingId((prevState) => [...(prevState ?? []), jobId]);
    };

    const removeLsStagingId = (jobId: string): void => {
        return setLsStagingId((prevState) => prevState.filter((id) => id !== jobId));
    };

    return {
        addLsStagingId,
        getLsStagingIds,
        removeLsStagingId,
    };
};
