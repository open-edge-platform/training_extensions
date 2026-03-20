// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSubmitJob } from 'hooks/api/jobs/jobs.hook';

export const useTrainModelMutation = () => {
    return useSubmitJob();
};
