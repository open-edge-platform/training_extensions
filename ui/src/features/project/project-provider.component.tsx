// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { v4 as uuid } from 'uuid';

import { $api } from '../../api/client';
import { TaskType } from './task-selection/interface';

type ProjectContextType = {
    createProject: ({ onSuccess }: { onSuccess?: () => void }) => void;
    setSelectedTask: Dispatch<SetStateAction<TaskType>>;
    setLabels: Dispatch<SetStateAction<{ name: string }[]>>;
    setName: Dispatch<SetStateAction<string>>;
    selectedTask: TaskType;
    labels: { name: string }[];
    name: string;
};

const ProjectContext = createContext<ProjectContextType | null>(null);

type ProjectProviderProps = {
    children: ReactNode;
};

export const ProjectProvider = ({ children }: ProjectProviderProps) => {
    const [selectedTask, setSelectedTask] = useState<TaskType>('detection');
    const [labels, setLabels] = useState<{ name: string }[]>([]);
    const [name, setName] = useState<string>('Project #1');

    const projectId = uuid();

    const createProjectMutation = $api.useMutation('post', '/api/projects');

    const createProject = ({ onSuccess }: { onSuccess?: () => void }) => {
        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    task: {
                        task_type: selectedTask,
                        exclusive_labels: selectedTask === 'classification',
                        labels,
                    },
                    name,
                },
            },
            { onSuccess }
        );
    };

    const value = {
        createProject,

        setSelectedTask,
        selectedTask,

        setLabels,
        labels,

        setName,
        name,
    };

    return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>;
};

export const useProject = (): ProjectContextType => {
    const context = useContext(ProjectContext);

    if (context === null) {
        throw new Error('useProject must be used within a ProjectProvider');
    }

    return context;
};
