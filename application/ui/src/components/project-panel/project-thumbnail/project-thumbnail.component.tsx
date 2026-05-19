// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { PhotoPlaceholder, View, type DimensionValue } from '@geti/ui';

import { type Project } from '../../../constants/shared-types';
import { getProjectThumbnailUrl } from '../../../shared/media-url.utils';

import classes from './project-thumbnail.module.scss';

interface ProjectThumbnailProps {
    project: Pick<Project, 'id' | 'name'>;
    height: DimensionValue;
    width: DimensionValue;
}

export const ProjectThumbnail = ({ project, height, width }: ProjectThumbnailProps) => {
    const [isError, setIsError] = useState(false);

    if (isError) {
        return (
            <PhotoPlaceholder
                name={project.name}
                indicator={project.id ?? project.name}
                height={height}
                width={width}
            />
        );
    }

    return (
        <View width={width} height={height} UNSAFE_className={classes.thumbnailWrapper}>
            <img
                src={getProjectThumbnailUrl(project.id)}
                alt={project.name}
                onError={() => setIsError(true)}
                className={classes.thumbnail}
            />
        </View>
    );
};
