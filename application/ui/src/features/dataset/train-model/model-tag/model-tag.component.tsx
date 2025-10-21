// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Tag } from '@geti/ui';

import classes from './model-tag.module.scss';

interface TemplateTagProps {
    name: string;
    isActive?: boolean;
    isObsolete?: boolean;
    isDeprecated?: boolean;
}

const getClassName = ({ isActive, isObsolete, isDeprecated }: Omit<TemplateTagProps, 'name'>) => {
    if (isActive) {
        return classes.activeModelTag;
    }
    if (isObsolete) {
        return classes.obsoleteModel;
    }
    if (isDeprecated) {
        return classes.deprecatedModelTag;
    }
    return classes.templateNameTag;
};

export const ModelTag = ({ name, isActive = false, isObsolete = false, isDeprecated = false }: TemplateTagProps) => {
    return <Tag text={name} withDot={false} className={getClassName({ isActive, isObsolete, isDeprecated })} />;
};
