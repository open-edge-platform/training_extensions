// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { Tag } from '@geti/ui';

import classes from './model-tag.module.scss';

interface TemplateTagProps {
    name: string;
    isActive?: boolean;
    isObsolete?: boolean;
    isDeprecated?: boolean;
}

export const ModelTag = ({ name, isActive = false, isObsolete = false, isDeprecated = false }: TemplateTagProps) => {
    if (isActive) {
        return <Tag text={name} withDot={false} className={classes.activeModelTag} />;
    }

    if (isObsolete) {
        return <Tag text={name} withDot={false} className={classes.obsoleteModel} />;
    }

    if (isDeprecated) {
        return <Tag text={name} withDot={false} className={classes.deprecatedModelTag} />;
    }

    return <Tag text={name} withDot={false} className={classes.templateNameTag} />;
};
