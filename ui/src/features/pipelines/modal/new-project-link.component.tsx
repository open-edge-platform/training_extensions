import { Text } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { Link } from 'react-router-dom';

import { paths } from '../../../router';

import classes from './new-project-link.module.scss';

export const NewProjectLink = () => {
    return (
        <Link to={paths.pipeline.model.pattern} className={classes.link}>
            <AddCircle />
            <Text>Add another project 2</Text>
        </Link>
    );
};
