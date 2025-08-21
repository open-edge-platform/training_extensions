import { dimensionValue, Text } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { Link } from 'react-router-dom';

import { paths } from '../../../router';

export const NewPipelineButton = () => {
    return (
        <Link
            to={paths.pipeline.source.pattern}
            style={{
                cursor: 'pointer',
                border: '1px dashed var(--spectrum-gray-700) !important',
                display: 'flex',
                alignItems: 'center',
                flexDirection: 'column',
                justifyContent: 'center',
                borderRadius: 'regular',
                backgroundColor: 'var(--spectrum-global-color-gray-300)',
                gap: dimensionValue('size-200'),
                padding: dimensionValue('size-275'),
                minHeight: dimensionValue('size-2000'),
            }}
        >
            <AddCircle />
            <Text>Add another pipeline</Text>
        </Link>
    );
};
