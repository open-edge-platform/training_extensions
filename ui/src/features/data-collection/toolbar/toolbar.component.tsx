import { Divider, Flex, Heading, Text } from '@geti/ui';

import { useSelectedData } from '../../../routes/data-collection/provider';
import { CheckboxInput } from '../checkbox-input';
import { response } from '../mock-response';
import { toggleMultipleSelection } from './util';

export const Toolbar = () => {
    const { setSelectedKeys, selectedKeys } = useSelectedData();
    const totalSelectedElements = selectedKeys instanceof Set ? selectedKeys.size : 0;

    const message = totalSelectedElements > 0 ? `${totalSelectedElements} selected` : `${response.items.length} images`;

    const handleToggleManyItemSelection = () => {
        const images = response.items.map((item) => item.image);
        setSelectedKeys(toggleMultipleSelection(images));
    };

    return (
        <Flex direction={'column'} gridArea={'toolbar'} gap={'size-200'} marginBottom={'size-200'}>
            <Heading level={1}>Data collection</Heading>
            <Divider size='S' />

            <Flex direction='row' alignItems='center' justifyContent={'space-between'} gap='size-100'>
                <CheckboxInput
                    name='select all'
                    onChange={handleToggleManyItemSelection}
                    isChecked={totalSelectedElements === response.items.length}
                />

                <Text>{message}</Text>
            </Flex>

            <Divider size='S' />
        </Flex>
    );
};
