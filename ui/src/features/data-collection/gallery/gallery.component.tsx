import { AriaComponentsListBox, GridLayout, ListBoxItem, Size, View, Virtualizer } from '@geti/ui';

import { useSelectedData } from '../../../routes/data-collection/provider';
import { CheckboxInput } from '../checkbox-input';
import { response } from '../mock-response';
import { MediaItem } from './media-item.component';

import classes from './gallery.module.scss';

const layoutOptions = {
    minSpace: new Size(8, 8),
    maxColumns: 8,
    preserveAspectRatio: true,
};

export const Gallery = () => {
    const { selectedKeys, setSelectedKeys } = useSelectedData();

    const isSetSelectedKeys = selectedKeys instanceof Set;

    return (
        <View UNSAFE_className={classes.mainContainer}>
            <Virtualizer layout={GridLayout} layoutOptions={layoutOptions}>
                <AriaComponentsListBox
                    layout='grid'
                    aria-label='data-collection-grid'
                    className={classes.container}
                    selectedKeys={selectedKeys}
                    selectionMode='multiple'
                    escapeKeyBehavior='clearSelection'
                    onSelectionChange={setSelectedKeys}
                >
                    {response.items.map((item) => (
                        <ListBoxItem
                            key={item.image}
                            id={item.image}
                            textValue={item.image}
                            className={classes.mediaItem}
                        >
                            <MediaItem
                                item={item}
                                topLeftElement={() => (
                                    <CheckboxInput
                                        isReadOnly
                                        name={`select-${item.image}`}
                                        isChecked={isSetSelectedKeys && selectedKeys.has(item.image)}
                                    />
                                )}
                            />
                        </ListBoxItem>
                    ))}
                </AriaComponentsListBox>
            </Virtualizer>
        </View>
    );
};
