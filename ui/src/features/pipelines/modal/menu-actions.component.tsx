import { ActionButton, Item, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

export const MenuActions = () => {
    return (
        <MenuTrigger>
            <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}>
                <MoreMenu />
            </ActionButton>
            <Menu>
                <Item>Export</Item>
                <Item>Duplicate</Item>
                <Item>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};
