import { View } from '@geti/ui';
import { TabPanel } from 'react-aria-components';
import { Outlet, useLocation } from 'react-router';

import { WizardTab, WizardTabList, WizardTabs } from '../../components/wizard-tabs/wizard-steps';
import { paths } from '../../router';
import Background from './../../assets/background.png';

type WizardStep = {
    href: string;
    name: string;
    isDisabled: boolean;
    isCompleted: boolean;
    isSelected: boolean;
};
type WizardState = Array<WizardStep>;

export function EditPipelineLayout() {
    const { pathname } = useLocation();

    // TODO: update according to server state
    const wizardState = [
        {
            href: paths.pipeline.input({}),
            name: 'Input configuration',
            isCompleted: false,
            isDisabled: false,
            isSelected: true,
        },
        {
            href: paths.pipeline.model({}),
            name: 'Model configuration',
            isCompleted: false,
            isDisabled: false,
            isSelected: false,
        },
        {
            href: paths.pipeline.output({}),
            name: 'Output & Integration configuration',
            isCompleted: false,
            isDisabled: false,
            isSelected: false,
        },
    ] satisfies WizardState;

    return (
        <View
            backgroundColor={'gray-100'}
            UNSAFE_style={{
                backgroundImage: `url(${Background})`,
                backgroundBlendMode: 'luminosity',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
                backgroundSize: 'cover',
            }}
            gridArea={'content'}
            height='100%'
            width='100%'
        >
            <View maxWidth={'1024px'} marginX='auto' paddingY='size-900'>
                <View UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>
                    <WizardTabs selectedKey={pathname}>
                        <WizardTabList aria-label='Pipeline wizard'>
                            {wizardState.map((step, idx) => {
                                return (
                                    <WizardTab
                                        key={step.href}
                                        id={step.href}
                                        href={step.href}
                                        number={idx + 1}
                                        isDisabled={step.isDisabled}
                                        isCompleted={step.isCompleted}
                                    >
                                        {step.name}
                                    </WizardTab>
                                );
                            })}
                        </WizardTabList>
                        <TabPanel id={paths.pipeline.input({})} style={{ display: 'flex', justifyContent: 'center' }}>
                            <Outlet />
                        </TabPanel>
                        <TabPanel id={paths.pipeline.model({})}>
                            <Outlet />
                        </TabPanel>
                        <TabPanel id={paths.pipeline.output({})}>
                            <Outlet />
                        </TabPanel>
                    </WizardTabs>
                </View>
            </View>
        </View>
    );
}
