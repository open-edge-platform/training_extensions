import { CSSProperties, Suspense } from 'react';

import { Loading, View } from '@geti/ui';
import { TabPanel } from 'react-aria-components';
import { Outlet, useLocation } from 'react-router';

import { WizardSteps } from '../../components/wizard-steps/wizard-steps';
import { WizardTabs } from '../../components/wizard-tabs/wizard-tabs';
import { paths } from '../../router';
import Background from './../../assets/background.png';

export const EditPipelineLayout = () => {
    const { pathname } = useLocation();

    // TODO: Replace this hardcoded value by $api.useQuery('get', '/api/pipelines') once available
    const hasActivePipeline = true;

    // TODO: update according to server state
    const wizardState = [
        {
            href: paths.pipeline.source({}),
            name: 'Source configuration',
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
            href: paths.pipeline.sink({}),
            name: 'Sink & Integration configuration',
            isCompleted: false,
            isDisabled: false,
            isSelected: false,
        },
    ];

    const tabPanelStyles: CSSProperties = {
        overflowY: 'auto',
        maxHeight: '85vh',
    };

    const tabsContent = {
        pathname,
        state: wizardState,
        content: (
            <View width={'100%'} height={'100%'} marginTop={'size-150'} maxWidth={'1320px'}>
                <Suspense fallback={<Loading mode='inline' />}>
                    <TabPanel id={paths.pipeline.source({})}>
                        <Outlet />
                    </TabPanel>
                    <TabPanel id={paths.pipeline.model({})} style={tabPanelStyles}>
                        <Outlet />
                    </TabPanel>
                    <TabPanel id={paths.pipeline.sink({})}>
                        <Outlet />
                    </TabPanel>
                </Suspense>
            </View>
        ),
    };

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
            <View paddingY='size-800'>
                <View
                    UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}
                    maxWidth={'1320px'}
                    marginX='auto'
                >
                    {hasActivePipeline ? <WizardTabs {...tabsContent} /> : <WizardSteps {...tabsContent} />}
                </View>
            </View>
        </View>
    );
};
