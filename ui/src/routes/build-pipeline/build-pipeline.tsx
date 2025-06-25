import { useState } from 'react';

import { View } from '@geti/ui';
import { TabPanel } from 'react-aria-components';

import { WizardTab, WizardTabList, WizardTabs } from '../../components/wizard-tabs/wizard-steps';
import { BuildInput } from './input';
import { BuildModel } from './model';
import { BuildOutput } from './output';

type WizardStep = {
    id: string;
    name: string;
    isDisabled: boolean;
    isCompleted: boolean;
    isSelected: boolean;
};
type WizardState = Array<WizardStep>;

export function BuildPipeline({ submit }: { submit: () => void }) {
    const [wizardState, setWizardState] = useState<WizardState>([
        {
            id: 'input',
            name: 'Input configuration',
            isCompleted: false,
            isDisabled: false,
            isSelected: true,
        },
        {
            id: 'model',
            name: 'Model configuration',
            isCompleted: false,
            isDisabled: false,
            isSelected: false,
        },
        {
            id: 'output',
            name: 'Output & Integration configuration',
            isCompleted: false,
            isDisabled: false,
            isSelected: false,
        },
    ]);

    return (
        <View
            UNSAFE_style={{
                color: 'var(--spectrum-global-color-gray-700)',
                userSelect: 'none',
            }}
        >
            <WizardTabs
                selectedKey={wizardState.find(({ isSelected }) => isSelected)?.id}
                onSelectionChange={(selectedId) => {
                    setWizardState((steps) => steps.map((step) => ({ ...step, isSelected: selectedId === step.id })));
                }}
            >
                <WizardTabList aria-label='Pipeline wizard'>
                    {wizardState.map((step, idx) => {
                        return (
                            <WizardTab
                                key={step.id}
                                id={step.id}
                                number={idx + 1}
                                isDisabled={step.isDisabled}
                                isCompleted={step.isCompleted}
                            >
                                {step.name}
                            </WizardTab>
                        );
                    })}
                </WizardTabList>
                <TabPanel id={'input'} style={{ display: 'flex', justifyContent: 'center' }}>
                    <BuildInput
                        next={() => {
                            setWizardState((steps) =>
                                steps.map((step) => ({ ...step, isSelected: step.id === 'model' }))
                            );
                        }}
                    />
                </TabPanel>
                <TabPanel id={'model'}>
                    <BuildModel
                        next={() => {
                            setWizardState((steps) =>
                                steps.map((step) => ({ ...step, isSelected: step.id === 'output' }))
                            );
                        }}
                        back={() => {
                            setWizardState((steps) =>
                                steps.map((step) => ({ ...step, isSelected: step.id === 'input' }))
                            );
                        }}
                    />
                </TabPanel>
                <TabPanel id={'output'}>
                    <BuildOutput
                        back={() => {
                            setWizardState((steps) =>
                                steps.map((step) => ({ ...step, isSelected: step.id === 'model' }))
                            );
                        }}
                        next={() => {
                            alert('submit');
                            submit();
                        }}
                    />
                </TabPanel>
                <TabPanel id={'pay'}>Pay</TabPanel>
            </WizardTabs>
        </View>
    );
}
