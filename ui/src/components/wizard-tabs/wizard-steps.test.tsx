import { render, screen } from '@testing-library/react';
import { TabPanel } from 'react-aria-components';

import { WizardTab, WizardTabList, WizardTabs } from './wizard-steps';

describe('WizardTabs', () => {
    it('Acts like a Tab', () => {
        render(
            <WizardTabs>
                <WizardTabList>
                    <WizardTab id='input' number={1} isCompleted>
                        Input
                    </WizardTab>
                    <WizardTab id='models' number={2}>
                        Models
                    </WizardTab>
                </WizardTabList>
                <TabPanel id={'input'}>Input</TabPanel>
                <TabPanel id={'models'}>Models</TabPanel>
            </WizardTabs>
        );

        const input = screen.getByRole('tab', { name: /Input/ });
        expect(input).toBeInTheDocument();
        expect(input).not.toHaveTextContent('1');
        expect(input).toHaveTextContent('Input');

        const models = screen.getByRole('tab', { name: /Models/ });
        expect(models).toBeInTheDocument();
        expect(models).toHaveTextContent('2');
        expect(models).toHaveTextContent('Models');

        expect(screen.getByRole('tabpanel', { name: /Input/ })).toBeInTheDocument();
        expect(screen.queryByRole('tabpanel', { name: /Model/ })).not.toBeInTheDocument();
    });
});
