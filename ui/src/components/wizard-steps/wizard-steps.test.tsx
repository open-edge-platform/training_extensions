import { render, screen } from '@testing-library/react';
import { TabPanel } from 'react-aria-components';

import { Step, StepList, Steps } from './wizard-steps';

describe('Tabs', () => {
    it('Acts like a Tab', () => {
        render(
            <Steps>
                <StepList>
                    <Step id='input' number={1} isCompleted>
                        Input
                    </Step>
                    <Step id='models' number={2}>
                        Models
                    </Step>
                </StepList>
                <TabPanel id={'input'}>Input</TabPanel>
                <TabPanel id={'models'}>Models</TabPanel>
            </Steps>
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
