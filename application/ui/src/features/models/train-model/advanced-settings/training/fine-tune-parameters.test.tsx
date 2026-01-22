// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { useState } from 'react';

import { fireEvent, screen } from '@testing-library/react';

import { providersRender as render } from '../../../../../../../test-utils/required-providers-render';
import { FineTuneParameters } from './fine-tune-parameters.component';

describe('FineTuneParameters', () => {
    const App = (props: { trainFromScratch: boolean; isReshufflingSubsetsEnabled: boolean }) => {
        const [trainFromScratch, setTrainFromScratch] = useState(props.trainFromScratch);
        const [isReshufflingSubsetsEnabled, setIsReshufflingSubsetsEnabled] = useState(
            props.isReshufflingSubsetsEnabled
        );

        return (
            <FineTuneParameters
                trainFromScratch={trainFromScratch}
                onTrainFromScratchChange={setTrainFromScratch}
                onReshufflingSubsetsEnabledChange={setIsReshufflingSubsetsEnabled}
                isReshufflingSubsetsEnabled={isReshufflingSubsetsEnabled}
            />
        );
    };

    it('Fine tune parameter changes based on the trainFromScratch', () => {
        render(<App trainFromScratch={true} isReshufflingSubsetsEnabled={false} />);

        expect(screen.getByText(/fine\-tune parameters/i)).toBeInTheDocument();
        expect(screen.getByRole('radio', { name: /pre\-trained weights/i })).toBeChecked();
        expect(screen.getByLabelText('Fine-tune parameters tag')).toHaveTextContent('Pre-trained weights');
        expect(screen.getByRole('radio', { name: /previous training weights/i })).not.toBeChecked();

        fireEvent.click(screen.getByRole('radio', { name: /previous training weights/i }));
        expect(screen.getByLabelText('Fine-tune parameters tag')).toHaveTextContent('Previous training weights');
        expect(screen.getByRole('radio', { name: /previous training weights/i })).toBeChecked();
        expect(screen.getByRole('radio', { name: /pre\-trained weights/i })).not.toBeChecked();
    });

    it('Reshuffle subsets is disabled when Previous training weights is selected, enabled when Pre-trained weights is selected', () => {
        render(<App trainFromScratch={true} isReshufflingSubsetsEnabled={false} />);

        expect(screen.getByRole('radio', { name: /pre\-trained weights/i })).toBeChecked();
        expect(screen.getByRole('checkbox', { name: /reshuffle subsets/i })).toBeEnabled();

        fireEvent.click(screen.getByRole('checkbox', { name: /reshuffle subsets/i }));
        expect(screen.getByRole('checkbox', { name: /reshuffle subsets/i })).toBeChecked();

        fireEvent.click(screen.getByRole('radio', { name: /previous training weights/i }));

        expect(screen.getByRole('radio', { name: /previous training weights/i })).toBeChecked();
        expect(screen.getByRole('checkbox', { name: /reshuffle subsets/i })).toBeDisabled();
    });
});
