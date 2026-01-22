// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { providersRender as render } from '../../../../../../../test-utils/required-providers-render';
import { RangeParameterField } from './range-parameter-field.component';

describe('RangeParameterField', () => {
    const defaultValue = [0.5, 1.5];
    const name = 'Scaling ratio range';
    const onChange = jest.fn();

    beforeEach(() => {
        jest.clearAllMocks();
    });

    const renderApp = ({ value = defaultValue, isDisabled = false }: { value?: number[]; isDisabled?: boolean }) => {
        return render(
            <RangeParameterField
                defaultValue={defaultValue}
                value={value}
                onChange={onChange}
                name={name}
                type={'array'}
                isDisabled={isDisabled}
                step={0.001}
            />
        );
    };

    it('renders two NumberFields and a RangeSlider', () => {
        renderApp({});

        expect(screen.getByLabelText(`Change ${name} start range value`)).toBeInTheDocument();
        expect(screen.getByLabelText(`Change ${name} end range value`)).toBeInTheDocument();
        expect(screen.getByLabelText(`Change ${name} value`)).toBeInTheDocument();
    });

    it('calls onChange when start value changes', async () => {
        renderApp({});

        const startField = screen.getByLabelText(`Change ${name} start range value`);
        await userEvent.clear(startField);
        await userEvent.type(startField, '1');
        // Simulate blur to trigger onChange
        startField.blur();

        expect(onChange).toHaveBeenCalledWith([1, 1.5]);
    });

    it('calls onChange when end value changes', async () => {
        renderApp({});

        const endField = screen.getByLabelText(`Change ${name} end range value`);
        await userEvent.clear(endField);
        await userEvent.type(endField, '1');
        endField.blur();

        expect(onChange).toHaveBeenCalledWith([0.5, 1]);
    });

    it('does not allow start and end to be equal', async () => {
        renderApp({});

        const startField = screen.getByLabelText(`Change ${name} start range value`);
        const endField = screen.getByLabelText(`Change ${name} end range value`);
        await userEvent.clear(startField);
        await userEvent.type(startField, '1');
        startField.blur();
        await userEvent.clear(endField);
        await userEvent.type(endField, '1');
        endField.blur();

        // onChange should not be called with [1, 1] because end value was not change to the same as start value
        expect(onChange).toHaveBeenCalledWith([1, 1.5]);
    });

    it('disables fields when isDisabled is true', () => {
        renderApp({ isDisabled: true });

        expect(screen.getByLabelText(`Change ${name} start range value`)).toBeDisabled();
        expect(screen.getByLabelText(`Change ${name} end range value`)).toBeDisabled();

        const slider = screen.getByLabelText(`Change ${name} value`);

        expect(slider).toHaveClass('A-RCEa_is-disabled');
    });

    it('does not allow range slider start and end to be equal', async () => {
        renderApp({});

        const handle = screen.getByRole('button', { name: 'Increase Change Scaling ratio range start range value' });

        fireEvent.mouseDown(handle, { clientX: 0 });
        fireEvent.mouseMove(handle, { clientX: 1000 }); // Move it far to the right
        fireEvent.mouseUp(handle);

        expect(onChange).not.toHaveBeenCalledWith([1.5, 1.5]);
    });
});
