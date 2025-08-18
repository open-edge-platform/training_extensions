export type WizardStep = {
    href: string;
    name: string;
    isDisabled: boolean;
    isCompleted: boolean;
    isSelected: boolean;
};

export type WizardState = Array<WizardStep>;
