// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSessionStorage } from 'usehooks-ts';

import { DatasetImportState, getParsedSessionStorage } from './utils';

export const useDatasetImportStorage = <TStep, TEntry extends DatasetImportState<TStep>>(storageKey: string) => {
    const [importEntries, setImportEntries] = useSessionStorage<TEntry[] | null>(
        storageKey,
        () => getParsedSessionStorage<TEntry[]>(storageKey) ?? null
    );

    const getAllImportEntries = (): TEntry[] => {
        return importEntries ?? [];
    };

    const getImportEntry = (stagedDatasetId: string): TEntry | null => {
        return importEntries?.find((item) => item.stagedDatasetId === stagedDatasetId) ?? null;
    };

    const appendImportEntry = (newEntry: TEntry) => {
        return setImportEntries((prev) => [...(prev ?? []), newEntry]);
    };

    const deleteImportEntry = (stagedDatasetId: string): void => {
        return setImportEntries((prev) => prev?.filter((item) => item.stagedDatasetId !== stagedDatasetId) ?? null);
    };

    const updateImportEntryStep = (stagedDatasetId: string, newStep: TStep): void => {
        return setImportEntries(
            (prev) =>
                prev?.map((item) => (item.stagedDatasetId === stagedDatasetId ? { ...item, step: newStep } : item)) ??
                null
        );
    };

    const updateImportEntry = (stagedDatasetId: string, newImportState: Partial<TEntry>): void => {
        return setImportEntries(
            (prev) =>
                prev?.map((item) =>
                    item.stagedDatasetId === stagedDatasetId ? { ...item, ...newImportState } : item
                ) ?? []
        );
    };

    return {
        getAllImportEntries,
        appendImportEntry,
        getImportEntry,
        deleteImportEntry,
        updateImportEntryStep,
        updateImportEntry,
    };
};
