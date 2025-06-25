import { useEffect, useRef, useState } from 'react';

import { ActionButton, Flex, Heading, Item, TabList, TabPanels, Tabs, View } from '@geti/ui';
import { CartesianGrid, Label, Line, LineChart, ReferenceLine, XAxis, YAxis } from 'recharts';

import { ReactComponent as DoubleChevronRight } from './../../assets/double-chevron-right-icon.svg';

const generateData = (n: number) => {
    const result = new Array(n);
    result[0] = 100;

    for (let idx = 1; idx < n; idx++) {
        const dx = 1 + Math.random() * 0.3 - 0.15;
        result[idx] = Math.max(0, Math.min(100, result[idx - 1] * dx));
    }

    return result;
};

function useData() {
    const [data, setData] = useState(
        generateData(60).map((value, idx) => {
            return { name: `${idx}`, value };
        })
    );

    const timeout = useRef<ReturnType<typeof setTimeout>>(undefined);

    useEffect(() => {
        timeout.current = setTimeout(
            () => {
                const newData = data.slice(1);
                const previous = newData.at(-1);

                if (!previous) {
                    return;
                }

                const dx = 1 + Math.random() * 0.3 - 0.15;
                newData.push({
                    name: `${Number(previous.name) + 1}`,
                    value: Math.max(0, Math.min(100, previous.value * dx)),
                });

                setData(newData);
            },
            Math.random() * 100 + 250
        );

        return () => {
            if (timeout.current) {
                clearTimeout(timeout.current);
            }
        };
    });

    return data;
}

function Graph({ label }: { label: string }) {
    const data = useData();

    return (
        <>
            <LineChart width={500} height={200} data={data}>
                <XAxis
                    minTickGap={32}
                    stroke='var(--spectrum-global-color-gray-800)'
                    dataKey='name'
                    tickLine={false}
                    tickMargin={8}
                    // tickFormatter={(value) => {
                    //     const date = new Date(value);
                    //     return date.toLocaleDateString('en-US', {
                    //         month: 'short',
                    //         day: 'numeric',
                    //     });
                    // }}
                />
                <YAxis
                    // ...
                    tickLine={false}
                    stroke='var(--spectrum-global-color-gray-900)'
                    dataKey='value'
                    tickFormatter={(value: number) => {
                        return value > 10 ? value.toFixed(0) : value.toFixed(2);
                    }}
                >
                    <Label
                        // ...
                        angle={-90}
                        value={label}
                        position='insideLeft'
                        offset={10}
                        style={{
                            textAnchor: 'middle',
                            fill: 'var(--spectrum-global-color-gray-900)',
                            fontSize: '10px',
                        }}
                    />
                </YAxis>
                {/* <CartesianAxis stroke='var(--spectrum-global-color-gray-800)' /> */}
                <CartesianGrid stroke='var(--spectrum-global-color-gray-400)' />
                <ReferenceLine
                    x={data[0].name}
                    // ...
                    stroke='var(--spectrum-global-color-gray-600)'
                    strokeWidth={2}
                />
                <Line
                    type='linear'
                    dataKey='value'
                    dot={false}
                    stroke='var(--energy-blue)'
                    isAnimationActive={false}
                    strokeWidth='3'
                />
            </LineChart>
        </>
    );
}

export function Aside() {
    const [isHidden, setIsHidden] = useState(false);

    return (
        <View backgroundColor={'gray-100'} gridArea='aside' padding={isHidden ? 'size-200' : 'size-400'}>
            <Tabs aria-label='Aside view' density='compact'>
                <Flex alignItems='center' gap='size-400'>
                    <ActionButton
                        isQuiet
                        onPress={() => setIsHidden((hidden) => !hidden)}
                        UNSAFE_style={{
                            transform: isHidden ? 'scaleX(-1)' : 'scaleX(1)',
                        }}
                    >
                        <DoubleChevronRight />
                    </ActionButton>
                    <TabList
                        isHidden={isHidden}
                        width='100%'
                        UNSAFE_style={{
                            '--spectrum-tabs-rule-height': '3px',
                            '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                        }}
                    >
                        <Item key='monitoring'>Monitoring</Item>
                        <Item key='data-collection'>Data collection</Item>
                    </TabList>
                </Flex>
                <TabPanels marginTop='size-200' isHidden={isHidden}>
                    <Item key='monitoring'>
                        <Flex height='100%' justifyContent={'space-between'} direction={'column'}>
                            <View paddingX='size-200'>
                                <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)' }} level={4}>
                                    Troughput
                                </Heading>
                                <Graph label='fps' />
                            </View>
                            <View paddingX='size-200'>
                                <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)' }} level={4}>
                                    Latency per frame
                                </Heading>
                                <Graph label='ms per frame' />
                            </View>
                            <View paddingX='size-200'>
                                <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)' }} level={4}>
                                    Model confidence
                                </Heading>
                                <Graph label='Score' />
                            </View>
                            <View paddingX='size-200'>
                                <Heading UNSAFE_style={{ color: 'var(--spectrum-gray-900)' }} level={4}>
                                    Resource utilization
                                </Heading>
                                <Graph label='CPU' />
                            </View>
                        </Flex>
                    </Item>
                    <Item key='data-collection'>Senatus Populusque Romanus.</Item>
                </TabPanels>
            </Tabs>
        </View>
    );
}
