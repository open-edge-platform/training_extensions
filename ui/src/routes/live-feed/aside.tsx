import { useEffect, useRef, useState } from 'react';

import { ActionButton, Flex, Grid, Heading, View } from '@geti/ui';
import { CartesianGrid, Label, Line, LineChart, ReferenceLine, XAxis, YAxis } from 'recharts';

import { ReactComponent as DoubleChevronRight } from './../../assets/icons/double-chevron-right-icon.svg';

const generateData = (n: number) => {
    const result = new Array(n);
    result[0] = 100;

    for (let idx = 1; idx < n; idx++) {
        const dx = 1 + Math.random() * 0.3 - 0.15;
        result[idx] = Math.max(0, Math.min(100, result[idx - 1] * dx));
    }

    return result;
};

const useData = () => {
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
};

const Graph = ({ label }: { label: string }) => {
    const data = useData();

    return (
        <View padding={{ top: 'size-250', left: 'size-200', right: 'size-200', bottom: 'size-125' }}>
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
        </View>
    );
};

export const Aside = () => {
    const [isHidden, setIsHidden] = useState(false);

    return (
        <Grid
            gridArea={'aside'}
            height={'90vh'}
            areas={['header', 'graphs']}
            rows={['min-content', 'minmax(0, 1fr)']}
            UNSAFE_style={{
                padding: 'var(--spectrum-global-dimension-size-200)',
                paddingLeft: isHidden
                    ? 'var(--spectrum-global-dimension-size-100)'
                    : 'var(--spectrum-global-dimension-size-200)',
                paddingRight: isHidden
                    ? 'var(--spectrum-global-dimension-size-100)'
                    : 'var(--spectrum-global-dimension-size-200)',
                backgroundColor: 'var(--spectrum-global-color-gray-100)',
                transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                width: isHidden
                    ? 'var(--spectrum-global-dimension-size-400)'
                    : 'var(--spectrum-global-dimension-size-6000)',
            }}
        >
            <Flex gridArea={'header'} alignItems='center' gap={'size-100'} marginBottom={'size-300'}>
                <ActionButton
                    isQuiet
                    onPress={() => setIsHidden((hidden) => !hidden)}
                    UNSAFE_style={{
                        transform: isHidden ? 'scaleX(-1)' : 'scaleX(1)',
                        cursor: 'pointer',
                    }}
                >
                    <DoubleChevronRight />
                </ActionButton>
                <Heading level={4} isHidden={isHidden}>
                    Model statistics
                </Heading>
            </Flex>
            <View gridArea={'graphs'} isHidden={isHidden} UNSAFE_style={{ overflow: 'hidden auto' }}>
                {/* TODO: Extract these into a shared component */}
                <View paddingY='size-200'>
                    <Heading level={4} marginBottom={'size-300'}>
                        Throughput
                    </Heading>
                    <Graph label='fps' />
                </View>
                <View paddingY='size-200'>
                    <Heading level={4} marginBottom={'size-300'}>
                        Latency per frame
                    </Heading>
                    <Graph label='ms per frame' />
                </View>
                <View paddingY='size-200'>
                    <Heading level={4} marginBottom={'size-300'}>
                        Model confidence
                    </Heading>
                    <Graph label='Score' />
                </View>
                <View paddingY='size-200'>
                    <Heading level={4} marginBottom={'size-300'}>
                        Resource utilization
                    </Heading>
                    <Graph label='CPU' />
                </View>
            </View>
        </Grid>
    );
};
