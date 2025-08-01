import { Button, Content, DropZone, Flex, Heading, IllustratedMessage, Text, View } from '@geti/ui';
import { Checkmark } from '@geti/ui/icons';
import { isNull } from 'lodash-es';
import { FileTrigger } from 'react-aria-components';

export const DropModelFiles = ({
    files,
    onDrop,
    noFileMessages,
    acceptedFileTypes,
}: {
    files: (File | null)[];
    onDrop: (file: File | null) => void;
    noFileMessages: string[];
    acceptedFileTypes?: string[];
}) => {
    return (
        <DropZone
            flexGrow={1}
            isFilled={files.every((file) => !isNull(file))}
            onDrop={async (dropEvent) => {
                const dropItem = dropEvent.items.at(0) ?? null;

                if (dropItem?.kind === 'file') {
                    onDrop(await dropItem.getFile());
                }
            }}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-50)',
            }}
        >
            <IllustratedMessage>
                <Flex direction='column' gap='size-200'>
                    <Heading UNSAFE_style={{ color: 'white' }}>{'Upload your model'}</Heading>

                    <Content>
                        <Flex direction='column' gap='size-400'>
                            <Text
                                UNSAFE_style={{
                                    color: 'var(--spectrum-global-color-gray-600)',
                                }}
                            >
                                Supported formats: OpenVINO IR (xml + bin)
                            </Text>

                            <Flex direction={'column'}>
                                {files.map((file, index) =>
                                    isNull(file) ? (
                                        <Text
                                            UNSAFE_style={{
                                                color: 'var(--spectrum-global-color-gray-600)',
                                            }}
                                            key={index}
                                        >
                                            {noFileMessages[index] ?? 'You have to drop the file'}
                                        </Text>
                                    ) : (
                                        <Flex key={`${index}-${file.name}`} justifyContent={'center'}>
                                            <Text>{file?.name}</Text>
                                            <Checkmark
                                                size='S'
                                                UNSAFE_style={{ color: 'var(--spectrum-global-color-green-500)' }}
                                            />
                                        </Flex>
                                    )
                                )}
                            </Flex>

                            <FileTrigger
                                acceptedFileTypes={acceptedFileTypes}
                                onSelect={(e) => {
                                    if (e === null) {
                                        return;
                                    }

                                    onDrop(e.item(0));
                                }}
                            >
                                <View>
                                    <Button variant='secondary'>Select files</Button>
                                </View>
                            </FileTrigger>
                        </Flex>
                    </Content>
                </Flex>
            </IllustratedMessage>
        </DropZone>
    );
};
