detection-model-target-dir := "data/projects/9d6af8e8-6017-4ebe-9126-33aae739c5fa/models/977eeb18-eaac-449d-bc80-e340fbe052ad"
segmentation-model-target-dir := "data/projects/a1b2c3d4-e5f6-7890-abcd-ef1234567890/models/c3d4e5f6-a7b8-9012-cdef-123456789012"

detection-video-url := "https://storage.geti.intel.com/test-data/geti-tune/media/card-video.mp4"
detection-video-target := "data/media/card-video.mp4"
detection-model-xml-url := "https://storage.geti.intel.com/test-data/geti-tune/models/ssd-card-detection.xml"
detection-model-xml-target := detection-model-target-dir + "/model.xml"
detection-model-bin-url := "https://storage.geti.intel.com/test-data/geti-tune/models/ssd-card-detection.bin"
detection-model-bin-target := detection-model-target-dir + "/model.bin"

segmentation-video-url := "https://storage.geti.intel.com/test-data/geti-tune/media/fish-video.mp4"
segmentation-video-target := "data/media/fish-video.mp4"
segmentation-model-xml-url := "https://storage.geti.intel.com/test-data/geti-tune/models/rtmdet-tiny-fish-segmentation.xml"
segmentation-model-xml-target := segmentation-model-target-dir + "/model.xml"
segmentation-model-bin-url := "https://storage.geti.intel.com/test-data/geti-tune/models/rtmdet-tiny-fish-segmentation.bin"
segmentation-model-bin-target := segmentation-model-target-dir + "/model.bin"

check-proxy:
    #!/usr/bin/env bash
    if [ -z ${https_proxy+x} ]; then
        echo "Error: https_proxy is unset";
    else
        echo "https_proxy is set to '$https_proxy'";
    fi

download-files: check-proxy
    #!/usr/bin/env bash
    echo "Downloading required files if not present..."
    # Download detection video
    if [ ! -f "{{ detection-video-target }}" ]; then
        mkdir -p "$(dirname "{{ detection-video-target }}")"
        echo "Downloading test video..."
        curl -fL "{{ detection-video-url }}" -o "{{ detection-video-target }}"
    else
        echo "Test video already exists at {{ detection-video-target }}"
    fi
    # Download segmentation video
    if [ ! -f "{{ segmentation-video-target }}" ]; then
        mkdir -p "$(dirname "{{ segmentation-video-target }}")"
        echo "Downloading test video..."
        curl -fL "{{ segmentation-video-url }}" -o "{{ segmentation-video-target }}"
    else
        echo "Test video already exists at {{ segmentation-video-target }}"
    fi
    # Download detection model XML
    if [ ! -f "{{ detection-model-xml-target }}" ]; then
        mkdir -p "$(dirname "{{ detection-model-xml-target }}")"
        echo "Downloading model XML..."
        curl -fL "{{ detection-model-xml-url }}" -o "{{ detection-model-xml-target }}"
    else
        echo "Model XML already exists at {{ detection-model-xml-target }}"
    fi
    # Download detection model BIN
    if [ ! -f "{{ detection-model-bin-target }}" ]; then
        mkdir -p "$(dirname "{{ detection-model-bin-target }}")"
        echo "Downloading model BIN..."
        curl -fL "{{ detection-model-bin-url }}" -o "{{ detection-model-bin-target }}"
    else
        echo "Model BIN already exists at {{ detection-model-bin-target }}"
    fi
    # Download segmentation model XML
    if [ ! -f "{{ segmentation-model-xml-target }}" ]; then
        mkdir -p "$(dirname "{{ segmentation-model-xml-target }}")"
        echo "Downloading model XML..."
        curl -fL "{{ segmentation-model-xml-url }}" -o "{{ segmentation-model-xml-target }}"
    else
        echo "Model XML already exists at {{ segmentation-model-xml-target }}"
    fi
    # Download segmentation model BIN
    if [ ! -f "{{ segmentation-model-bin-target }}" ]; then
        mkdir -p "$(dirname "{{ segmentation-model-bin-target }}")"
        echo "Downloading model BIN..."
        curl -fL "{{ segmentation-model-bin-url }}" -o "{{ segmentation-model-bin-target }}"
    else
        echo "Model BIN already exists at {{ segmentation-model-bin-target }}"
    fi

