# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import Field, model_validator

from .base_model_no_extra import BaseModelNoExtra


class RandomResizeCrop(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable random resize crop",
        description="Whether to apply random resize and crop to the image. "
        "Note: this augmentation is not supported when Tiling algorithm is enabled.",
    )
    crop_ratio_range: list[float] | None = Field(
        default=None,
        title="Crop resize ratio range",
        description=(
            "Range (min, max) of crop ratios to apply during resize crop operation. "
            "Specifies the fraction of the original image dimensions to retain after cropping. "
            "For example, (0.8, 1.0) will randomly crop between 80% and 100% of the original size. "
            "Both values should be between 0.0 and 1.0."
        ),
    )
    aspect_ratio_range: list[float] | None = Field(
        default=None,
        title="Aspect ratio range",
        description=(
            "Range (min, max) of aspect ratios to apply during resize crop operation. "
            "Aspect ratio is defined as width divided by height. "
            "For example, (0.75, 1.33) allows the crop to have an aspect ratio between 3:4 and 4:3."
        ),
    )

    @model_validator(mode="after")
    def validate_crop_range(self) -> "RandomResizeCrop":
        if self.crop_ratio_range is None:
            return self
        if len(self.crop_ratio_range) != 2:
            raise ValueError("crop_ratio_range must be a list of exactly two float values")
        if self.crop_ratio_range[0] >= self.crop_ratio_range[1]:
            raise ValueError("The first value in crop_ratio_range must be less than the second value")
        if not (0 <= self.crop_ratio_range[0] <= 1) or not (0 <= self.crop_ratio_range[1] <= 1):
            raise ValueError("Values in crop_ratio_range must be between 0 and 1")
        return self

    @model_validator(mode="after")
    def validate_aspect_ratio_range(self) -> "RandomResizeCrop":
        if self.aspect_ratio_range is None:
            return self
        if len(self.aspect_ratio_range) != 2:
            raise ValueError("aspect_ratio_range must be a list of exactly two float values")
        if self.aspect_ratio_range[0] >= self.aspect_ratio_range[1]:
            raise ValueError("The first value in aspect_ratio_range must be less than the second value")
        if self.aspect_ratio_range[0] <= 0 or self.aspect_ratio_range[1] <= 0:
            raise ValueError("Values in aspect_ratio_range must be greater than 0")
        return self


class RandomAffine(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable random affine",
        description="Whether to apply random affine transformations to the image",
    )
    max_rotate_degree: float | None = Field(
        default=None,
        ge=0.0,
        title="Rotation degrees",
        description=(
            "Maximum rotation angle in degrees for affine transformation. "
            "A random angle in the range [-max_rotate_degree, max_rotate_degree] will be applied. "
            "For example, max_rotate_degree=10 allows up to ±10 degrees rotation."
        ),
    )
    max_translate_ratio: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        title="Horizontal translation",
        description=(
            "Maximum translation as a fraction of image width or height. "
            "A random translation in the range [-max_translate_ratio, max_translate_ratio] "
            "will be applied along both axes. For example, 0.1 allows up to ±10% translation."
        ),
    )
    scaling_ratio_range: list[float] | None = Field(
        default=None,
        title="Scaling ratio range",
        description=(
            "Range (min, max) of scaling factors to apply during affine transformation. "
            "Both values should be > 0.0. "
            "For example, (0.8, 1.2) will randomly scale the image between 80% and 120% of its original size."
        ),
    )
    max_shear_degree: float | None = Field(
        default=None,
        title="Maximum shear degree",
        description=(
            "Maximum absolute shear angle in degrees to apply during affine transformation. "
            "A random shear in the range [-max_shear_degree, max_shear_degree] will be applied."
        ),
    )

    @model_validator(mode="after")
    def validate_scaling_ratio_range(self) -> "RandomAffine":
        if self.scaling_ratio_range is None:
            return self
        if len(self.scaling_ratio_range) != 2:
            raise ValueError("scaling_ratio_range must be a list of exactly two float values")
        if self.scaling_ratio_range[0] >= self.scaling_ratio_range[1]:
            raise ValueError("The first value in scaling_ratio_range must be less than the second value")
        if self.scaling_ratio_range[0] <= 0 or self.scaling_ratio_range[1] <= 0:
            raise ValueError("Values in scaling_ratio_range must be greater than 0")
        return self


class RandomHorizontalFlip(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable random horizontal flip",
        description="Whether to apply random flip images horizontally along the vertical axis (swap left and right)",
    )
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying horizontal flip. "
            "A value of 0.5 means each image has a 50% chance to be flipped horizontally."
        ),
    )


class RandomVerticalFlip(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable random vertical flip",
        description="Whether to apply random flip images vertically along the horizontal axis (swap top and bottom)",
    )
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying vertical flip. "
            "A value of 0.5 means each image has a 50% chance to be flipped vertically."
        ),
    )


class RandomIOUCrop(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable random IoU crop",
        description=(
            "Whether to apply random cropping based on IoU criteria. "
            "A random crop is selected such that the minimum IoU with any object is above a threshold. "
            "Note: this augmentation is not supported when Tiling algorithm is enabled."
        ),
    )


class TopdownAffine(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable topdown affine",
        description="Whether to apply topdown affine transformations for keypoint detection",
    )
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Affine transforms probability",
        description=(
            "Probability of applying affine transformations for keypoint detection. "
            "A value of 1.0 means affine transformation is always applied."
        ),
    )


class GaussianBlur(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable Gaussian blur",
        description="Whether to apply Gaussian blur to the image",
    )
    kernel_size: int | None = Field(
        default=None,
        gt=0,
        title="Kernel size",
        description=(
            "Size of the Gaussian kernel. "
            "Larger kernel sizes result in stronger blurring. "
            "Must be a positive odd integer."
        ),
    )
    sigma: list[float] | None = Field(
        default=None,
        title="Sigma range",
        description=(
            "Range (min, max) of sigma values for Gaussian blur. "
            "Sigma controls the amount of blurring. "
            "A random value from this range will be used for each image."
        ),
    )
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying Gaussian blur. A value of 0.5 means each image has a 50% chance to be blurred."
        ),
    )

    @model_validator(mode="after")
    def validate_kernel_size(self) -> "GaussianBlur":
        if self.kernel_size is not None and self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        return self

    @model_validator(mode="after")
    def validate_sigma_range(self) -> "GaussianBlur":
        if self.sigma is None:
            return self
        if len(self.sigma) != 2:
            raise ValueError("sigma must be a list of exactly two float values")
        if self.sigma[0] >= self.sigma[1]:
            raise ValueError("The first value in sigma must be less than the second value")
        if self.sigma[0] < 0 or self.sigma[1] < 0:
            raise ValueError("Values in sigma must be non-negative")
        return self


class ColorJitter(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable color jitter",
        description="Whether to apply random color jitter to the image",
    )
    brightness: list[float] | None = Field(
        default=None,
        title="Brightness range",
        description=(
            "Range (min, max) of brightness adjustment factors. "
            "A random factor from this range will be multiplied with the image brightness. "
            "For example, (0.8, 1.2) means brightness can be reduced by 20% or increased by 20%."
        ),
    )
    contrast: list[float] | None = Field(
        default=None,
        title="Contrast range",
        description=(
            "Range (min, max) of contrast adjustment factors. "
            "A random factor from this range will be multiplied with the image contrast. "
            "For example, (0.5, 1.5) means contrast can be halved or increased by up to 50%."
        ),
    )
    saturation: list[float] | None = Field(
        default=None,
        title="Saturation range",
        description=(
            "Range (min, max) of saturation adjustment factors. "
            "A random factor from this range will be multiplied with the image saturation. "
            "For example, (0.5, 1.5) means saturation can be halved or increased by up to 50%."
        ),
    )
    hue: list[float] | None = Field(
        default=None,
        title="Hue range",
        description=(
            "Range (min, max) of hue adjustment values. "
            "A random value from this range will be added to the image hue. "
            "For example, (-0.05, 0.05) means hue can be shifted by up to ±0.05."
        ),
    )
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying color jitter. "
            "A value of 0.5 means each image has a 50% chance to be color jittered."
        ),
    )

    @model_validator(mode="after")
    def validate_brightness_range(self) -> "ColorJitter":
        if self.brightness is None:
            return self
        if len(self.brightness) != 2:
            raise ValueError("brightness must be a list of exactly two float values")
        if self.brightness[0] >= self.brightness[1]:
            raise ValueError("The first value in brightness must be less than the second value")
        if self.brightness[0] < 0:
            raise ValueError("Values in brightness must be non-negative")
        return self

    @model_validator(mode="after")
    def validate_contrast_range(self) -> "ColorJitter":
        if self.contrast is None:
            return self
        if len(self.contrast) != 2:
            raise ValueError("contrast must be a list of exactly two float values")
        if self.contrast[0] >= self.contrast[1]:
            raise ValueError("The first value in contrast must be less than the second value")
        if self.contrast[0] < 0:
            raise ValueError("Values in contrast must be non-negative")
        return self

    @model_validator(mode="after")
    def validate_saturation_range(self) -> "ColorJitter":
        if self.saturation is None:
            return self
        if len(self.saturation) != 2:
            raise ValueError("saturation must be a list of exactly two float values")
        if self.saturation[0] >= self.saturation[1]:
            raise ValueError("The first value in saturation must be less than the second value")
        if self.saturation[0] < 0:
            raise ValueError("Values in saturation must be non-negative")
        return self

    @model_validator(mode="after")
    def validate_hue_range(self) -> "ColorJitter":
        if self.hue is None:
            return self
        if len(self.hue) != 2:
            raise ValueError("hue must be a list of exactly two float values")
        if self.hue[0] >= self.hue[1]:
            raise ValueError("The first value in hue must be less than the second value")
        return self


class GaussianNoise(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable Gaussian noise",
        description="Whether to apply Gaussian noise to the image",
    )
    mean: float | None = Field(
        default=None,
        title="Mean",
        description=("Mean of the Gaussian noise to be added to the image. Typically set to 0.0 for zero-mean noise."),
    )
    sigma: float | None = Field(
        default=None,
        ge=0.0,
        title="Standard deviation",
        description=(
            "Standard deviation of the Gaussian noise. "
            "Controls the intensity of the noise. "
            "Higher values result in noisier images."
        ),
    )
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying Gaussian noise. "
            "A value of 0.5 means each image has a 50% chance to have noise added."
        ),
    )


class PhotometricDistort(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable photometric distort",
        description="Whether to apply photometric distortion to the image",
    )
    brightness_delta: int | None = Field(
        default=None,
        ge=0,
        title="Brightness delta",
        description=(
            "Maximum delta for brightness adjustment. "
            "A random value in [-brightness_delta, brightness_delta] will be added to the pixel values. "
            "For example, brightness_delta=32 means the pixel values can "
            "be randomly increased or decreased by up to 32. "
            "There is no strict upper limit, but values much larger than 32 may cause unnatural images."
        ),
    )
    contrast: list[float] | None = Field(
        default=None,
        title="Contrast range",
        description=(
            "Range of contrast adjustment factors. "
            "A random factor will be chosen from this range and multiplied with the pixel values. "
            "For example, (0.5, 1.5) means the image contrast can be halved or increased by up to 50%. "
            "Both values should be positive and reasonable to avoid extreme contrast changes."
        ),
    )
    saturation: list[float] | None = Field(
        default=None,
        title="Saturation range",
        description=(
            "Range of saturation adjustment factors. "
            "A random factor from this range will be applied to the image's saturation. "
            "For example, (0.5, 1.5) means saturation can be reduced by half or increased by up to 50%. "
            "Values should be positive and not extreme to keep images realistic."
        ),
    )
    hue_delta: int | None = Field(
        default=None,
        title="Hue delta",
        description=(
            "Maximum delta for hue adjustment. "
            "A random value in [-hue_delta, hue_delta] will be added to the hue channel. "
            "For example, hue_delta=18 means the hue can be shifted by up to ±18 units. "
            "There is no strict upper limit, but large values may cause unnatural color shifts."
        ),
    )
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Probability",
        description="Probability of applying photometric distortion",
    )

    @model_validator(mode="after")
    def validate_contrast_range(self) -> "PhotometricDistort":
        if self.contrast is None:
            return self
        if len(self.contrast) != 2:
            raise ValueError("contrast must be a list of exactly two float values")
        if self.contrast[0] >= self.contrast[1]:
            raise ValueError("The first value in contrast must be less than the second value")
        if self.contrast[0] <= 0 or self.contrast[1] <= 0:
            raise ValueError("Values in contrast must be positive")
        return self

    @model_validator(mode="after")
    def validate_saturation_range(self) -> "PhotometricDistort":
        if self.saturation is None:
            return self
        if len(self.saturation) != 2:
            raise ValueError("saturation must be a list of exactly two float values")
        if self.saturation[0] >= self.saturation[1]:
            raise ValueError("The first value in saturation must be less than the second value")
        if self.saturation[0] <= 0 or self.saturation[1] <= 0:
            raise ValueError("Values in saturation must be positive")
        return self


class Tiling(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable tiling",
        description="Whether to apply tiling to the image",
    )
    adaptive_tiling: bool | None = Field(
        default=False, title="Adaptive tiling", description="Whether to use adaptive tiling based on image content"
    )
    tile_size: int | None = Field(
        default=128,
        gt=0,
        title="Tile size",
        description=(
            "Size of each tile in pixels. "
            "Decreasing the tile size typically results in higher accuracy, "
            "but it is also more computationally expensive due to the higher number of tiles. "
            "In any case, the tile must be large enough to capture the entire object and its surrounding context, "
            "so choose a value larger than the size of most annotations."
        ),
    )
    tile_overlap: float | None = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        title="Tile overlap",
        description="Overlap between adjacent tiles as a fraction of tile size",
    )


class Mosaic(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable mosaic",
        description="Whether to apply mosaic augmentation (combines 4 images into one)",
    )


class Mixup(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable mixup",
        description="Whether to apply mixup augmentation (blends two images and their labels)",
    )
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Probability",
        description="Probability of applying mixup augmentation",
    )


class HSVRandomAug(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable HSV random augmentation",
        description="Whether to apply random HSV (Hue, Saturation, Value) augmentation",
    )
    hue_delta: int | None = Field(
        default=None,
        ge=0,
        title="Hue delta",
        description="Maximum delta for hue adjustment",
    )
    saturation_delta: int | None = Field(
        default=None,
        ge=0,
        title="Saturation delta",
        description="Maximum delta for saturation adjustment",
    )
    value_delta: int | None = Field(
        default=None,
        ge=0,
        title="Value delta",
        description="Maximum delta for value (brightness) adjustment",
    )


class RandomZoomOut(BaseModelNoExtra):
    enable: bool = Field(
        default=False,
        title="Enable random zoom out",
        description="Whether to apply random zoom out augmentation to the image",
    )
    fill: int | None = Field(
        default=None,
        ge=0,
        le=255,
        title="Fill value",
        description=(
            "Fill value for the area outside the image when zooming out. "
            "Typically 0 for black padding. Value should be between 0 and 255."
        ),
    )


class AugmentationParameters(BaseModelNoExtra):
    """Configuration parameters for data augmentation during training."""

    topdown_affine: TopdownAffine | None = Field(
        default=None, title="Topdown affine", description="Settings for topdown affine transformations"
    )
    random_zoom_out: RandomZoomOut | None = Field(
        default=None, title="Random zoom out", description="Settings for random zoom out augmentation"
    )
    iou_random_crop: RandomIOUCrop | None = Field(
        default=None,
        title="IoU random crop",
        description="Randomly crop images based on Intersection over Union (IoU) criteria",
    )
    mosaic: Mosaic | None = Field(default=None, title="Mosaic", description="Settings for mosaic augmentation")
    random_resize_crop: RandomResizeCrop | None = Field(
        default=None, title="Random resize crop", description="Settings for random resize and crop augmentation"
    )
    random_affine: RandomAffine | None = Field(
        default=None, title="Random affine", description="Settings for random affine transformations"
    )
    mixup: Mixup | None = Field(default=None, title="Mixup", description="Settings for mixup augmentation")
    hsv_random_aug: HSVRandomAug | None = Field(
        default=None, title="HSV random augmentation", description="Settings for HSV random augmentation"
    )
    random_horizontal_flip: RandomHorizontalFlip | None = Field(
        default=None,
        title="Random horizontal flip",
        description="Randomly flip images horizontally along the vertical axis (swap left and right)",
    )
    random_vertical_flip: RandomVerticalFlip | None = Field(
        default=None,
        title="Random vertical flip",
        description="Randomly flip images vertically along the horizontal axis (swap top and bottom)",
    )
    color_jitter: ColorJitter | None = Field(
        default=None,
        title="Color jitter",
        description="Settings for random color jitter (brightness, contrast, saturation, hue)",
    )
    gaussian_blur: GaussianBlur | None = Field(
        default=None, title="Gaussian blur", description="Settings for Gaussian blur augmentation"
    )
    photometric_distort: PhotometricDistort | None = Field(
        default=None, title="Photometric distort", description="Settings for photometric distortion augmentation"
    )
    gaussian_noise: GaussianNoise | None = Field(
        default=None, title="Gaussian noise", description="Settings for Gaussian noise augmentation"
    )
    tiling: Tiling | None = Field(default=None, title="Tiling", description="Settings for image tiling")
