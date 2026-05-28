# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, model_validator


class BaseAugmentationParameter(BaseModel):
    enable: bool = Field(
        default=False,
        title="Enable",
        description="Toggle to apply this augmentation.",
    )


class RandomResizeCrop(BaseAugmentationParameter):
    crop_ratio_range: tuple[float, float] = Field(
        title="Crop resize ratio range",
        description=(
            "Range (min, max) of crop ratios to apply during resize crop operation. "
            "Specifies the fraction of the original image dimensions to retain after cropping. "
            "For example, (0.8, 1.0) will randomly crop between 80% and 100% of the original size. "
            "Both values should be between 0.0 and 1.0."
        ),
        json_schema_extra={"min_value": 0.0, "max_value": 1.0},
    )
    aspect_ratio_range: tuple[float, float] = Field(
        title="Aspect ratio range",
        description=(
            "Range (min, max) of aspect ratios to apply during resize crop operation. "
            "Aspect ratio is defined as width divided by height. "
            "For example, (0.75, 1.33) allows the crop to have an aspect ratio between 3:4 and 4:3."
        ),
        json_schema_extra={"min_value": 0.0, "max_value": 10.0},
    )

    @model_validator(mode="after")
    def validate_crop_range(self) -> "RandomResizeCrop":
        if len(self.crop_ratio_range) != 2:
            raise ValueError("crop_ratio_range must be a list of exactly two float values")
        if self.crop_ratio_range[0] >= self.crop_ratio_range[1]:
            raise ValueError("The first value in crop_ratio_range must be less than the second value")
        if not (0 <= self.crop_ratio_range[0] <= 1) or not (0 <= self.crop_ratio_range[1] <= 1):
            raise ValueError("Values in crop_ratio_range must be between 0 and 1")
        return self

    @model_validator(mode="after")
    def validate_aspect_ratio_range(self) -> "RandomResizeCrop":
        if len(self.aspect_ratio_range) != 2:
            raise ValueError("aspect_ratio_range must be a list of exactly two float values")
        if self.aspect_ratio_range[0] >= self.aspect_ratio_range[1]:
            raise ValueError("The first value in aspect_ratio_range must be less than the second value")
        if self.aspect_ratio_range[0] <= 0 or self.aspect_ratio_range[1] <= 0:
            raise ValueError("Values in aspect_ratio_range must be greater than 0")
        return self


class RandomAffine(BaseAugmentationParameter):
    max_rotate_degree: float = Field(
        ge=0.0,
        title="Rotation degrees",
        description=(
            "Maximum rotation angle in degrees for affine transformation. "
            "A random angle in the range [-max_rotate_degree, max_rotate_degree] will be applied. "
            "For example, max_rotate_degree=10 allows up to ±10 degrees rotation."
        ),
    )
    max_translate_ratio: float = Field(
        ge=0.0,
        lt=1.0,
        title="Horizontal translation",
        description=(
            "Maximum translation as a fraction of image width or height. "
            "A random translation in the range [-max_translate_ratio, max_translate_ratio] "
            "will be applied along both axes. For example, 0.1 allows up to ±10% translation."
        ),
    )
    scaling_ratio_range: tuple[float, float] = Field(
        title="Scaling ratio range",
        description=(
            "Range (min, max) of scaling factors to apply during affine transformation. "
            "Both values should be > 0.0. "
            "For example, (0.8, 1.2) will randomly scale the image between 80% and 120% of its original size."
        ),
        json_schema_extra={"min_value": 0.0, "max_value": 10.0},
    )
    max_shear_degree: float = Field(
        title="Maximum shear degree",
        description=(
            "Maximum absolute shear angle in degrees to apply during affine transformation. "
            "A random shear in the range [-max_shear_degree, max_shear_degree] will be applied."
        ),
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        title="Probability",
        description=(
            "Probability of applying the affine transformation. "
            "A value of 0.5 means each image has a 50% chance to be transformed."
        ),
    )

    @model_validator(mode="after")
    def validate_scaling_ratio_range(self) -> "RandomAffine":
        if len(self.scaling_ratio_range) != 2:
            raise ValueError("scaling_ratio_range must be a list of exactly two float values")
        if self.scaling_ratio_range[0] > self.scaling_ratio_range[1]:
            raise ValueError("The first value in scaling_ratio_range must be less than or equal to the second value")
        if self.scaling_ratio_range[0] <= 0 or self.scaling_ratio_range[1] <= 0:
            raise ValueError("Values in scaling_ratio_range must be greater than 0")
        return self


class RandomHorizontalFlip(BaseAugmentationParameter):
    probability: float = Field(
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying horizontal flip. "
            "A value of 0.5 means each image has a 50% chance to be flipped horizontally."
        ),
    )


class RandomVerticalFlip(BaseAugmentationParameter):
    probability: float = Field(
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying vertical flip. "
            "A value of 0.5 means each image has a 50% chance to be flipped vertically."
        ),
    )


class RandomIOUCrop(BaseAugmentationParameter):
    probability: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        title="Probability",
        description=(
            "Probability of applying IoU random crop. A value of 1.0 means the crop is always applied when enabled."
        ),
    )
    min_scale: float = Field(
        gt=0.0,
        le=1.0,
        default=0.3,
        title="Minimum scale",
        description=(
            "Minimum fraction of the original image area to retain after cropping. "
            "For example, 0.3 means the crop will be at least 30% of the original area."
        ),
    )
    max_scale: float = Field(
        gt=0.0,
        le=1.0,
        default=1.0,
        title="Maximum scale",
        description=(
            "Maximum fraction of the original image area to retain after cropping. "
            "A value of 1.0 means the crop can be as large as the entire image."
        ),
    )

    @model_validator(mode="after")
    def validate_scale_range(self) -> "RandomIOUCrop":
        if self.min_scale >= self.max_scale:
            raise ValueError("min_scale must be less than max_scale")
        return self


class GaussianBlur(BaseAugmentationParameter):
    kernel_size: int = Field(
        gt=0,
        title="Kernel size",
        description=(
            "Size of the Gaussian kernel. "
            "Larger kernel sizes result in stronger blurring. "
            "Must be a positive odd integer."
        ),
    )
    sigma: tuple[float, float] = Field(
        title="Sigma range",
        description=(
            "Range (min, max) of sigma values for Gaussian blur. "
            "Sigma controls the amount of blurring. "
            "A random value from this range will be used for each image."
        ),
        json_schema_extra={"min_value": 0.0, "max_value": 10.0},
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying Gaussian blur. A value of 0.5 means each image has a 50% chance to be blurred."
        ),
    )

    @model_validator(mode="after")
    def validate_kernel_size(self) -> "GaussianBlur":
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        return self

    @model_validator(mode="after")
    def validate_sigma_range(self) -> "GaussianBlur":
        if len(self.sigma) != 2:
            raise ValueError("sigma must be a list of exactly two float values")
        if self.sigma[0] >= self.sigma[1]:
            raise ValueError("The first value in sigma must be less than the second value")
        if self.sigma[0] < 0 or self.sigma[1] < 0:
            raise ValueError("Values in sigma must be non-negative")
        return self


class ColorJitter(BaseAugmentationParameter):
    brightness: tuple[float, float] = Field(
        title="Brightness range",
        description=(
            "Range (min, max) of brightness adjustment factors. "
            "A random factor from this range will be multiplied with the image brightness. "
            "For example, (0.8, 1.2) means brightness can be reduced by 20% or increased by 20%."
        ),
        json_schema_extra={"min_value": 0.0, "max_value": 5.0},
    )
    contrast: tuple[float, float] = Field(
        title="Contrast range",
        description=(
            "Range (min, max) of contrast adjustment factors. "
            "A random factor from this range will be multiplied with the image contrast. "
            "For example, (0.5, 1.5) means contrast can be halved or increased by up to 50%."
        ),
        json_schema_extra={"min_value": 0.0, "max_value": 5.0},
    )
    saturation: tuple[float, float] = Field(
        title="Saturation range",
        description=(
            "Range (min, max) of saturation adjustment factors. "
            "A random factor from this range will be multiplied with the image saturation. "
            "For example, (0.5, 1.5) means saturation can be halved or increased by up to 50%."
        ),
        json_schema_extra={"min_value": 0.0, "max_value": 5.0},
    )
    hue: tuple[float, float] = Field(
        title="Hue range",
        description=(
            "Range (min, max) of hue adjustment values. "
            "A random value from this range will be added to the image hue. "
            "For example, (-0.05, 0.05) means hue can be shifted by up to ±0.05."
        ),
        json_schema_extra={"min_value": -0.5, "max_value": 0.5},
    )
    probability: float = Field(
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
        if len(self.brightness) != 2:
            raise ValueError("brightness must be a list of exactly two float values")
        if self.brightness[0] >= self.brightness[1]:
            raise ValueError("The first value in brightness must be less than the second value")
        if self.brightness[0] < 0:
            raise ValueError("Values in brightness must be non-negative")
        return self

    @model_validator(mode="after")
    def validate_contrast_range(self) -> "ColorJitter":
        if len(self.contrast) != 2:
            raise ValueError("contrast must be a list of exactly two float values")
        if self.contrast[0] >= self.contrast[1]:
            raise ValueError("The first value in contrast must be less than the second value")
        if self.contrast[0] < 0:
            raise ValueError("Values in contrast must be non-negative")
        return self

    @model_validator(mode="after")
    def validate_saturation_range(self) -> "ColorJitter":
        if len(self.saturation) != 2:
            raise ValueError("saturation must be a list of exactly two float values")
        if self.saturation[0] >= self.saturation[1]:
            raise ValueError("The first value in saturation must be less than the second value")
        if self.saturation[0] < 0:
            raise ValueError("Values in saturation must be non-negative")
        return self

    @model_validator(mode="after")
    def validate_hue_range(self) -> "ColorJitter":
        if len(self.hue) != 2:
            raise ValueError("hue must be a list of exactly two float values")
        if self.hue[0] >= self.hue[1]:
            raise ValueError("The first value in hue must be less than the second value")
        return self


class GaussianNoise(BaseAugmentationParameter):
    mean: float = Field(
        title="Mean",
        description=("Mean of the Gaussian noise to be added to the image. Typically set to 0.0 for zero-mean noise."),
    )
    sigma: float = Field(
        ge=0.0,
        title="Standard deviation",
        description=(
            "Standard deviation of the Gaussian noise. "
            "Controls the intensity of the noise. "
            "Higher values result in noisier images."
        ),
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying Gaussian noise. "
            "A value of 0.5 means each image has a 50% chance to have noise added."
        ),
    )


class Tiling(BaseAugmentationParameter):
    enable_adaptive_tiling: bool = Field(
        default=False, title="Adaptive tiling", description="Whether to use adaptive tiling based on image content"
    )
    tile_size: int = Field(
        default=128,
        ge=64,
        title="Tile size",
        description=(
            "Size of each tile in pixels. "
            "Decreasing the tile size typically results in higher accuracy, "
            "but it is also more computationally expensive due to the higher number of tiles. "
            "In any case, the tile must be large enough to capture the entire object and its surrounding context, "
            "so choose a value larger than the size of most annotations."
        ),
    )
    tile_overlap: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        title="Tile overlap",
        description="Overlap between adjacent tiles as a fraction of tile size",
    )


class Mosaic(BaseAugmentationParameter):
    probability: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        title="Probability",
        description=(
            "Probability of applying mosaic augmentation. A value of 1.0 means mosaic is always applied when enabled."
        ),
    )
    scale: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        title="Scale",
        description=(
            "Scale factor for random perspective crop applied after mosaic assembly. "
            "The random scale is sampled from [1-scale, 1+scale]. "
            "Higher values produce more zoom variation."
        ),
    )
    translate: float = Field(
        ge=0.0,
        le=0.5,
        default=0.1,
        title="Translate",
        description=(
            "Translation fraction for random perspective crop. "
            "The crop center shifts by up to ±translate * image_size pixels."
        ),
    )


class Mixup(BaseAugmentationParameter):
    probability: float = Field(
        ge=0.0,
        le=1.0,
        title="Probability",
        description="Probability of applying mixup augmentation",
    )
    alpha: float = Field(
        ge=0.1,
        le=10.0,
        default=1.5,
        title="Alpha",
        description=(
            "Controls how two images are blended together. "
            "Low values (e.g. 0.5) produce uneven blending where one image dominates. "
            "A value of 1.0 gives any blend ratio equal chance. "
            "Higher values (e.g. 1.5-3.0) favour an equal 50/50 mix of both images."
        ),
    )


class RandomErasing(BaseAugmentationParameter):
    scale: tuple[float, float] = Field(
        title="Erasing area scale range",
        description=(
            "Range (min, max) of the proportion of the image area to erase. "
            "For example, (0.02, 0.33) means erasing between 2% and 33% of the image area."
        ),
        json_schema_extra={"min_value": 0.0, "max_value": 1.0},
    )
    ratio: tuple[float, float] = Field(
        title="Erasing aspect ratio range",
        description=(
            "Range (min, max) of the aspect ratio of the erased area. "
            "For example, (0.3, 3.3) allows the erased rectangle to have varying proportions."
        ),
        json_schema_extra={"min_value": 0.0, "max_value": 10.0},
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying random erasing. "
            "A value of 0.5 means each image has a 50% chance to have a region erased."
        ),
    )
    value: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        title="Fill value",
        description=(
            "Fill value for the erased region, normalized to [0, 1]. "
            "A value of 0.0 fills with black. A value of 1.0 fills with white."
        ),
    )

    @model_validator(mode="after")
    def validate_scale_range(self) -> "RandomErasing":
        if len(self.scale) != 2:
            raise ValueError("scale must be a list of exactly two float values")
        if self.scale[0] >= self.scale[1]:
            raise ValueError("The first value in scale must be less than the second value")
        if self.scale[0] < 0 or self.scale[1] > 1:
            raise ValueError("Values in scale must be between 0 and 1")
        return self

    @model_validator(mode="after")
    def validate_ratio_range(self) -> "RandomErasing":
        if len(self.ratio) != 2:
            raise ValueError("ratio must be a list of exactly two float values")
        if self.ratio[0] >= self.ratio[1]:
            raise ValueError("The first value in ratio must be less than the second value")
        if self.ratio[0] <= 0:
            raise ValueError("Values in ratio must be greater than 0")
        return self


class RandomGrayscale(BaseAugmentationParameter):
    probability: float = Field(
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of converting the image to grayscale. "
            "A value of 0.1 means each image has a 10% chance to be converted to grayscale."
        ),
    )


class RandomSharpness(BaseAugmentationParameter):
    sharpness: float = Field(
        ge=0.0,
        title="Sharpness factor",
        description=(
            "Factor controlling the strength of the sharpness adjustment. "
            "A value of 0.0 means no sharpening, higher values increase the effect. "
            "Typical values are between 0.0 and 1.0."
        ),
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        title="Probability",
        description=(
            "Probability of applying sharpness adjustment. "
            "A value of 0.5 means each image has a 50% chance to be sharpened."
        ),
    )


class RandomZoomOut(BaseAugmentationParameter):
    fill: int = Field(
        ge=0,
        le=255,
        title="Fill value",
        description=(
            "Fill value for the area outside the image when zooming out. "
            "Typically 0 for black padding. Value should be between 0 and 255."
        ),
    )
    side_range: tuple[float, float] = Field(
        default=(1.0, 4.0),
        title="Side range",
        description=(
            "Range (min, max) of the zoom-out scale factor for each side. "
            "The image will be placed on a canvas scaled by a random factor from this range. "
            "For example, (1.0, 4.0) means the canvas can be up to 4x the original image size. "
            "The minimum value must be >= 1.0."
        ),
        json_schema_extra={"min_value": 1.0, "max_value": 16.0},
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        title="Probability",
        description=(
            "Probability of applying random zoom out. "
            "A value of 0.5 means each image has a 50% chance to be zoomed out."
        ),
    )

    @model_validator(mode="after")
    def validate_side_range(self) -> "RandomZoomOut":
        if len(self.side_range) != 2:
            raise ValueError("side_range must be a tuple of exactly two float values")
        if self.side_range[0] >= self.side_range[1]:
            raise ValueError("The first value in side_range must be less than the second value")
        if self.side_range[0] < 1.0:
            raise ValueError("The minimum side_range value must be >= 1.0")
        return self


class AugmentationParameters(BaseModel):
    """
    Configuration parameters for data augmentation during training.

    Each field corresponds to a specific augmentation technique and contains settings for that technique.
    If the field is set to None, it means that the augmentation is not applicable to the model architecture.
    Each augmentation technique can be enabled or disabled using the 'enable' field within its respective configuration.
    """

    deim_framework: bool | None = Field(
        default=None,
        title="DEIM framework",
        description=(
            "Controls the DEIM multi-stage augmentation scheduling framework. "
            "When enabled, training uses complex multi-stage augmentations that "
            "automatically transition between strong and light augmentation phases "
            "during training. This may lead to better accuracy but increases "
            "training time. When disabled, a simpler static augmentation pipeline "
            "is used instead, allowing direct control over individual augmentations. "
            "WARNING: when the DEIM framework is enabled, all other augmentation "
            "parameters below are ignored and will have no effect on training."
        ),
    )
    random_zoom_out: RandomZoomOut | None = Field(
        default=None,
        title="Random zoom out",
        description=(
            "Randomly zoom out the image by placing it on a larger canvas with padding. Applied before resize."
        ),
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    iou_random_crop: RandomIOUCrop | None = Field(
        default=None,
        title="IoU random crop",
        description=(
            "Randomly crop images based on Intersection over Union (IoU) criteria. "
            "Applied before resize. "
            "Note: this augmentation is not supported when Tiling algorithm is enabled."
        ),
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    mosaic: Mosaic | None = Field(
        default=None,
        title="Mosaic",
        description="Combines 4 images into one mosaic for augmentation. Applied before resize.",
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    random_resize_crop: RandomResizeCrop | None = Field(
        default=None,
        title="Random resize crop",
        description=(
            "Randomly resize and crop the image. Applied instead of resize. "
            "When disabled, a standard resize to the target input size is used instead. "
            "Note: this augmentation is not supported when Tiling algorithm is enabled."
        ),
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    random_affine: RandomAffine | None = Field(
        default=None,
        title="Random affine",
        description=(
            "Apply random affine transformations (rotation, translation, scaling, shear) to the image. "
            "Applied after resize."
        ),
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    mixup: Mixup | None = Field(
        default=None,
        title="Mixup",
        description="Blends two images and their labels for augmentation. Applied before resize.",
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    random_horizontal_flip: RandomHorizontalFlip | None = Field(
        default=None,
        title="Random horizontal flip",
        description=(
            "Randomly flip images horizontally along the vertical axis (swap left and right). Applied after resize."
        ),
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    random_vertical_flip: RandomVerticalFlip | None = Field(
        default=None,
        title="Random vertical flip",
        description=(
            "Randomly flip images vertically along the horizontal axis (swap top and bottom). Applied after resize."
        ),
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    color_jitter: ColorJitter | None = Field(
        default=None,
        title="Color jitter",
        description="Randomly adjust brightness, contrast, saturation, and hue of the image. Applied after resize.",
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    gaussian_blur: GaussianBlur | None = Field(
        default=None,
        title="Gaussian blur",
        description="Apply Gaussian blur to the image. Applied after resize.",
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    gaussian_noise: GaussianNoise | None = Field(
        default=None,
        title="Gaussian noise",
        description="Add Gaussian noise to the image. Applied after resize.",
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    random_erasing: RandomErasing | None = Field(
        default=None,
        title="Random erasing",
        description=(
            "Randomly erase a rectangular region in the image and fill it with a constant value. "
            "Also known as Cutout. Helps the model learn to rely on broader context rather than "
            "specific local features. Applied after resize."
        ),
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    random_grayscale: RandomGrayscale | None = Field(
        default=None,
        title="Random grayscale",
        description=(
            "Randomly convert the image to grayscale. Forces the model to learn shape and texture "
            "features rather than relying solely on color information. Applied after resize."
        ),
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    random_sharpness: RandomSharpness | None = Field(
        default=None,
        title="Random sharpness",
        description=(
            "Randomly adjust the sharpness of the image. Complements Gaussian blur by also "
            "allowing images to become sharper, improving robustness to varying image quality. "
            "Applied after resize."
        ),
        json_schema_extra={"depends_on": {"deim_framework": [False, None]}},
    )
    tiling: Tiling | None = Field(
        default=None,
        title="Tiling",
        description="Split images into overlapping tiles for processing, useful for detecting small objects.",
    )
