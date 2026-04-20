namespace Onnxify.Safetensors;

public enum SafetensorErrorCode
{
    InvalidHeader = 1,
    InvalidHeaderDeserialization = 2,
    HeaderTooLarge = 3,
    HeaderTooSmall = 4,
    InvalidHeaderLength = 5,
    TensorNotFound = 6,
    TensorInvalidInfo = 7,
    InvalidOffset = 8,
    IoError = 9,
    JsonError = 10,
    InvalidTensorView = 11,
    MetadataIncompleteBuffer = 12,
    ValidationOverflow = 13,
    MisalignedSlice = 14,
}
