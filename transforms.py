import torch
import numpy as np


class ToNumpyArray:
    def __call__(self, data):
        # Transform the data into a numpy array
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                raise ValueError(f"Unable to convert input to a NumPy array: {e}")
        # Convert the data to float32
        data = data.astype(np.float32)

        return data


class NormalizeArray:
    def __call__(self, data):
        # Step 1: Find the minimum and maximum values
        min_value = np.min(data)
        max_value = np.max(data)

        # Step 2: Normalize the image to the range [0, 1]
        data = (data - min_value) / (max_value - min_value)

        return data

class RotateIfNeeded:
    def __call__(self, data):
        height, width = data.shape
        if height > width:
            rotated_image = np.rot90(data)
            return rotated_image
        else:
            return data

class CropArray:
    def __call__(self, data):
        # Get the current dimensions
        height, width = data.shape[:2]

        # Calculate the new dimensions
        new_height = height - (height % 2)
        new_width = width - (width % 2)

        # Crop the array
        cropped_array = data[:new_height, :new_width, ...]

        return cropped_array

class GenerateN2NData:
    def __call__(self, data):
        h, w = data.shape

        # Generate separate standard deviations for input and label
        std_input = np.random.uniform(0, 0.2)
        std_label = np.random.uniform(0, 0.2)

        # Generate noise for input and label
        noise_input = np.random.normal(0, std_input, (h, w))
        noise_label = np.random.normal(0, std_label, (h, w))

        # Create noisy images for input and label
        input_data = np.array(data) + noise_input
        label_data = np.array(data) + noise_label

        # Clip values to be in the range [0, 1]
        input_data = np.clip(input_data, 0, 1).astype(np.float32)
        label_data = np.clip(label_data, 0, 1).astype(np.float32)

        return {'input': input_data[..., None], 'label': label_data[..., None], 'clean': data[..., None]}

class ToTensor:
    def __call__(self, data):
        if isinstance(data, dict):
            # Convert each value (assumed to be a numpy array or tensor) in the dictionary
            # to a PyTorch tensor and rearrange the axes
            return {key: torch.from_numpy(np.moveaxis(value.copy(), [0, 1, 2], [1, 2, 0])) 
                    if isinstance(value, np.ndarray) else value for key, value in data.items()}
        else:
            raise ValueError("Input must be a dictionary of tensors.")



class BackToNumpyArray:
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            # Convert the input tensor to a NumPy array and rearrange the axes
            numpy_array = data.detach().cpu().numpy()
            return numpy_array
        else:
            raise ValueError("Input must be a tensor.")



    

