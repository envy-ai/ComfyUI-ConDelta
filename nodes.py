import torch
import safetensors.torch
import nodes
import logging
import folder_paths
import os

condelta_dir = os.path.join(folder_paths.models_dir, "ConDelta")

try:
    if condelta_dir not in folder_paths.get_folder_paths("ConDelta"):
        raise KeyError
except KeyError:
    try:
        os.mkdir(condelta_dir)
    except OSError:
        # Already exists
        pass

    folder_paths.add_model_folder_path("ConDelta", condelta_dir)

def tensor_mean(tensor):
    return torch.mean(torch.abs(torch.flatten(tensor, start_dim=1))).item()

def tensor_max(tensor):
    return torch.max(torch.abs(torch.flatten(tensor, start_dim=1))).item()

def tensor_median(tensor):
    return torch.median(torch.abs(torch.flatten(tensor, start_dim=1))).item()

def tensor_ratio(tensor, type="mean"):
    if type == "mean":
        return tensor_mean(tensor)
    elif type == "max":
        return tensor_max(tensor)
    elif type == "median":
        return tensor_median(tensor)
    else:
        raise ValueError("Invalid type: " + type)
    
# Various experiments with noise to see if I can find something that is predictable and useful
def tensor_mask(tensor1, tensor2):
    """
    Clips all the values in tensor1 to the magnitude of the same values in tensor2.
    """
    return torch.sign(tensor1) * torch.minimum(torch.abs(tensor1), torch.abs(tensor2))

def tenser_add_gaussian_noise(tensor, strength):
    """
    Adds gaussian noise to the tensor, multiplied by the strength.
    """
    return tensor + torch.randn_like(tensor) * strength

def tensor_add_multiplied_gaussian_noise(tensor1, strength):
    """
    Makes a tensor of gaussian noise, then multiplies each element by the absolute value of
    each corresponding element in tensor1. This is multiplied by strength and added to tensor1.
    """
    return tensor1 + torch.randn_like(tensor1) * torch.abs(tensor1) * strength

def tensor_get_type2_noise(tensor, strength, seed=None):
    """
    Makes a tensor of gaussian noise, clips it with tanh, then multiplies each element by the 
    absolute value of each corresponding element in tensor1. This is multiplied by strength
    and returned
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.tanh(torch.randn_like(tensor)) * torch.abs(tensor) * strength

def tensor_add_type2_noise(tensor, strength, seed=None):
    """
    Makes a tensor of gaussian noise, clips it with tanh, then multiplies each element by the 
    absolute value of each corresponding element in tensor1. This is multiplied by strength 
    and added to tensor1.
    """
    return tensor + tensor_get_type2_noise(tensor, strength, seed)

def tensor_get_type1_noise(tensor, strength, seed=None):
    """
    Makes a tensor of gaussian noise, then multiplies each element by the absolute value of
    each corresponding element in tensor. This is multiplied by strength and returned
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.randn_like(tensor) * torch.abs(tensor) * strength

def tensor_add_type1_noise(tensor, strength, seed=None):
    """
    Makes a tensor of gaussian noise, then multiplies each element by the absolute value of
    each corresponding element in tensor. This is multiplied by strength and added to tensor
    """
    return tensor + tensor_get_type1_noise(tensor, strength, seed)

def tensor_get_type3_noise(tensor, strength, seed=None):
    """
    Makes a tensor of gaussian noise, takes the absolute value, then multiples each element
    by each value in the given tensor. This is multiplied by strength.
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.abs(torch.randn_like(tensor)) * tensor * strength

def tensor_add_type3_noise(tensor, strength, seed=None):
    """
    Makes a tensor of gaussian noise, takes the absolute value, then multiples each element
    by each value in the given tensor. This is multiplied by strength and added to the tensor.
    """
    return tensor + tensor_get_type3_noise(tensor, strength, seed)

def tensor_get_type4_noise(tensor, strength, seed=None):
    """
    Makes a tensor of gaussian noise, takes the absolute value, runs tanh on it, then multiples
    each element by each value in the given tensor. This is multiplied by strength.
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.tanh(torch.abs(torch.randn_like(tensor))) * tensor * strength

def tensor_add_type4_noise(tensor, strength, seed=None):
    """
    Makes a tensor of gaussian noise, takes the absolute value, runs tanh on it, then multiples
    each element by each value in the given tensor. This is multiplied by strength and added to the tensor.
    """
    return tensor + tensor_get_type4_noise(tensor, strength, seed)

class ConditioningGetNoise:
    """
    This generates type1 or type2 noise based on the input conditioning and strength
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ), 
                             "strength": ("FLOAT", {"default": 1.0, "step": 0.01}),
                             "type": (["type1", "type2", "type3", "type4"], {"default": "type1"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "getNoise"

    CATEGORY = "conditioning"

    def getNoise(self, conditioning, strength, type, seed):
        out = []

        noise_functions = {
            "type1": tensor_get_type1_noise,
            "type2": tensor_get_type2_noise,
            "type3": tensor_get_type3_noise,
            "type4": tensor_get_type4_noise,
        }

        noise_function = noise_functions[type]

        for i in range(len(conditioning)):
            t1 = conditioning[i][0]
            pooled_output = conditioning[i][1].get("pooled_output", None)
            tw = noise_function(t1, strength, seed)
            if pooled_output is not None:
                pooled_output = noise_function(pooled_output, strength, seed)

            t_to = conditioning[i][1].copy()
            if pooled_output is not None:
                t_to["pooled_output"] = pooled_output

            n = [tw, t_to]
            out.append(n)
        return (out, )

class MaskConDelta:
    """
    This node masks the conditioning delta with another conditioning.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_delta": ("CONDITIONING", ),
                "mask": ("CONDITIONING", )
            }
        }
    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "mask"
    OUTPUT_NODE = True

    CATEGORY = "conditioning"

    def mask(self, conditioning_delta, mask):
        out = []

        for i in range(len(conditioning_delta)):
            t1 = conditioning_delta[i][0]
            pooled_output = conditioning_delta[i][1].get("pooled_output", None)
            t0 = mask[i][0]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = tensor_mask(t1, t0)
            t_to = conditioning_delta[i][1].copy()
            mask_pooled_output = mask[i][1].get("pooled_output", None)
            if pooled_output is not None:
                t_to["pooled_output"] = tensor_mask(pooled_output, mask_pooled_output)

            n = [tw, t_to]
            out.append(n)
        return (out, )

class SaveConditioningDelta:
    """
    This node saves the conditioning as a safetensors file for later use.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_delta": ("CONDITIONING", ),
                "file_name": ("STRING", ),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "If true, will overwrite the file if it already exists."})
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "conditioning"

    def save(self, conditioning_delta, file_name, overwrite):
        # remove .safetenors from the file_name
        if file_name.endswith(".safetensors"):
            file_name = file_name[:-12]

        #file_path = sys.path[0] + f"/models/ConDelta/{file_name}.safetensors"
        file_path = os.path.join(condelta_dir, f"{file_name}.safetensors")
        print("Saving conditioning delta to file: ", file_path)

        # Throw exception if file already exists
        if not overwrite and os.path.exists(file_path):        
            raise Exception(f"File already exists: {file_path}")
        #print(conditioning_delta)

        # Extract the tensor data from the conditioning delta
        tensor_data = conditioning_delta[0][0]
        pooled_output = conditioning_delta[0][1].get("pooled_output", None)
        #print(tensor_data)
        # Save the tensor data to a safetensors file
        try:
            safetensors.torch.save_file({"conditioning_delta": tensor_data, "pooled_output": pooled_output}, file_path)
        except Exception as e:
            print("Error saving conditioning delta: ", e)
        return ()
    
class LoadConditioningDelta:
    """
    This node loads the conditioning delta from a safetensors file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condelta": (folder_paths.get_filename_list("ConDelta"), {"tooltip": "The name of the LoRA."}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "load"

    CATEGORY = "conditioning"

    def load(self, condelta):
        #file_path = sys.path[0] + f"/models/ConDelta/{file_name}"
        file_path = os.path.join(condelta_dir, condelta)
        print("Loading conditioning delta from file: ", file_path)
        # Load the tensor data from the safetensors file
        tensor_data = safetensors.torch.load_file(file_path)["conditioning_delta"]
        pooled_output = safetensors.torch.load_file(file_path)["pooled_output"]
        # Wrap the tensor data in the expected format
        if pooled_output is not None:
            conditioning_delta = [[tensor_data, {"pooled_output": pooled_output}]]
        else:
            conditioning_delta = [[tensor_data, {}]]
        #print(conditioning_delta)
        return (conditioning_delta,)

# Load a conditioning delta from a file and apply it to the conditioning
class ApplyConDelta:
    """
    This node applies a condelta to a conditioning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ), 
                             "condelta": (folder_paths.get_filename_list("ConDelta"), {"tooltip": "The name of the LoRA."}),
                            "strength": ("FLOAT", {"default": 1.0, "step": 0.01})}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"

    CATEGORY = "conditioning"

    def apply(self, conditioning, condelta, strength):
        out = []
        file_path = os.path.join(condelta_dir, condelta)
        print("Loading conditioning delta from file: ", file_path)

        # Load the tensor data from the safetensors file
        tensor_data = safetensors.torch.load_file(file_path)["conditioning_delta"]
        pooled_output = safetensors.torch.load_file(file_path)["pooled_output"]

        # Wrap the tensor data in the expected format
        if pooled_output is not None:
            conditioning_delta = [[tensor_data, {"pooled_output": pooled_output}]]
        else:
            conditioning_delta = [[tensor_data, {}]]

        if len(conditioning_delta) > 1:
            logging.warning("Warning: ApplyConditioningDelta conditioning_delta contains more than 1 cond, only the first one will actually be applied to conditioning.")

        cond_delta = conditioning_delta[0][0]
        pooled_output_from = conditioning_delta[0][1].get("pooled_output", None)

        for i in range(len(conditioning)):
            t1 = conditioning[i][0]
            pooled_output_to = conditioning[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_delta[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = t1 + torch.mul(t0, strength)
            t_to = conditioning[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = pooled_output_to + torch.mul(pooled_output_from, strength)
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return (out, )

class ApplyConDeltaAutoScale:
    """
    This node applies a conditioning delta to a conditioning, scaling the delta to match the conditioning.
    It does this by determining the ratio of the means of the two vectors.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ), 
                             "condelta": (folder_paths.get_filename_list("ConDelta"), {"tooltip": "The name of the LoRA."}),
                             "ratio_type": (["mean", "max", "median"], {"default": "median"}),
                            "strength": ("FLOAT", {"default": 1.0, "step": 0.01}),
                            "clamp": ("BOOLEAN", {"default": False, "tooltip": "If true, will clamp the ConDelta's values."}),
                            "clamp_value": ("FLOAT", {"default": 3.0, "step": 0.01})
                            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"

    CATEGORY = "conditioning"

    def apply(self, conditioning, ratio_type, condelta, strength, clamp, clamp_value):
        out = []
        file_path = os.path.join(condelta_dir, condelta)
        print("Loading conditioning delta from file: ", file_path)

        # Load the tensor data from the safetensors file
        tensor_data = safetensors.torch.load_file(file_path)["conditioning_delta"]
        pooled_output = safetensors.torch.load_file(file_path)["pooled_output"]

        # Wrap the tensor data in the expected format
        if pooled_output is not None:
            conditioning_delta = [[tensor_data, {"pooled_output": pooled_output}]]
        else:
            conditioning_delta = [[tensor_data, {}]]

        if len(conditioning_delta) > 1:
            logging.warning("Warning: ApplyConditioningDelta conditioning_delta contains more than 1 cond, only the first one will actually be applied to conditioning.")

        cond_delta = conditioning_delta[0][0]
        pooled_output_from = conditioning_delta[0][1].get("pooled_output", None)

        if clamp:
            cond_delta = torch.clamp(cond_delta, -clamp_value, clamp_value)
            pooled_output_from = torch.clamp(pooled_output_from, -clamp_value, clamp_value)

        for i in range(len(conditioning)):
            t1 = conditioning[i][0]
            pooled_output_to = conditioning[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_delta[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            mean_ratio = tensor_ratio(t1, ratio_type) / tensor_ratio(t0, ratio_type)

            tw = t1 + torch.mul(t0, strength * mean_ratio)
            t_to = conditioning[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                mean_ratio_pooled = tensor_ratio(pooled_output_to, ratio_type) / tensor_ratio(pooled_output_from, ratio_type)
                t_to["pooled_output"] = pooled_output_to + torch.mul(pooled_output_from, strength * mean_ratio_pooled)
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return (out, )


class ConditioningScale:
    """
    This node multiplies the conditioning by a scalar.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ), "scalar": ("FLOAT", {"default": 1.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "multiply"

    CATEGORY = "conditioning"

    def multiply(self, conditioning, scalar):
        out = []

        for i in range(len(conditioning)):
            t1 = conditioning[i][0]
            pooled_output = conditioning[i][1].get("pooled_output", None)
            tw = torch.mul(t1, scalar)
            t_to = conditioning[i][1].copy()
            if pooled_output is not None:
                t_to["pooled_output"] = torch.mul(pooled_output, scalar)

            n = [tw, t_to]
            out.append(n)
        return (out, )

class ConditioningSubtract:
    """
    This node subtracts one conditioning from another (A-B), producing a conditioning delta.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_a": ("CONDITIONING", ), "conditioning_b": ("CONDITIONING", ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "subtract"

    CATEGORY = "conditioning"

    def subtract(self, conditioning_a, conditioning_b):
        out = []

        cond_b = conditioning_b[0][0]
        pooled_output_b = conditioning_b[0][1].get("pooled_output", None)

        for i in range(len(conditioning_a)):
            t1 = conditioning_a[i][0]
            pooled_output_a = conditioning_a[i][1].get("pooled_output", pooled_output_b)
            t0 = cond_b[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = t1 + torch.mul(t0, -1)
            t_to = conditioning_a[i][1].copy()
            if pooled_output_b is not None and pooled_output_a is not None:
                t_to["pooled_output"] = pooled_output_a + torch.mul(pooled_output_b, -1)
            elif pooled_output_b is not None:
                t_to["pooled_output"] = pooled_output_b

            n = [tw, t_to]
            out.append(n)
        return (out, )

class ThresholdConditioning:
    """
    This node applies a threshold to the conditioning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ), "threshold": ("FLOAT", {"default": 0.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "threshold"

    CATEGORY = "conditioning"

    def threshold(self, conditioning, threshold):
        out = []

        for i in range(len(conditioning)):
            t1 = conditioning[i][0]
            print(t1.size())
            print(t1)
            pooled_output = conditioning[i][1].get("pooled_output", None)
            print(pooled_output.size())
            tw = torch.where(torch.abs(t1) < threshold, 0, t1)
            t_to = conditioning[i][1].copy()
            if pooled_output is not None:
                t_to["pooled_output"] = torch.where(torch.abs(pooled_output) < threshold, 0, pooled_output)

            n = [tw, t_to]
            out.append(n)
        return (out, )
    
class ClampConDelta:
    """
    Use tanh to clip the conditioning between -1 and 1.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "clip"

    CATEGORY = "conditioning"

    def clip(self, conditioning):
        out = []

        for i in range(len(conditioning)):
            t1 = conditioning[i][0]
            pooled_output = conditioning[i][1].get("pooled_output", None)
            tw = torch.tanh(t1)
            t_to = conditioning[i][1].copy()
            if pooled_output is not None:
                t_to["pooled_output"] = torch.tanh(pooled_output)

            n = [tw, t_to]
            out.append(n)
        return (out, )
    
class HardClampConDelta:
    """
    Use clamp to clip the conditioning between -strength and strength.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ), "strength": ("FLOAT", {"default": 1.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "hardclip"

    CATEGORY = "conditioning"

    def hardclip(self, conditioning, strength):
        out = []

        for i in range(len(conditioning)):
            t1 = conditioning[i][0]
            pooled_output = conditioning[i][1].get("pooled_output", None)
            tw = torch.clamp(t1, -strength, strength)
            t_to = conditioning[i][1].copy()
            if pooled_output is not None:
                t_to["pooled_output"] = torch.clamp(pooled_output, -strength, strength)

            n = [tw, t_to]
            out.append(n)
        return (out, )

class ConditioningAddConDelta:
    """
    This node adds a conditioning delta with a specific strength to a conditioning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_base": ("CONDITIONING", ), "conditioning_delta": ("CONDITIONING", ),
                              "conditioning_delta_strength": ("FLOAT", {"default": 1.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "addDelta"

    CATEGORY = "conditioning"

    def addDelta(self, conditioning_base, conditioning_delta, conditioning_delta_strength):
        out = []

        if len(conditioning_delta) > 1:
            logging.warning("Warning: ExtendedConditioningAverage conditioning_delta contains more than 1 cond, only the first one will actually be applied to conditioning_base.")

        cond_delta = conditioning_delta[0][0]
        pooled_output_from = conditioning_delta[0][1].get("pooled_output", None)

        for i in range(len(conditioning_base)):
            t1 = conditioning_base[i][0]
            pooled_output_to = conditioning_base[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_delta[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = t1 + torch.mul(t0, conditioning_delta_strength)
            t_to = conditioning_base[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = pooled_output_to + torch.mul(pooled_output_from, conditioning_delta_strength)
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return (out, )    
    
class ConditioningAddConDeltaAutoScale:
    """
    This node adds a conditioning delta with a specific strength to a conditioning, scaling the delta to match the conditioning.
    It does this by determining the ratio of the means of the two vectors.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_base": ("CONDITIONING", ), "conditioning_delta": ("CONDITIONING", ),
                             "ratio_type": (["mean", "max", "median"], {"default": "median"}),
                              "conditioning_delta_strength": ("FLOAT", {"default": 1.0, "step": 0.01}),
                            "clamp": ("BOOLEAN", {"default": False, "tooltip": "If true, will clamp the ConDeltas's values."}),
                            "clamp_value": ("FLOAT", {"default": 3.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "addDelta"

    CATEGORY = "conditioning"

    def addDelta(self, conditioning_base, ratio_type, conditioning_delta, conditioning_delta_strength, clamp, clamp_value):
        out = []

        if len(conditioning_delta) > 1:
            logging.warning("Warning: ExtendedConditioningAverage conditioning_delta contains more than 1 cond, only the first one will actually be applied to conditioning_base.")

        cond_delta = conditioning_delta[0][0]
        pooled_output_from = conditioning_delta[0][1].get("pooled_output", None)

        if clamp:
            cond_delta = torch.clamp(cond_delta, -clamp_value, clamp_value)
            pooled_output_from = torch.clamp(pooled_output_from, -clamp_value, clamp_value)

        for i in range(len(conditioning_base)):
            t1 = conditioning_base[i][0]
            pooled_output_to = conditioning_base[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_delta[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            mean_ratio = tensor_ratio(t1, ratio_type) / tensor_ratio(t0, ratio_type)

            tw = t1 + torch.mul(t0, conditioning_delta_strength * mean_ratio)
            t_to = conditioning_base[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                mean_ratio_pooled = tensor_ratio(pooled_output_to, ratio_type) / tensor_ratio(pooled_output_from, ratio_type)
                t_to["pooled_output"] = pooled_output_to + torch.mul(pooled_output_from, conditioning_delta_strength * mean_ratio_pooled)
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return (out, )
    
class ConditioningAverageMultiple:
    """
    This node averages up to 10 conditioning vectors
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning0": ("CONDITIONING", ),
            "conditioning1": ("CONDITIONING", ),
            "conditioning2": ("CONDITIONING", ),
            "conditioning3": ("CONDITIONING", ),
            "conditioning4": ("CONDITIONING", ),
            "conditioning5": ("CONDITIONING", ),
            "conditioning6": ("CONDITIONING", ),
            "conditioning7": ("CONDITIONING", ),
            "conditioning8": ("CONDITIONING", ),
            "conditioning9": ("CONDITIONING", ),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "average"

    CATEGORY = "conditioning"

    def average(self, conditionings):
        out = []

        # Add them all up, counting the number that aren't null
        count = 0
        sum = None
        pooled_output = None
        for i in range(10):
            if len(conditionings[i]) > 0:
                count += 1
                if sum is None:
                    sum = conditionings[i][0][0]
                    pooled_output = conditionings[i][0][1].get("pooled_output", None)
                else:
                    sum += conditionings[i][0][0]
                    if pooled_output is not None:
                        pooled_output += conditionings[i][0][1].get("pooled_output", None)

        # Divide by the count
        if count > 0:
            sum /= count
            if pooled_output is not None:
                pooled_output /= count

        # Return the result
        for i in range(len(conditionings[0])):
            n = [sum, {"pooled_output": pooled_output}]
            out.append(n)

        return (out, )

class ExtendedConditioningAverage:
    """
    This node is the basic Conditioning Average node, but with the weight capped at [-9, 10] instead of [0, 1]. It's otherwise a drop-in replacement.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING", ), "conditioning_from": ("CONDITIONING", ),
                              "conditioning_to_strength": ("FLOAT", {"default": 1.0, "min": -9.0, "max": 10.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "addWeighted"

    CATEGORY = "conditioning"

    def addWeighted(self, conditioning_to, conditioning_from, conditioning_to_strength):
        out = []

        if len(conditioning_from) > 1:
            logging.warning("Warning: ExtendedConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return (out, )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ExtendedConditioningAverage": ExtendedConditioningAverage,
    "ConditioningSubtract": ConditioningSubtract,
    "ConditioningAddConDelta": ConditioningAddConDelta,
    "ConditioningAddConDeltaAutoScale": ConditioningAddConDeltaAutoScale,
    "SaveConditioningDelta": SaveConditioningDelta,
    "LoadConditioningDelta": LoadConditioningDelta,
    "ConditioningScale": ConditioningScale,
    "ApplyConDelta": ApplyConDelta,
    "ApplyConDeltaAutoScale": ApplyConDeltaAutoScale,
    "ThresholdConditioning": ThresholdConditioning,
    "ClampConDelta": ClampConDelta,
    "HardClampConDelta": HardClampConDelta,
    "MaskConDelta": MaskConDelta,
    "ConditioningAverageMultiple": ConditioningAverageMultiple,
    "ConditioningGetNoise": ConditioningGetNoise,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtendedConditioningAverage": "Conditioning Average Extended",
    "ConditioningSubtract": "Conditioning Subtract (Create ConDelta)",
    "ConditioningAddConDelta": "Conditioning Add ConDelta",
    "ConditioningAddConDeltaAutoScale": "Conditioning Add ConDelta AutoScale",
    "SaveConditioningDelta": "Save ConDelta",
    "LoadConditioningDelta": "Load ConDelta",
    "ConditioningScale": "ConDelta Scale (Multiple ConDelta by a Float)",
    "ApplyConDelta": "Apply ConDelta",
    "ApplyConDeltaAutoScale": "Apply ConDelta AutoScale",
    "ThresholdConditioning": "Threshold ConDelta",
    "ClampConDelta": "Smooth Clamp ConDelta between -1 and 1",
    "HardClampConDelta": "Hard Clamp ConDelta between -strength and strength",
    "MaskConDelta": "Mask ConDelta (Clamp ConDelta with another ConDelta or Conditioning)",
    "ConditioningAverageMultiple": "Average Multiple Conditionings or ConDeltas",
    "ConditioningGetNoise": "Get Noise from a Conditioning or ConDelta",
}

print("\033[94mConDelta Nodes: \033[92mLoaded\033[0m")
