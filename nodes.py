import torch
import safetensors.torch
import nodes
import logging
import folder_paths
import os
import re
from comfy.comfy_types import IO, InputTypeDict


condelta_dir = os.path.join(folder_paths.models_dir, "ConDelta")

# A list of prompts for various things to use to generate a style conditioning delta
style_prompts = {
    "misc": [
        "A tabby cat with a blue collar and a red bowtie, wearing glasses, sitting on a stack of books",
        "A sexy anime succubus with a red dress and black wings, holding a pitchfork, with a mischievous smile",
        "A scene of a futuristic city with flying cars and neon lights, with a giant robot in the background",
        "A tranquil mountain landscape with a clear blue lake and a small cabin, with a sunset in the background",
        "An old man sitting at a rustic table.",
        "A surreal scene with a giant clock melting over a tree.",
        "A cliff overlooking a stormy sea, with a lighthouse in the distance.",
    ],
    "anime": [
        "A cute girl with long pink hair and a blue dress, holding a bouquet of flowers",
        "A sexy woman with red fox ears and nine tails wearing a revealing kimono and smiling seductively, standing in front of a Japanese shrine",
        "A teen age boy with spiky blue hair and a black jacket, riding a motorcycle",
        "An old railway track running through the japanese countryside, with a forest and mountains in the background",
        "A girl with blue hair in an elaborate, revealing sci-fi plugsuit, standing in a mecha hanger",
        "A cyber-ninja in a tight black suit with neon highlighs, sneaking down a hallway in a futuristic factory",
        "A cute young girl with her pet monster. The monster is glowing blue asnd has large cute eyes",
        "A samurai engaged in battle at night. There are burning buildings in the background",
    ]
}

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

def tenser_get_gaussian_noise(tensor, strength, seed=None):
    """
    Returns a tensor of gaussian noise, multiplied by the strength.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    return torch.randn_like(tensor) * strength

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

def conditioning_add(conditioning_a, conditioning_b):
    """
    Adds two conditionings together (A + B), producing a conditioning delta.
    """

    out = []

    cond_b = conditioning_b[0][0]
    pooled_output_b = conditioning_b[0][1].get("pooled_output", None)
    conditioning_llama3_b = conditioning_b[0][1].get("conditioning_llama3", None)

    for i in range(len(conditioning_a)):
        # Print the keys of conditioning_a[i][1]
        if conditioning_a[i][1] is None:
            logging.warning(f"conditioning_a[{i}][1] is None, skipping.")
        if conditioning_a[i][1] is not None:
            logging.debug(f"conditioning_a[{i}][1] keys: {list(conditioning_a[i][1].keys())}")
        
        t1 = conditioning_a[i][0]
        pooled_output_a = conditioning_a[i][1].get("pooled_output", pooled_output_b)
        conditioning_llama3_a = conditioning_a[i][1].get("conditioning_llama3", None)
        t0 = cond_b[:,:t1.shape[1]]
        if t0.shape[1] < t1.shape[1]:
            t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)
            
        # Pad conditioning_llama3_a or _b if [1, 32, a, 4096] is not [1, 32, b, 4096]
        if conditioning_llama3_b is not None and conditioning_llama3_a is not None:
            if conditioning_llama3_b.shape[2] < conditioning_llama3_a.shape[2]:
                conditioning_llama3_b = torch.cat([conditioning_llama3_b] + [torch.zeros((1, 32, (conditioning_llama3_a.shape[2] - conditioning_llama3_b.shape[2]), 4096))], dim=2)
            elif conditioning_llama3_b.shape[2] > conditioning_llama3_a.shape[2]:
                conditioning_llama3_a = torch.cat([conditioning_llama3_a] + [torch.zeros((1, 32, (conditioning_llama3_b.shape[2] - conditioning_llama3_a.shape[2]), 4096))], dim=2)

        tw = t1 + torch.mul(t0, 1)
        t_to = conditioning_a[i][1].copy()
        if pooled_output_b is not None and pooled_output_a is not None:
            t_to["pooled_output"] = pooled_output_a + torch.mul(pooled_output_b, 1)
        elif pooled_output_b is not None:
            t_to["pooled_output"] = pooled_output_b
            
        if conditioning_llama3_b is not None and conditioning_llama3_a is not None:                
            t_to["conditioning_llama3"] = conditioning_llama3_a + torch.mul(conditioning_llama3_b, 1)
        elif conditioning_llama3_b is not None:
            t_to["conditioning_llama3"] = conditioning_llama3_b

        n = [tw, t_to]
        out.append(n)
    return out

def conditioning_scale(conditioning, scalar):
    """
    Multiplies the conditioning by a scalar.
    """

    out = []

    for i in range(len(conditioning)):
        t1 = conditioning[i][0]
        pooled_output = conditioning[i][1].get("pooled_output", None)
        conditioning_llama3 = conditioning[i][1].get("conditioning_llama3", None)
        tw = torch.mul(t1, scalar)
        #print(f"Mean before multiply: {tensor_mean(t1)}, Mean after multiply: {tensor_mean(tw)}, Scalar: {scalar}")
        t_to = conditioning[i][1].copy()
        
        #print("scale: conditioning[i][1] keys: ", t_to.keys())
        
        if pooled_output is not None:
            t_to["pooled_output"] = torch.mul(pooled_output, scalar)
            print(f"Pooled Mean before multiply: {tensor_mean(pooled_output)}, Mean after multiply: {tensor_mean(t_to['pooled_output'])}, Scalar: {scalar}")
        #else:
            #print("scale: No pooled_output found in conditioning, skipping.")
        if conditioning_llama3 is not None:
            mean_before_multiply = tensor_mean(conditioning_llama3)
            t_to["conditioning_llama3"] = torch.mul(conditioning_llama3, scalar)
            mean_after_multiply = tensor_mean(t_to["conditioning_llama3"])
            print(f"Llama3 Mean before multiply: {mean_before_multiply}, Mean after multiply: {mean_after_multiply}, Scalar: {scalar}")
        #else:
            #print("scale: No conditioning_llama3 found in conditioning, skipping.")

        n = [tw, t_to]
        out.append(n)
    return out

def conditioning_subtract(conditioning_a, conditioning_b):
    """
    Subtracts one conditioning from another (A-B), producing a conditioning delta by
    multiplying B by -1 and adding it to A with conditionining_scale and conditioning_add.
    """
    
    conditioning_b = conditioning_scale(conditioning_b, -1)
    return conditioning_add(conditioning_a, conditioning_b)
    
def get_conditioning_from_prompt(prompt, clip, **kwargs):
    """
    This function gets the conditioning from a prompt and a clip model.
    """
    
    # Get the conditioning from the prompt
    tokens = clip.tokenize(prompt, **kwargs)
    return clip.encode_from_tokens_scheduled(tokens)


class QuickConDelta:
    """
    This shortcut node takes a conditioning, clip, and prompt as input.  It encodes the prompt
    using the clip model, then encodes a "" prompt using the clip model.  It subtracts the "" conditioning
    from the prompt conditioning, producing a conditioning delta. Then it adds the conditioning delta
    to the user supplied conditioning, producing a new conditioning.
    """
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "prompt": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "strength": ("FLOAT", {"default": 0.6, "step": 0.01, "min": -100.0, "max": 100.0}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "getConditioning"

    CATEGORY = "conditioning"

    def getConditioning(self, conditioning, clip, prompt, strength):
        # Get the conditioning from the prompt        
        conditioning_from_prompt = get_conditioning_from_prompt(prompt, clip)
        
        blank_conditioning = get_conditioning_from_prompt("", clip)
        # Subtract the blank conditioning from the conditioning from the prompt
        conditioning_delta = conditioning_subtract(conditioning_from_prompt, blank_conditioning)
        
        # Scale the conditioning delta by the negative prompt strength
        conditioning_delta = conditioning_scale(conditioning_delta, strength)
        
        # Subtract the conditioning from the conditioning from the prompt
        new_conditioning = conditioning_add(conditioning, conditioning_delta)
        # Return the conditioning delta
        return (new_conditioning, )

class CFGlessNegativePrompt:
    """
    This shortcut node takes a conditioning, clip, and prompt as input.  It encodes the prompt
    using the clip model, then encodes a "" prompt using the clip model.  It subtracts the "" conditioning
    from the prompt conditioning, producing a conditioning delta. Then it substracts the conditioning delta
    from the user supplied conditioning, producing a new conditioning.
    """
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "negative_prompt": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "negative_prompt_strength": ("FLOAT", {"default": 0.6, "step": 0.01, "min": -100.0, "max": 100.0}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "getConditioning"

    CATEGORY = "conditioning"

    def getConditioning(self, conditioning, clip, negative_prompt, negative_prompt_strength):
        # Get the conditioning from the prompt
        conditioning_from_prompt = get_conditioning_from_prompt(negative_prompt, clip)
        blank_conditioning = get_conditioning_from_prompt("", clip)
        # Subtract the blank conditioning from the conditioning from the prompt
        conditioning_delta = conditioning_subtract(conditioning_from_prompt, blank_conditioning)
        
        # Scale the conditioning delta by the negative prompt strength
        conditioning_delta = conditioning_scale(conditioning_delta, negative_prompt_strength)
        
        # Subtract the conditioning from the conditioning from the prompt
        conditioning_with_negative = conditioning_subtract(conditioning, conditioning_delta)
        # Return the conditioning delta
        return (conditioning_with_negative, )

class GetConDeltaFromPrompt:
    """
    This node gets a conditioning delta from a prompt and a clip model.
    """

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "prompt": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "prompt_type": ([*style_prompts.keys(), 'custom', 'none'], {"default": "misc"}),
                "custom_prompts": (IO.STRING, {"multiline": True, "tooltip": "Custom prompts to use for conditioning delta, one per line."}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "getConDelta"

    CATEGORY = "conditioning"

    def getConDelta(self, prompt, clip, prompt_type, custom_prompts):
        # Get the style prompts
        _style_prompts = []
        
        if prompt_type == "custom":
            _style_prompts = custom_prompts.split("\n")
        elif prompt_type == "none":
            _style_prompts = [""]
        else:
            _style_prompts = style_prompts[prompt_type]

        
        conditioning_delta = None
        # Iterate through the style prompts
        for style_prompt in _style_prompts:
            # Append the style prompt to the prompt
            prompt_with_style = f"{prompt}. {style_prompt}"
            # Get the conditioning from the prompt with the style prompt appended
            conditioning_with_style = get_conditioning_from_prompt(prompt_with_style, clip)
            # Get the conditioning from the prompt without the style appended
            conditioning_without_style = get_conditioning_from_prompt(prompt, clip)
            # Subtract the conditioning without the style from the conditioning with the style
            conditioning = conditioning_subtract(conditioning_with_style, conditioning_without_style)
            
            # If the conditioning delta is None, set it to the conditioning
            if conditioning_delta is None:
                conditioning_delta = conditioning
            else:
                # Add the conditioning to the conditioning delta
                conditioning_delta = conditioning_add(conditioning_delta, conditioning)
        
        # Scale the conditioning delta by the number of style prompts
        conditioning_delta = conditioning_scale(conditioning_delta, 1.0 / len(style_prompts))
        # Return the conditioning delta
        return (conditioning_delta, )

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
    
class ConditioningGetRandom:
    """
    This generates random gaussian noise based on the input conditioning and strength
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ), 
                             "strength": ("FLOAT", {"default": 1.0, "step": 0.01}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "getRandom"

    CATEGORY = "conditioning"

    def getRandom(self, conditioning, strength, seed):
        out = []

        for i in range(len(conditioning)):
            t1 = conditioning[i][0]
            pooled_output = conditioning[i][1].get("pooled_output", None)
            tw = tenser_get_gaussian_noise(t1, strength, seed)
            if pooled_output is not None:
                pooled_output = tenser_get_gaussian_noise(pooled_output, strength, seed)

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
                            "strength": ("FLOAT", {"default": 1.0, "step": 0.01, "min": -100.0, "max": 100.0, })}}
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
                            "strength": ("FLOAT", {"default": 1.0, "step": 0.01, "min": -100.0, "max": 100.0, }),
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
            conditioning_llama3 = conditioning[i][1].get("conditioning_llama3", None)
            tw = torch.mul(t1, scalar)
            t_to = conditioning[i][1].copy()
            if pooled_output is not None:
                t_to["pooled_output"] = torch.mul(pooled_output, scalar)
            if conditioning_llama3 is not None:
                t_to["conditioning_llama3"] = torch.mul(conditioning_llama3, scalar)

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
        '''
        out = []
        
        cond_b = conditioning_b[0][0]
        pooled_output_b = conditioning_b[0][1].get("pooled_output", None)
        conditioning_llama3_b = conditioning_b[0][1].get("conditioning_llama3", None)

        for i in range(len(conditioning_a)):
            # Print all keys in conditioning_a and conditioning_b
            if conditioning_a[i][1] is not None:
                print(f"conditioning_a[{i}] keys: ", conditioning_a[i][1].keys())
            if conditioning_b[i][1] is not None:
                print(f"conditioning_b[{i}] keys: ", conditioning_b[i][1].keys())

            t1 = conditioning_a[i][0]
            pooled_output_a = conditioning_a[i][1].get("pooled_output", pooled_output_b)
            conditioning_llama3_a = conditioning_a[i][1].get("conditioning_llama3", None)
            t0 = cond_b[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = t1 + torch.mul(t0, -1)
            t_to = conditioning_a[i][1].copy()
            if pooled_output_b is not None and pooled_output_a is not None:
                t_to["pooled_output"] = pooled_output_a + torch.mul(pooled_output_b, -1)
            elif pooled_output_b is not None:
                t_to["pooled_output"] = pooled_output_b
                
            if conditioning_llama3_b is not None and conditioning_llama3_a is not None:                
                t_to["conditioning_llama3"] = conditioning_llama3_a + torch.mul(conditioning_llama3_b, -1)
            elif conditioning_llama3_b is not None:
                t_to["conditioning_llama3"] = conditioning_llama3_b

            n = [tw, t_to]
            out.append(n)
        return (out, )
        '''
        out = conditioning_subtract(conditioning_a, conditioning_b)
        # Print the keys of the first conditioning in the output
        if out[0][1] is not None:
            print(f"Output conditioning keys: {list(out[0][1].keys())}")
        else:
            print("Output conditioning has no keys.")
        return (out, )

class ThresholdConditioning:
    """
    This node applies a threshold to the conditioning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ), "threshold": ("FLOAT", {"default": 1.0, "step": 0.01, "min": -100.0, "max": 100.0, })
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

class PromptTravel:
    """
    Travels between two prompts by a specified amount (can exceed 0-1 range).

    Inputs: 
      prompt A (string), 
      prompt B (string), 
      travel_amount (float), 
      clip (for encoding prompts)
    Returns: new conditioning
    """
    
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "prompt_a": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "prompt_b": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "travel_amount": ("FLOAT", {"default": 0.5, "step": 0.01, "min": -100.0, "max": 100.0}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "travel"

    CATEGORY = "conditioning"

    def travel(self, prompt_a, prompt_b, travel_amount, clip):
        # Get the conditioning from the prompts        
        conditioning_a = get_conditioning_from_prompt(prompt_a, clip)
        conditioning_b = get_conditioning_from_prompt(prompt_b, clip)
        
        # Subtract the conditioning from prompt A from the conditioning from prompt B
        conditioning_delta = conditioning_subtract(conditioning_b, conditioning_a)
        
        # Scale the conditioning delta by the travel amount
        conditioning_delta = conditioning_scale(conditioning_delta, travel_amount)
        
        # Add the conditioning delta to the conditioning from prompt A
        new_conditioning = conditioning_add(conditioning_a, conditioning_delta)
        # Return the new conditioning
        return (new_conditioning, )
    
class MultiDimensionalPromptTravel:
    """
    Inputs:
       Prompt (string, see special formatting below)
       Clip (for encoding prompts)
       
    Returns: new conditioning
    
    Prompt format:
       The prompt is written as normal, except that it contains sections enclosed in angle brackets <>.  Each angle bracket section is formatted as follows: <base_subprompt:destination1:weight1:destination2:weight2:...>.

       For instance:
            A beautiful <cat:dog:0.5:wolf:-0.3> sitting on a <red:blue:0.25:green:0.1> mat.
            
    First, it determines the base prompt by replacing each angle bracket section with the base subprompt.  In the above example, the base prompt would be "A beautiful cat sitting on a red mat."  It encodes this prompt to get the base conditioning.
    
    Then, for each angle bracket section, it processes each destination:weight pair. For each pair, it:
    1. Creates a prompt with the destination subprompt replacing the base subprompt
    2. Encodes this destination prompt to get its conditioning
    3. Subtracts the base conditioning from the destination conditioning to get a conditioning delta
    4. Scales the conditioning delta by the specified weight
    5. Adds the scaled conditioning delta to the base conditioning
    
    For the above example, it would generate the following delta prompts:
        1. A beautiful dog sitting on a red mat. (weight: 0.5)
        2. A beautiful wolf sitting on a red mat. (weight: -0.3)
        3. A beautiful cat sitting on a blue mat. (weight: 0.25)  
        4. A beautiful cat sitting on a green mat. (weight: 0.1)

    The result is a conditioning that has traveled towards each of the destination subprompts by their specified weights.
    """
    
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "prompt": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded.  Use special formatting for multi-dimensional travel."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "travel"

    CATEGORY = "conditioning"

    def travel(self, prompt, clip):
        # Find all bracket patterns and their positions
        bracket_pattern = r"<([^>]+)>"
        
        # Create base prompt by replacing each bracket with its first element
        base_prompt = re.sub(bracket_pattern, lambda m: m.group(1).split(":")[0], prompt)
        
        print("Original prompt: ", prompt)
        print("Base prompt: ", base_prompt)

        # Get the conditioning from the base prompt        
        base_conditioning = get_conditioning_from_prompt(base_prompt, clip)
        original_base_conditioning = base_conditioning
        
        matches = list(re.finditer(bracket_pattern, prompt))
        
        for match in matches:
            full_bracket = match.group(0)
            bracket_content = match.group(1)
            
            print(f"Processing bracket: {full_bracket}")
            
            parts = bracket_content.split(":")
            if len(parts) < 3:
                print(f"Invalid delta prompt format (need at least base:destination:weight): {bracket_content}")
                continue
            if len(parts) % 2 == 0:
                print(f"Invalid delta prompt format (need base:destination:weight pairs): {bracket_content}")
                continue
            
            base_subprompt = parts[0]
            
            for i in range(1, len(parts), 2):
                if i + 1 >= len(parts):
                    print(f"Missing weight for destination '{parts[i]}' in: {bracket_content}")
                    break
                
                destination_subprompt = parts[i]
                try:
                    travel_distance = float(parts[i + 1])
                except ValueError:
                    print(f"Invalid travel distance '{parts[i + 1]}' for destination '{destination_subprompt}': {bracket_content}")
                    continue
                
                # Let's build the destination prompt from the original prompt by replacing just the current bracket
                # and all other brackets with their base values.
                
                # To do this, we can iterate through all matches and build the string segment by segment.
                
                current_prompt_for_dest = ""
                last_index = 0
                for m in matches:
                    # Add the text between the last match and this one
                    current_prompt_for_dest += prompt[last_index:m.start()]
                    
                    if m.group(0) == full_bracket:
                        # This is the bracket we're processing, so use the destination subprompt
                        current_prompt_for_dest += destination_subprompt
                    else:
                        # This is a different bracket, so use its base subprompt
                        current_prompt_for_dest += m.group(1).split(":")[0]
                    
                    last_index = m.end()
                
                # Add the remaining part of the prompt after the last match
                current_prompt_for_dest += prompt[last_index:]

                print(f"  Processing: {base_subprompt} -> {destination_subprompt} (weight: {travel_distance})")
                print(f"  Full destination prompt: {current_prompt_for_dest}")
                
                conditioning_destination = get_conditioning_from_prompt(current_prompt_for_dest, clip)
                
                conditioning_delta = conditioning_subtract(conditioning_destination, original_base_conditioning)
                
                conditioning_delta = conditioning_scale(conditioning_delta, travel_distance)
                
                base_conditioning = conditioning_add(base_conditioning, conditioning_delta)
            
        return (base_conditioning, )
            
class ConditioningAddConDelta:
    """
    This node adds a conditioning delta with a specific strength to a conditioning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_base": ("CONDITIONING", ), "conditioning_delta": ("CONDITIONING", ),
                              "conditioning_delta_strength": ("FLOAT", {"default": 1.0, "step": 0.01, "min": -100.0, "max": 100.0, })
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "addDelta"

    CATEGORY = "conditioning"

    def addDelta(self, conditioning_base, conditioning_delta, conditioning_delta_strength):
        '''
        out = []

        if len(conditioning_delta) > 1:
            logging.warning("Warning: ExtendedConditioningAverage conditioning_delta contains more than 1 cond, only the first one will actually be applied to conditioning_base.")

        cond_delta = conditioning_delta[0][0]
        pooled_output_from = conditioning_delta[0][1].get("pooled_output", None)
        conditioning_llama3_from = conditioning_delta[0][1].get("conditioning_llama3", None)

        for i in range(len(conditioning_base)):
            t1 = conditioning_base[i][0]
            pooled_output_to = conditioning_base[i][1].get("pooled_output", pooled_output_from)
            conditioning_llama3_to = conditioning_base[i][1].get("conditioning_llama3", conditioning_llama3_from)
            t0 = cond_delta[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = t1 + torch.mul(t0, conditioning_delta_strength)
            t_to = conditioning_base[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = pooled_output_to + torch.mul(pooled_output_from, conditioning_delta_strength)
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from
                
            if conditioning_llama3_from is not None and conditioning_llama3_to is not None:
                t_to["conditioning_llama3"] = conditioning_llama3_to + torch.mul(conditioning_llama3_from, conditioning_delta_strength)
            elif conditioning_llama3_from is not None:
                t_to["conditioning_llama3"] = conditioning_llama3_from

            n = [tw, t_to]
            out.append(n)
        return (out, )'''
        conditioning_delta = conditioning_scale(conditioning_delta, conditioning_delta_strength)
        out = conditioning_add(conditioning_base, conditioning_delta)

        # Print the keys of the first conditioning in the output
        if out[0][1] is not None:
            print(f"Output conditioning keys: {list(out[0][1].keys())}")
        else:
            print("Output conditioning has no keys.")
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
                              "conditioning_delta_strength": ("FLOAT", {"default": 1.0, "step": 0.01, "min": -100.0, "max": 100.0, }),
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
            }, "optional": {
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

    def average(self, conditioning0, conditioning1 = None, conditioning2 = None, conditioning3 = None, conditioning4 = None, conditioning5 = None, conditioning6 = None, conditioning7 = None, conditioning8 = None, conditioning9 = None):
        conditionings = [conditioning0, conditioning1, conditioning2, conditioning3, conditioning4, conditioning5, conditioning6, conditioning7, conditioning8, conditioning9]

        # Add them all up, counting the number that aren't null
        count = 0
        sum = None
        for i in range(10):
            if conditionings[i] is not None:
                count += 1
                if sum is None:
                    sum = conditionings[i]
                else:
                    # Call ConditioningAddConDelta with a strength of 1.0
                    sum = self.addDelta(sum, conditionings[i], 1.0)


        # Divide by the count
        if count > 0:
            sum = self.multiply(sum, 1.0 / count)

        # Return the result
        return (sum, )
    
    
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

class ExtendedConditioningAverage:
    """
    This node is the basic Conditioning Average node, but with the weight capped at [-9, 10] instead of [0, 1]. It's otherwise a drop-in replacement.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING", ), "conditioning_from": ("CONDITIONING", ),
                              "conditioning_to_strength": ("FLOAT", {"default": 1.0, "step": 0.01, "min": -100.0, "max": 100.0, })
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
    "GetConDeltaFromPrompt": GetConDeltaFromPrompt,
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
    "ConditioningGetRandom": ConditioningGetRandom,
    "CFGlessNegativePrompt": CFGlessNegativePrompt,
    "QuickConDelta": QuickConDelta,
    "PromptTravel": PromptTravel,
    "MultiDimensionalPromptTravel": MultiDimensionalPromptTravel,
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
    "ConditioningGetRandom": "Get random Gaussian noise from a Conditioning or ConDelta",
    "GetConDeltaFromPrompt": "Get ConDelta from Prompt",
    "CFGlessNegativePrompt": "CFG-less Negative Prompt",
    "QuickConDelta": "Quick ConDelta",
    "PromptTravel": "Prompt Travel",
    "MultiDimensionalPromptTravel": "Multi-Dimensional Prompt Travel",
}

print("\033[94mConDelta Nodes: \033[92mLoaded\033[0m")
