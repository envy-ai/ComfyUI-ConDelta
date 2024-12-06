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
class ApplyConditioningDelta:
    """
    This node applies a conditioning delta to a conditioning.
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

class ConditioningAddConDelta:
    """
    This node adds a conditioning delta with a specific strength to a conditioning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_base": ("CONDITIONING", ), "conditioning_delta": ("CONDITIONING", ),
                              "conditioning_delta_strength": ("FLOAT", {"default": 1.0, "min": -9.0, "max": 10.0, "step": 0.01})
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
    "SaveConditioningDelta": SaveConditioningDelta,
    "LoadConditioningDelta": LoadConditioningDelta,
    "ConditioningScale": ConditioningScale,
    "ApplyConditioningDelta": ApplyConditioningDelta,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtendedConditioningAverage": "Conditioning Average Extended",
    "ConditioningSubtract": "Conditioning Subtract",
    "ConditioningAddConDelta": "Conditioning Add ConDelta",
    "SaveConditioningDelta": "Save ConDelta",
    "LoadConditioningDelta": "Load ConDelta",
    "ConditioningScale": "Conditioning Scale (Multiple ConDelta by a Float)",
    "ApplyConditioningDelta": "Apply ConDelta",
}

print("\033[94mConDelta Nodes: \033[92mLoaded\033[0m")
