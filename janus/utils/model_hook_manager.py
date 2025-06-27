# ModelHookManager: utility for PyTorch model instrumentation
import torch

class ModelHookManager:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.activations = {}

    def add_hook(self, layer_name):
        layer = dict([*self.model.named_modules()])[layer_name]
        handle = layer.register_forward_hook(self._hook_fn(layer_name))
        self.handles.append(handle)

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = {}
