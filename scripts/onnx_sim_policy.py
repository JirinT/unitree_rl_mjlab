"""Feedforward actor ONNX playback for any compatible registered robot task."""
import numpy as np
import torch


class OnnxSimPolicy:
    def __init__(self, path, action_dim=None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ins, outs = self.session.get_inputs(), self.session.get_outputs()
        if len(ins) != 1 or len(outs) != 1 or len(ins[0].shape) != 2 or len(outs[0].shape) != 2:
            raise ValueError("Expected one rank-2 actor input and action output; recurrent policies require an adapter")
        if isinstance(ins[0].shape[0], int) and ins[0].shape[0] != 1:
            raise ValueError("Expected an ONNX input batch of 1 or a dynamic batch")
        if isinstance(outs[0].shape[0], int) and outs[0].shape[0] != 1:
            raise ValueError("Expected an ONNX output batch of 1 or a dynamic batch")
        self.obs_dim = ins[0].shape[1]
        output_dim = outs[0].shape[1]
        if action_dim is not None and isinstance(output_dim, int) and output_dim != action_dim:
            raise ValueError(f"ONNX has {output_dim} actions; selected task requires {action_dim}")
        self.action_dim = action_dim if action_dim is not None else output_dim
        self.input, self.output = ins[0].name, outs[0].name

    @torch.inference_mode()
    def __call__(self, obs):
        tensor = obs["actor"]
        x = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected finite, nonempty rank-2 actor observations")
        if isinstance(self.obs_dim, int) and x.shape[1] != self.obs_dim:
            raise ValueError(f"ONNX requires {self.obs_dim} observations; selected task supplies {x.shape[1]}")
        # Exported models include normalization. Evaluate fixed-batch models per env.
        action = np.concatenate([self.session.run([self.output], {self.input: row[None]})[0] for row in x])
        if action.ndim != 2 or len(action) != len(x) or not np.isfinite(action).all():
            raise ValueError("Invalid ONNX actions")
        if isinstance(self.action_dim, int) and action.shape[1] != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} actions, got {action.shape[1]}")
        return torch.from_numpy(action).to(device=tensor.device)

    def reset(self, dones=None):
        pass
