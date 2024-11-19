import torch
import os
from dit import DiT_models
from vae import VAE_models
from torchvision.io import write_video
from utils import sigmoid_beta_schedule, one_hot_actions
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
from tqdm import tqdm


class WorldModel:
    """
    World Model class for generating videos using the OASIS model.

    This class encapsulates the functionality to load pre-trained DiT and VAE models,
    preprocess input prompts and actions, perform diffusion steps, and generate videos
    based on a sequence of actions.
    """

    def __init__(
        self,
        oasis_ckpt,
        vae_ckpt,
        num_frames=32,
        n_prompt_frames=1,
        ddim_steps=10,
        fps=20,
        scaling_factor=0.07843137255,
        max_noise_level=1000,
        noise_abs_max=20,
        stabilization_level=15,
        seed=0,
    ):
        """
        Initialize the WorldModel with specified parameters and load the models.

        Args:
            oasis_ckpt (str): Path to the Oasis DiT checkpoint file.
            vae_ckpt (str): Path to the ViT-VAE checkpoint file.
            num_frames (int, optional): Total number of frames to generate. Defaults to 32.
            n_prompt_frames (int, optional): Number of initial prompt frames. Defaults to 1.
            ddim_steps (int, optional): Number of DDIM steps for inference. Defaults to 10.
            fps (int, optional): Frames per second for the output video. Defaults to 20.
            scaling_factor (float, optional): Scaling factor for VAE encoding. Defaults to 0.07843137255.
            max_noise_level (int, optional): Maximum noise level for diffusion. Defaults to 1000.
            noise_abs_max (int, optional): Maximum absolute noise value. Defaults to 20.
            stabilization_level (int, optional): Stabilization level for noise scheduling. Defaults to 15.
        """
        assert torch.cuda.is_available(), "CUDA is not available"
        self.device = "cuda:0"
        self.num_frames = num_frames
        self.n_prompt_frames = n_prompt_frames
        self.ddim_steps = ddim_steps
        self.fps = fps
        self.scaling_factor = scaling_factor
        self.max_noise_level = max_noise_level
        self.noise_abs_max = noise_abs_max
        self.stabilization_level = stabilization_level

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Load DiT model (Oasis)
        self.model = DiT_models["DiT-S/2"]()
        print(f"Loading Oasis-500M from {os.path.abspath(oasis_ckpt)}...")
        if oasis_ckpt.endswith(".pt"):
            ckpt = torch.load(oasis_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt, strict=False)
        elif oasis_ckpt.endswith(".safetensors"):
            load_model(self.model, oasis_ckpt)
        self.model = self.model.to(self.device).eval()

        # Load VAE model (ViT-VAE)
        self.vae = VAE_models["vit-l-20-shallow-encoder"]()
        print(f"Loading ViT-VAE-L/20 from {os.path.abspath(vae_ckpt)}...")
        if vae_ckpt.endswith(".pt"):
            vae_ckpt_data = torch.load(vae_ckpt, map_location=self.device)
            self.vae.load_state_dict(vae_ckpt_data)
        elif vae_ckpt.endswith(".safetensors"):
            load_model(self.vae, vae_ckpt)
        self.vae = self.vae.to(self.device).eval()

        # Sampling parameters
        self.noise_range = torch.linspace(-1, self.max_noise_level - 1, ddim_steps + 1).to(self.device)

    def preprocess_input(self, prompt_tensor, actions_dict_list):
        """
        Preprocesses input prompt and action sequences for the world model.

        This involves resizing and normalizing the prompt tensor and converting action dictionaries
        into one-hot encoded tensors.

        Args:
            prompt_tensor (torch.Tensor): Prompt tensor of shape (T, C, H, W).
            actions_dict_list (list of dict): List of action dictionaries for each frame.

        Returns:
            torch.Tensor: Preprocessed prompt tensor. Resized and normalized to (1, T, C, H, W).
            torch.Tensor: One-hot encoded actions tensor of shape (1, T, D).
        """
        # Resize and normalize prompt tensor
        prompt = torch.nn.functional.interpolate(prompt_tensor, size=(360, 640))
        prompt = rearrange(prompt, "t c h w -> 1 t c h w")
        prompt = prompt.float() / 255.0
        x = prompt.to(self.device)

        # Transform action dictionaries to one_hot actions
        actions = one_hot_actions(actions_dict_list).to(self.device)  # Shape: (T, D)
        actions = actions.unsqueeze(0)  # Shape: (1, T, D)

        return x, actions

    def vae_encoding(self, x, actions):
        """
        Encode the prompt using the VAE.

        Args:
            x (torch.Tensor): Preprocessed prompt tensor of shape (B, T, C, H, W).
            actions (torch.Tensor): One-hot encoded actions tensor of shape (1, T, D).

        Returns:
            torch.Tensor: Encoded prompt tensor of shape (B, n_prompt_frames, C, H, W).
            torch.Tensor: One-hot encoded actions tensor of shape (1, T, D).
        """
        # Ensure the input tensor has 5 dimensions: (B, T, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # Rearrange to merge batch and temporal dimensions for VAE encoding
            x = rearrange(x, "b t c h w -> (b t) c h w")
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")

        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                # Encode the input using the VAE
                x_encoded = self.vae.encode(x * 2 - 1).mean * self.scaling_factor

        # Rearrange back to (B, T, C', H', W') where C' is the encoded channels
        x_encoded = rearrange(
            x_encoded,
            "(b t) (h w) c -> b t c h w",
            b=B,
            t=T,
            h=H // self.vae.patch_size,
            w=W // self.vae.patch_size,
        )

        # Select only the prompt frames
        x_encoded = x_encoded[:, : self.n_prompt_frames]

        return x_encoded, actions

    def diffusion(self, x_encoded, actions):
        """
        Perform the diffusion process to generate frames.

        Args:
            x_encoded (torch.Tensor): Encoded prompt tensor of shape (B, T, C, H, W).
            actions (torch.Tensor): One-hot encoded actions tensor of shape (1, T, D).

        Returns:
            x_encoded (torch.Tensor): Encoded prompt tensor with generated frames of shape (B, num_frames, C, H, W).
        """
        B = x_encoded.shape[0]
        frames_list = [x_encoded]

        # Initialize alphas and betas for noise scheduling
        betas = sigmoid_beta_schedule(self.max_noise_level).float().to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)  # Shape: (T,)
        alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

        for i in tqdm(range(self.n_prompt_frames, self.num_frames), desc="Generating frames"):
            # Add random noise to the current frame
            chunk = torch.randn((B, 1, *x_encoded.shape[-3:]), device=self.device)
            chunk = torch.clamp(chunk, -self.noise_abs_max, +self.noise_abs_max)
            x_encoded = torch.cat([x_encoded, chunk], dim=1)  # Shape: (B, T+1, C, H, W)
            start_frame = max(0, i + 1 - self.model.max_frames)

            for noise_idx in reversed(range(1, self.ddim_steps + 1)):
                # Set up noise values based on noise range and stabilization
                t_ctx = torch.full(
                    (B, i),
                    self.stabilization_level - 1,
                    dtype=torch.long,
                    device=self.device,
                )
                t = torch.full((B, 1), self.noise_range[noise_idx], dtype=torch.long, device=self.device)
                t_next = torch.full((B, 1), self.noise_range[noise_idx - 1], dtype=torch.long, device=self.device)
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1)
                t_next = torch.cat([t_ctx, t_next], dim=1)

                # Sliding window
                x_curr = x_encoded.clone()
                x_curr = x_curr[:, start_frame:]
                t = t[:, start_frame:]
                t_next = t_next[:, start_frame:]

                # Get model predictions
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.half):
                        v = self.model(x_curr, t, actions[:, start_frame : i + 1])

                # Compute the start and noise values
                x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

                # Compute the next noise level
                alpha_next = alphas_cumprod[t_next]
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])

                # Update the frame
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x_encoded[:, -1:] = x_pred[:, -1:]

        return x_encoded

    def vae_decoding(self, x_encoded):
        """
        Decode the encoded frames using the VAE.

        Args:
            x_encoded (torch.Tensor): Encoded prompt tensor with generated frames of shape (B, num_frames, C, H, W).

        Returns:
            torch.Tensor: Decoded frames as RGB numpy arrays.
        """
        x = rearrange(x_encoded, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            x = (self.vae.decode(x / self.scaling_factor) + 1) / 2
        x = rearrange(x, "(b t) c h w -> b t h w c", t=self.num_frames)

        return x

    def run(
        self,
        prompt_tensor,
        actions_dict_list,
        output_path="output.mp4",
    ):
        """
        Run the full pipeline to generate a video from the prompt and actions.

        Args:
            prompt_tensor (torch.Tensor): Prompt tensor of shape (1, C, H, W).
            actions_dict_list (list of dict): List of action dictionaries for each frame.
            output_path (str, optional): Path where generated video will be saved. Defaults to "output.mp4".

        Returns:
            list: List of generated frames as RGB numpy arrays.
        """
        # Preprocess input prompt and actions
        x, actions = self.preprocess_input(prompt_tensor, actions_dict_list)

        # VAE Encoding
        x_encoded, actions = self.vae_encoding(x, actions)

        # Diffusion
        frames_list = self.diffusion(x_encoded, actions)

        # VAE Decoding
        x = self.vae_decoding(frames_list)

        # Save video
        x = torch.clamp(x, 0, 1)
        x = (x * 255).byte()
        if output_path:
            write_video(output_path, x[0].cpu(), fps=self.fps)
            print(f"Generation saved to {output_path}.")

        # Convert frames to numpy arrays
        frames_np = x[0].cpu().numpy()

        return frames_np
