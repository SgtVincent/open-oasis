import torch
import os
from world_model import WorldModel
from PIL import Image
from torchvision.io import read_image, read_video, write_png, write_video
from utils import load_prompt
from einops import rearrange

# Load example image/ video
prompt_path = "sample_data/Player729-f153ac423f61-20210806-224813.chunk_000.mp4"
video_offset = 0
n_prompt_frames = 1
# prompt = load_prompt(
#     prompt_path, video_offset=video_offset, n_prompt_frames=n_prompt_frames
# )
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
VIDEO_EXTENSIONS = {"mp4"}
if prompt_path.lower().split(".")[-1] in IMAGE_EXTENSIONS:
    print("prompt is image; ignoring video_offset and n_prompt_frames")
    prompt = read_image(prompt_path)
    # add frame dimension
    prompt = rearrange(prompt, "c h w -> 1 c h w")
elif prompt_path.lower().split(".")[-1] in VIDEO_EXTENSIONS:
    prompt = read_video(prompt_path, pts_unit="sec", output_format="TCHW")[0]
    if video_offset is not None:
        prompt = prompt[video_offset:]
    prompt = prompt[:n_prompt_frames]
else:
    raise ValueError(f"unrecognized prompt file extension; expected one in {IMAGE_EXTENSIONS} or {VIDEO_EXTENSIONS}")
assert (
    prompt.shape[0] == n_prompt_frames
), f"input prompt {prompt_path} had less than n_prompt_frames={n_prompt_frames} frames"

# Input action/actions
actions_dict_list = torch.load("sample_data/Player729-f153ac423f61-20210806-224813.chunk_000.actions.pt")

# Initialize the WorldModel
wm = WorldModel(
    oasis_ckpt="/data2/cjunting/.cache/huggingface/hub/models--Etched--oasis-500m/snapshots/4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c/oasis500m.safetensors",
    vae_ckpt="/data2/cjunting/.cache/huggingface/hub/models--Etched--oasis-500m/snapshots/4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c/vit-l-20.safetensors",
    num_frames=32,
    n_prompt_frames=1,
    ddim_steps=10,
    fps=20,
)

# # Generate video
# if not os.path.exists("outputs"):
#     os.makedirs("outputs", exist_ok=True)
# video_frames = wm.run_diffusion(prompt, actions_dict_list, output_path="outputs/test_video.mp4")

# # save numpy video frames to png
# if not os.path.exists("outputs/test_video_frames"):
#     os.makedirs("outputs/test_video_frames", exist_ok=True)
# for i, frame in enumerate(video_frames):
#     Image.fromarray(frame).save(f"outputs/test_video_frames/frame_{i}.png")

# Test single step action
# Full action space, but you only need to provide non-zero actions
# action_dict = {
#     "ESC": 0,
#     "back": 0,
#     "drop": 0,
#     "forward": 0,
#     "hotbar.1": 0,
#     "hotbar.2": 0,
#     "hotbar.3": 1,
#     "hotbar.4": 0,
#     "hotbar.5": 0,
#     "hotbar.6": 0,
#     "hotbar.7": 0,
#     "hotbar.8": 0,
#     "hotbar.9": 0,
#     "inventory": 0,
#     "jump": 0,
#     "left": 1,
#     "right": 0,
#     "sneak": 0,
#     "sprint": 0,
#     "swapHands": 0,
#     "camera": torch.tensor([40, 40]),
#     "attack": 0,
#     "use": 0,
#     "pickItem": 0,
# }
forwrd_left_action = {
    "forward": 1,
    "left": 1,
    # currently, each action MUST include camera key
    "camera": torch.tensor([40, 40]),
}

forward_right_action = {
    "forward": 1,
    "right": 1,
    # currently, each action MUST include camera key
    "camera": torch.tensor([40, 40]),
}

frame = wm.step_single_action(
    prompt, forwrd_left_action, action_repeats=8, output_path="outputs/test_single_action_video.mp4"
)

# save numpy frame to png
Image.fromarray(frame).save("outputs/test_single_action_frame.png")


# Test multiple step actions
actions_dict_list = [forwrd_left_action, forward_right_action] * 5
frames = wm.step_actions(prompt, actions_dict_list, action_repeats=8)

# save numpy frames to png
if not os.path.exists("outputs/test_multi_actions_frames"):
    os.makedirs("outputs/test_multi_actions_frames", exist_ok=True)
for i, frame in enumerate(frames):
    Image.fromarray(frame).save(f"outputs/test_multi_actions_frames/frame_{i}.png")
