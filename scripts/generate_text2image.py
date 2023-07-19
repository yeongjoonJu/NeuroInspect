import argparse, os
import torch

from diffusers import DiffusionPipeline


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A painting of a cat")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="results/unseen_images")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(f"cuda:{args.device}")
    
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.to(device)
    
    for i in range(10):
        images = pipeline(args.prompt, num_images_per_prompt=args.batch_size, num_inference_steps=50).images
        
        for j, img in enumerate(images):
            img.save(f"{args.save_dir}/gen_{i}_{j}.jpg")