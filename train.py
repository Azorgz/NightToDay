import torch
import yaml
import json

from tqdm import tqdm

from Datasets import get_dataloaders
from NightToday import get_config


def load_config(config_path):
    """Load YAML or JSON config file."""
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be .yaml or .json")


def build_model_from_config():
    """Builds ImageToImageGAT_Dual + optional LossScheduler + segmentation nets."""
    from NightToday.NTIR2Day import Image2ImageGAT_Dual
    # --- Model creation ---
    model_params = get_config()
    model = Image2ImageGAT_Dual(model_params)
    dataloader = get_dataloaders(model_params.data)
    return model, dataloader, model_params


if __name__ == "__main__":

    # Build model from config
    model, dataloader, opt = build_model_from_config()
    total_steps = 0
    batch_size = opt.data.loader.batch_size

    for e in range(opt.training.start_epoch, opt.training.total_epochs):
        epoch_iter = 0
        bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {e+1}/{opt.training.total_epochs}")
        for i, data in bar:
            # Train step
            model.optimize_parameters(**data, epoch=e)
            errors = model.get_current_errors()
            if total_steps % opt.training.visualize_freq < batch_size:
                model.visualize_current_results(save=True)

            total_steps += batch_size
            epoch_iter += batch_size

            bar.set_description(f"epoch : {e}, loss_G : {errors['G']}, loss_D : {errors['D']}")
            torch.cuda.empty_cache()

            if i % opt.training.checkpoint_save_latest < batch_size and i != 0:
                model.save('latest')

        if e % opt.training.checkpoint_freq < batch_size:
            print(f'saving the model at the end of epoch {e}, iters {total_steps}')
            model.save(e)
        # if opt.training.test_freq > 0 and e % opt.training.test_freq==0:
        #     model.test(epoch=e)
