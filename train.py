import torch
import yaml
import json

from tqdm import tqdm

from NightToday import build_train_data_from_config
from NightToday.NTIR2Day import Image2ImageGAT_Dual


if __name__ == "__main__":

    displayed_errors = ['trafficlight', 'color']

    # Build model from config  (Default: NightToday/NightToday.yaml)
    model = Image2ImageGAT_Dual(trainable=True)
    train_dataloaders, test_dataloaders, opt = build_train_data_from_config()
    total_steps = 0
    batch_size = opt.data.loader.batch_size

    for e in range(opt.training.start_epoch, opt.training.total_epochs):
        epoch_iter = 0
        bar = tqdm(enumerate(train_dataloaders), total=len(train_dataloaders), desc=f"Epoch {e+1}/{opt.training.total_epochs}")

        for i, data in bar:
            # Train step
            model.optimize_parameters(**data, epoch=e)
            errors = model.get_current_errors()
            if total_steps % opt.training.visualize_freq < batch_size:
                model.visualize_current_results(save=True)

            total_steps += batch_size
            epoch_iter += batch_size
            list_errors = [f'{key}: {errors[key]}' for key in displayed_errors]
            bar.set_description(f"epoch : {e}, {', '.join(list_errors)}")
            torch.cuda.empty_cache()

            if i % opt.training.checkpoint_save_latest < batch_size and i != 0:
                model.save('latest')

        if e % opt.training.checkpoint_freq < batch_size:
            print(f'saving the model at the end of epoch {e}, iters {total_steps}')
            model.save(e)

        if opt.training.test_freq > 0 and e % opt.training.test_freq == 0:
            bar = tqdm(enumerate(test_dataloaders), total=len(test_dataloaders))
            for i, data in bar:
                fake_D, fused_IR = model(data, return_fused_IR=True)

