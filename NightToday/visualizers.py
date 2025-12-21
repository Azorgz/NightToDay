from ImagesCameras import ImageTensor
from . import TrainConfig


class Visualizer:
    def __init__(self, opt: TrainConfig):
        self.save_dir = opt.visualize_dir
        self.display_freq = opt.visualize_freq
        self.size = opt.input_size
        self.screen = None

    def save_current_results(self, visuals: dict[str, ImageTensor], epoch: int):
        """Saves the current visuals to disk."""

        for label, image_tensor in visuals.items():
            image_tensor.save(self.save_dir, name=f'{label}_epoch_{epoch}', ext='png', depth=8)

    def display_current_results(self, visuals: dict[str, ImageTensor]):
        """Display the current visuals on screen."""
        labels = ['real_D', 'real_T', 'real_N', 'fake_T', 'remapped_T', 'real_TN', 'rec_D', 'rec_T', 'fake_D']
        shape = visuals['real_D'].shape
        composition = None
        row = None
        for i, label in enumerate(labels):
            image = visuals[label].RGB().cpu() if label in visuals else ImageTensor.rand(*shape) * 0
            if row is None:
                row = image
            else:
                row = row.hstack(image)
            if (i+1) % 3 == 0:
                if composition is None:
                    composition = row
                else:
                    composition = composition.vstack(row)
                row = None
        if self.screen is not None:
            self.screen.update(composition)
        else:
            self.screen = composition.show(name=f'Learning on going...', opencv=True, asyncr=True)


