import os
import gif
import matplotlib.pyplot as plt
import tqdm
import time



class ProgressBar:
    def __init__(self, num_iterations: int, verbose: bool = True):
        if verbose:  # create a nice little progress bar
            self.scalar_tracker = tqdm.tqdm(total=num_iterations, desc="Scalars", bar_format="{desc}",
                                            position=0, leave=True)
            progress_bar_format = '{desc} {n_fmt:' + str(
                len(str(num_iterations))) + '}/{total_fmt}|{bar}|{elapsed}<{remaining}'
            self.progress_bar = tqdm.tqdm(total=num_iterations, desc='Iteration', bar_format=progress_bar_format,
                                          position=1, leave=True)
        else:
            self.scalar_tracker = None
            self.progress_bar = None

    def __call__(self, **kwargs):
        if self.progress_bar is not None:
            formatted_scalars = {key: "{:.3e}".format(value[-1] if isinstance(value, list) else value)
                                 for key, value in kwargs.items()}
            description = ("Scalars: " + "".join([str(key) + "=" + value + ", "
                                                  for key, value in formatted_scalars.items()]))[:-2]
            self.scalar_tracker.set_description(description)
            self.progress_bar.update(1)


# specify the path to save the recordings of this run to.
data_path = 'data'
data_path = os.path.join(data_path, time.strftime("%d-%m-%Y_%H-%M"))
if not (os.path.exists(data_path)):
    os.makedirs(data_path)


# this function will automatically save your figure
def save_figure(save_name: str) -> None:
    assert save_name is not None, "Need to provide a filename to save to"
    plt.savefig(os.path.join(data_path, save_name + ".png"))


def save_gif(frames: list, save_name: str) -> None:
    assert save_name is not None, "Need to provide a filename to save to"
    gif.save(frames, os.path.join(data_path, save_name + ".gif"), duration=3.5, unit="s", between="startend")
