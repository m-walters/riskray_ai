import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def image_grid(arrays, nrow, ncol, save=False, save_name=None, annotations=None):
    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(nrow, ncol),
        axes_pad=0.1
    )

    for i, arr in enumerate(arrays):
        grid[i].imshow(arr)
        if annotations is not None:
            grid[i].annotate(
                f"{annotations[i]}", xy=(0.05, 0.9), xycoords='axes fraction',
                fontsize=32, color=(0.8, 0.1, 0.1),
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=None, alpha=0.5)
            )

    # Remove remaining blank axes
    n_img = len(arrays)
    [g_ax.remove() for g_ax in grid[n_img:]]

    if save:
        if not save_name:
            raise Exception("Need to provide image save name")
        else:
            plt.gca().set_axis_off()
            plt.subplots_adjust(
                top=0.97, bottom=0.03, right=1, left=0.,
                hspace=0, wspace=0
            )
            plt.margins(0, 0)
            plt.savefig(save_name)

    else:
        # Just show it
        plt.show()

    # Clear the figure
    plt.clf()


def plot_grid_df(df, nrow, ncol, start=0, num=None, save=False, save_path=None, annotate=True):
    """
    Call image_grid on some number of 'PixelArray' images in a dataframe

    :param df:              Pandas dataframe
    :param nrow:            Number of rows per image
    :param ncol:            Number of columns per image
    :param start:           Starting index of the dataframe to begin iterating through
    :param num:             Number of individual xrays to plot. If None do as many as possible
    :param save:            Turn on/off saving
    :param save_path:       Path to save file, ending with the filename.
                            Index ranges will automatically be appended to filenames
    :param annotate:        Annotate DF row integer
    """

    if save and (save_path is None or not os.path.exists(save_path.rpartition("/")[0])):
        raise Exception(f"Cannot save to save_path '{save_path}'")

    n_avail = df.shape[0] - start
    num = num or n_avail
    last = min(df.shape[0], start + num)
    n_image_per = nrow * ncol

    # Our image_grid function gracefully handles things is the number of images it's passed
    # is less than nrow * ncol
    irow = start
    print(f"Plotting {num} images for dataframe with shape {df.shape} ranging indices {start} to {last}")
    while irow < last:
        end = irow + n_image_per
        end = min(last, end)
        images = df['PixelArray'][irow:end]
        annotations = None
        if 'Annotation' in df and annotate:
            annotations = df['Annotation'][irow:end].values
        elif annotate:
            annotations = [i for i in range(irow, end)]
        print(f"\tPlotting image grid for dataframe rows {irow}:{end}")
        save_name = None
        if save:
            save_name = save_path.rpartition('.')[0] if '.' in save_path else save_path
            ext = save_path.rpartition('.')[1] if '.' in save_path else "jpg"
            save_name += f"_{irow}-{end - 1}.{ext}"

        image_grid(images, nrow, ncol, save, save_name, annotations)

        irow += n_image_per
    print("Done")
