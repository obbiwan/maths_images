"""
Contains functions for plotting data

Written by: Benjamin Smith
     email: ben297@gmail.com
    github: https://github.com/obbiwan
"""
import matplotlib.pyplot as plt
import os
from utils import PlotData


def scatter(data: PlotData, dpi: int = 500, image_width: float = 10.0, image_height: float = 10.0, axis: bool = True,
            plot_title_font_size: int = 32, save_to_file: bool = False):
    """
    Create a scatter plot using the provided data
    :param data: a PlotData dictionary, containing the data to plot
    :param dpi: The resolution of the output image, in dots per inch
    :param image_width: The width of the output image, in inches
    :param image_height: The height of the output image, in inches
    :param axis: Whether to draw the horizontal and vertical axes on the plot
    :param plot_title_font_size: The font size of the plot title
    :param save_to_file: Whether to save the plot to an image file. If False, the plot will be shown but not saved.
    """
    # Create an empty figure
    fig = plt.figure(figsize=(image_width, image_height),
                     dpi=dpi)

    # Print the title on the plot
    fig.suptitle(data['plot_title'], fontsize=plot_title_font_size)

    # Disable the plot axis if necessary
    if not axis:
        plt.axis('off')

    # Set the plot area
    plt.xlim(data['x_min'], data['x_max'])
    plt.ylim(data['y_min'], data['y_max'])

    # Plot the data
    plt.scatter(data['x'], data['y'], marker='o', s=(72.0/dpi), lw=0)

    # The output directory of saved plots
    output_dir = 'images'

    # Save the plot to a file (or just show it)
    if save_to_file:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        plt.savefig(os.path.join(output_dir, f"{data['filename']}.png"))
    else:
        plt.show()
