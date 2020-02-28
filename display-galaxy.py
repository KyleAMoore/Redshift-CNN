import matplotlib.pyplot as plt
import numpy as np
from random import sample

class BetlinVisualizer:
    """Given an image in 5 SDSS bands,
       create an intensity plot using these images.
       This method follows recommendations from to plot these
       images.
       http://www.aspbooks.org/publications/461/263.pdf
    """
    def __init__(self,
                 gamma=2.4,
                 alpha=2.0,
                 a=12.92,
                 it=0.00304,
                 b=0.55,
                 image_channels=5):
        self.gamma = gamma
        self.b = b
        self.a = a
        self.alpha = alpha
        self.it = it
        # self.image_shape = image_shape

    def convert_to_rgb(self,image):
        return [[[c[3],c[2],c[1]] for c in r] for r in image]

    def save_jpeg(self, images, channels_to_use=(1, 2, 3)):
        """Save the jpeg image of the galaxies
        This method assumes that we are using images as a
        """
        for image in images:
            channel_one = image[:, :, channels_to_use[0]]
            channel_two = image[:, :, channels_to_use[1]]
            channel_three = image[:, :, channels_to_use[2]]
            assert channel_one.shape == (image.shape[0], image.shape[1])
            assert channel_two.shape == (image.shape[0], image.shape[1])
            assert channel_two.shape == (image.shape[0], image.shape[1])
            rgb_matrix = self.map_colors(channel_one, channel_two, channel_three)
            print(rgb_matrix.shape)
            assert rgb_matrix.shape == (image.shape[0], image.shape[1], 3)
            final_rgb_matrix = self.gamma_correct(rgb_matrix)
            plt.imshow(final_rgb_matrix)
            plt.show()

    def map_colors(self, r, g, b):
        Y = (r + g + b) / 3
        r_alpha = (Y + self.alpha * ((2 * r - g - b)/3))
        g_alpha = (Y + self.alpha * ((2 * g - r - b)/3))
        b_alpha = (Y + self.alpha * ((2 * b - r - g)/3))
        return np.stack([r_alpha, g_alpha, b_alpha], axis=-1)

    def gamma_correct(self, rgb_matrix):
        final_rgb_matrix = np.where(np.logical_and(rgb_matrix >= 0, rgb_matrix < self.it),
                                    rgb_matrix * self.a,
                                    (1 + self.b) * rgb_matrix**(1/self.gamma) - self.b)
        return final_rgb_matrix

    def plot_images(self, images, redshifts,
                    output_filename,
                    num_bins = 7,
                    img_per_bin = 1):
        """
            The main method of this class. Generates a figure for comparing
            galaxy images across redshift value classes. The range of the
            classes is evenly split along the range of provided redshift
            values. Images from each class are selected for display uniformly
            at random.

            The form of the figure is dependant upon the number of images per
            class. For n > 1, n galaxies in each class are displayed in RGB.
            For n = 1 classes, one image from each class is selected and
            displayed in RGB, along with an intensity image for each of the 5
            UGRIZ image channels (giving a total of 6 images per galaxy/class).
        """
        assert img_per_bin >= 1
        assert num_bins >= 1
        assert len(images) == len(redshifts)
        assert len(images) >= num_bins

        if not output_filename.lower().endswith('.png'):
            output_filename += '.png'

        # Bin images by redshift
        bins = [{'img': [], 'rs': []} for _ in range(num_bins)]

        maxY = max(redshifts)
        minY = min(redshifts)
        # 0.005 addition lengthens range negligibly to ensure max is included in last bin
        bin_width = (maxY - minY + 0.005) / num_bins

        for i in range(len(images)):
            bin_num = int(redshifts[i] / bin_width)
            bins[bin_num]['img'].append(images[i])
            bins[bin_num]['rs'].append(redshifts[i])

        samples = [{'img': [], 'rs': []} for _ in range(num_bins)]
        for b_n,b in enumerate(bins):
            for i in sample(list(range(len(b['img']))), img_per_bin):
                samples[b_n]['img'].append(b['img'][i])
                samples[b_n]['rs'].append(b['rs'][i])

        bin_maxes = [float(minY + bin_width * (i+1)) for i in range(num_bins)]



        fig, splts = plt.subplots(img_per_bin if img_per_bin > 1 else 6,
                                  num_bins,
                                  figsize=(num_bins*1.5,
                                           (img_per_bin if img_per_bin > 1 else 6)*1.5))
        for b in range(num_bins):
            if img_per_bin == 1:
                # converts image to list of images where each image contains a grayscale representation of a single color channel
                bin_images = [np.array([[[col[chan]] * 3 for col in row] for row in samples[b]['img'][0]]) for chan in range(5)]
                bin_images.append(self.convert_to_rgb(samples[b]['img'][0]))

            else:
                bin_images = [self.convert_to_rgb(s) for s in samples[b]['img']]
            for c in range(len(bin_images)):
                splts[c][b].imshow(bin_images[c])
                splts[c][b].tick_params(
                    axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False
                )
                splts[c][b].tick_params(
                    axis='y',
                    which='both',
                    left=False,
                    right=False,
                    labelleft=False
                )
        for b in range(num_bins):
            splts[-1][b].set_xlabel("{:.4f}-{:.4f}".format(0 if b==0 else bin_maxes[b-1],
                                                           bin_maxes[b]))

        ylabels = list(range(img_per_bin)) if img_per_bin > 1 else ['u','g','r','i','z','rgb']
        for i,c in enumerate(ylabels):
            splts[i][0].set_ylabel(c, fontsize='large', rotation='horizontal', labelpad=10)

        fig.text(0.5,0.06,
                 'Redshift bins',
                 ha='center', va='center', fontsize='x-large')
        fig.text(0.1,0.5,
                 'Samples' if img_per_bin > 1 else 'Photometric Channels',
                 ha='center', va='center', fontsize='x-large', rotation='vertical')
    
        # plt.show()
        plt.savefig(output_filename, bbox_inches='tight')

   
if __name__ == "__main__":
    from pickle import load
    with open('data/combined_dataset.pkl', 'rb') as pklfile:
        images, redshifts = load(pklfile).values()

    rsvis = BetlinVisualizer()
    rsvis.plot_images(images, redshifts, 
                      'binned-galaxies-decomp.png',
                      num_bins=12, img_per_bin=1)
