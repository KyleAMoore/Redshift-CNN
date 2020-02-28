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
                 b=0.055,
                 image_channels=5):
        self.gamma = gamma
        self.b = b
        self.a = a
        self.alpha = alpha
        self.it = it

    def process_img(self,image):
        """
            Runs image through all processing steps described by Bertin
        """
        img_norm = self.normalize_brightness(image)

        channel_one = img_norm[:, :, 0]
        channel_two = img_norm[:, :, 1]
        channel_three = img_norm[:, :, 2]
        assert channel_one.shape == (img_norm.shape[0], img_norm.shape[1])
        assert channel_two.shape == (img_norm.shape[0], img_norm.shape[1])
        assert channel_two.shape == (img_norm.shape[0], img_norm.shape[1])
        rgb_matrix = self.map_colors(channel_one, channel_two, channel_three)
        assert rgb_matrix.shape == (img_norm.shape[0], img_norm.shape[1], 3)
        final_rgb_matrix = self.gamma_correct(rgb_matrix)

        return np.clip(final_rgb_matrix,0,1)

    def normalize_brightness(self, images):
        """
            Experimenting with the surface brightness cuts described in
            Bertin found that the computed min and max are both zero.
            This is presumably because the bightness normalization
            is only necessary in relatively sparse views.

            As such, no meaningful normalization is deemed necessary
            and this function simply clips the intensity to the range
            0-255
        """
        return np.clip(images, 0, 1)

    def ugriz_to_rgb(self, image):
        """
            converts a 5-channel ugriz image to 3-channel rgb. No information
            on more complicated methods have been found, so this function maps
            the i, r, and g channels to r, g, b.

            Note that the r and g channels in the two encodings do not match up.
            This is to match convention of SDSS mentioned in
            https://www.sdss.org/dr16/imaging/jpg-images-on-skyserver/
        """
        return image[:,:,3:0:-1]

    def map_colors(self, r, g, b):
        """
            Adjusts colors for a given image. Color adjustments are approximately
            equivalent to scaling the color saturation of the image by a factor
            of alpha
        """
        Y = (r + g + b) / 3
        r_alpha = (Y + self.alpha * ((2 * r - g - b)/3))
        g_alpha = (Y + self.alpha * ((2 * g - r - b)/3))
        b_alpha = (Y + self.alpha * ((2 * b - r - g)/3))
        return np.stack([r_alpha, g_alpha, b_alpha], axis=-1)

    def gamma_correct(self, rgb_matrix):
        """
            Applies gamma compression to image. This is necessary due to the
            gamma conversion performed on all SDSS images to bring measured
            passband magnitudes to their correct values
        """
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
            try:
                for i in sample(list(range(len(b['img']))), img_per_bin):
                    samples[b_n]['img'].append(b['img'][i])
                    samples[b_n]['rs'].append(b['rs'][i])
            except ValueError:
                samples[b_n]['img'] = b['img']
                samples[b_n]['rs'] = b['rs']

        bin_maxes = [float(minY + bin_width * (i+1)) for i in range(num_bins)]


        # Convert images to RGB format (with preprocessing) and add them to plot
        fig, splts = plt.subplots(img_per_bin if img_per_bin > 1 else 6,
                                  num_bins,
                                  figsize=(num_bins*1.5,
                                           (img_per_bin if img_per_bin > 1 else 6)*1.5))
        for b in range(num_bins):
            if img_per_bin == 1:
                # convert image into 5 grayscale images where the intensity is a
                # single ugriz channel intensity
                bin_images = [np.array([
                    [
                        [col[chan]] * 3 for col in row
                    ] for row in samples[b]['img'][0]
                ]) for chan in range(5)]
                bin_images.append(self.ugriz_to_rgb(samples[b]['img'][0]))

            else:
                bin_images = [self.ugriz_to_rgb(s) for s in samples[b]['img']]

            bin_images = [self.process_img(img) for img in bin_images]
            for c in range(len(bin_images)):
                splts[c][b].imshow(bin_images[c])
    

        # Format plot axes and labels
        for b in range(num_bins):
            for c in range(img_per_bin if img_per_bin > 1 else 6):
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

        ylabels = list(range(1,img_per_bin+1)) if img_per_bin > 1 else ['u','g','r','i','z','rgb']
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

    rsvis = BetlinVisualizer(alpha=1.2)
    rsvis.plot_images(images, redshifts, 
                      'binned-galaxies-mult.png',
                      num_bins=10, img_per_bin=1)

    rsvis.plot_images(images, redshifts, 
                      'binned-galaxies-decomp.png',
                      num_bins=10, img_per_bin=8)
