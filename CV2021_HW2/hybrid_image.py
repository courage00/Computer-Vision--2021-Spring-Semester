import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


class Hybrid(object):
    """ Hybrid Image"""
    def __init__(self, img1, img2, s):
        self.img1 = np.asarray(img1)
        self.img2 = np.asarray(img2)
        self.shift = s
        # print(self.img1.shape)

    def DoHybrid(self, d0_h, d0_l):
        h_img_gaussian = np.zeros(self.img1.shape)
        l_img_gaussian = np.zeros(self.img1.shape)
        h_img_ideal = np.zeros(self.img1.shape)
        l_img_ideal = np.zeros(self.img1.shape)

        for k in range(3):
            h_img_g, h_img_i = self.cal_hybrid(self.img1[:, :, k], d0_h, 1)
            l_img_g, l_img_i = self.cal_hybrid(self.img2[:, :, k], d0_l, 0)
            for i in range(self.img1.shape[0]):
                for j in range(self.img1.shape[1]):
                    h_img_gaussian[i, j, 2-k] = np.real(h_img_g[i, j])
                    l_img_gaussian[i, j, 2-k] = np.real(l_img_g[i, j])
                    h_img_ideal[i, j, 2-k] = np.real(h_img_i[i, j])
                    l_img_ideal[i, j, 2-k] = np.real(l_img_i[i, j])

        hybrid_img_g = h_img_gaussian + l_img_gaussian
        hybrid_img_i = h_img_ideal + l_img_ideal

        return hybrid_img_g, hybrid_img_i

    def center_transform(self, img):
        '''Multiply by (-1)^(x+y) to center the transform.'''
        img_out = np.copy(img)
        rows, cols = img_out.shape
        for x in range(rows):
            for y in range(cols):
                img_out[x][y] = img[x][y] * ((-1)**(x+y))

        return img_out

    def makefiltermatrix(self, row, col, d0, highpass):
        centeri = int(row/2)+1 if row%2==1 else int(row/2)
        centerj = int(col/2)+1 if col%2==1 else int(col/2)

        def gaussian(i, j):
            d = ((i-centeri)**2 + (j-centerj)**2)**(1/2)
            coef = np.exp(-1*d**2 / (2*d0**2))
            if highpass == 1:
                coef = 1 - coef
            return coef

        def ideal(i, j):
            d = ((i-centeri)**2 + (j-centerj)**2)**(1/2)
            coef = 1 if d <= d0 else 0
            if highpass == 1:
                coef = 1 - coef
            return coef

        gaussian_filter = np.array(
            [[gaussian(i, j) for j in range(col)] for i in range(row)])
        ideal_filter = np.array(
            [[ideal(i, j) for j in range(col)] for i in range(row)])

        return gaussian_filter, ideal_filter

    def filterDFT(self, img, filtermatrix):
        # Compute Fourier transformation of input image, i.e. F(u,v).
        if self.shift:
            shiftedDFT = fftshift(fft2(img))
        else:
            shiftedDFT = fft2(self.center_transform(img.astype(np.float32)))

        # Multiply F(u,v) by a filter function H(u,v).
        filteredDFT = shiftedDFT * filtermatrix

        # Compute the inverse Fourier transformation of the result
        if self.shift:
            return ifft2(ifftshift(filteredDFT))
        else:
            filteredDFT = ifft2(filteredDFT)
            return self.center_transform(filteredDFT)

    def cal_hybrid(self, img, d0, highpass):
        """Compute the hybrid image
        highpass 1: for high-pass filter
        highpass 0: for low-pass filter
        """
        row, col = img.shape
        gaussianfilter, idealfilter = self.makefiltermatrix(
            row, col, d0, highpass)

        return self.filterDFT(img, gaussianfilter), self.filterDFT(img, idealfilter)