"""
List of functions to produce figures describing output of reconstruction toolkit

Laurence Jackson, BME, KCL, 2019
"""

import matplotlib
matplotlib.use('Agg')  # use agg backend to prevent figure plotting - no need for x11 permissions when running remotely
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np


# disable interactive plots
plt.ioff()


def plot_respiration_frequency(freqmat, respii, freqs, centerline, width, sz, name='respiration_frequency'):

    plt.imshow(freqmat[:, respii[0]], aspect='auto')
    plt.imshow(freqmat, vmax=np.mean(freqmat) * 1.5, extent=(freqs[0], freqs[sz[2] - 1], sz[0], 0), aspect='auto')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Row index')
    plt.axhline(centerline - width, color='black')
    plt.axhline(centerline, color='red')
    plt.axhline(centerline + width, color='black')

    plt.savefig('fig_' + name, dpi=300)
    plt.close()


def plot_respiration_summary(img_sum, y_mean, resp, sts, name='respiration_summary'):

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[2, 1], height_ratios=[1, 1, 1], figure=fig)

    ax1 = plt.subplot(gs[0])
    qq = get_expand_ylim(img_sum, 0.1, 0.9, edge_factor=0.15)
    ax1.set_ylim(qq[0], qq[1])

    ax1.set_title('Raw body area')
    ax1.scatter(range(0, img_sum.shape[0]), img_sum, marker=".", c='r', s=1)
    ax1.plot(range(0, img_sum.shape[0]), y_mean, c='k')
    ax2 = plt.subplot(gs[1])
    ax2.set_title('Raw body area zoom')
    ax2.set_yticklabels([])
    ax2.scatter(range(0, img_sum.shape[0]), img_sum, marker="o", c='r', s=5)
    extractx = np.arange(img_sum.shape[0])
    extractx = extractx[
        range(img_sum.shape[0] - int(0.2 * img_sum.shape[0]), img_sum.shape[0] - int(0.1 * img_sum.shape[0]))]
    extracty = img_sum[
        range(img_sum.shape[0] - int(0.2 * img_sum.shape[0]), img_sum.shape[0] - int(0.1 * img_sum.shape[0]))]
    ax2.set_xlim([np.min(extractx), np.max(extractx)])
    ax2.set_ylim([np.min(extracty) - 50, np.max(extracty) + 50])
    rect = patches.Rectangle((np.min(extractx), np.min(extracty) - 50), (np.max(extractx) - np.min(extractx)),
                             ((np.max(extracty) + 50) - (np.min(extracty) - 50)), linewidth=1, edgecolor='k',
                             facecolor='none')
    ax1.add_patch(rect)

    ax3 = plt.subplot(gs[2])
    qq = get_expand_ylim(resp, 0.1, 0.9, edge_factor=0.15)
    ax3.set_ylim(qq[0], qq[1])
    ax3.set_title('Filtered respiration')
    ax3.plot(range(0, resp.shape[0]), resp, c='k', linewidth=0.5, zorder=1)
    ax3.scatter(range(0, resp.shape[0]), resp, marker=".", c='r', s=1, zorder=2)

    ax4 = plt.subplot(gs[3])
    ax4.set_title('Filtered respiration zoom')
    ax4.set_yticklabels([])
    ax4.plot(range(0, resp.shape[0]), resp, c='k', linewidth=1)
    ax4.scatter(range(0, resp.shape[0]), resp, marker="o", c='r', s=5)
    extractx = np.arange(resp.shape[0])
    extractx = extractx[range(resp.shape[0] - int(0.2 * resp.shape[0]), resp.shape[0] - int(0.1 * resp.shape[0]))]
    extracty = resp[range(resp.shape[0] - int(0.2 * resp.shape[0]), resp.shape[0] - int(0.1 * resp.shape[0]))]
    ax4.set_xlim([np.min(extractx), np.max(extractx)])
    ax4.set_ylim([np.min(extracty) - 50, np.max(extracty) + 50])
    rect = patches.Rectangle((np.min(extractx), np.min(extracty)- 50), (np.max(extractx) - np.min(extractx)),
                             ((np.max(extracty) + 50) - (np.min(extracty) - 50)), linewidth=1, edgecolor='k',
                             facecolor='none')
    ax3.add_patch(rect)

    ax5 = plt.subplot(gs[4])
    qq = get_expand_ylim(resp, 0.1, 0.9, edge_factor=0.15)
    ax5.set_ylim(qq[0], qq[1])
    ax5.set_title('Classification')
    ax5.plot(range(0, resp.shape[0]), resp, c='k', linewidth=0.5, zorder=5)
    ax5.scatter(range(0, resp.shape[0]), resp, c=sts, s=50, zorder=10)

    ax6 = plt.subplot(gs[5])
    ax6.set_title('Classification zoom')
    ax6.set_yticklabels([])
    ax6.plot(range(0, resp.shape[0]), resp, c='k', linewidth=1, zorder=5)
    ax6.scatter(range(0, resp.shape[0]), resp, marker="o", c=sts, s=50, zorder=10)
    extractx = np.arange(resp.shape[0])
    extractx = extractx[range(resp.shape[0] - int(0.2 * resp.shape[0]), resp.shape[0] - int(0.1 * resp.shape[0]))]
    extracty = resp[range(resp.shape[0] - int(0.2 * resp.shape[0]), resp.shape[0] - int(0.1 * resp.shape[0]))]
    ax6.set_xlim([np.min(extractx), np.max(extractx)])
    ax6.set_ylim([np.min(extracty) - 50, np.max(extracty) + 50])
    rect = patches.Rectangle((np.min(extractx), np.min(extracty)- 50), (np.max(extractx) - np.min(extractx)),
                             ((np.max(extracty) + 50) - (np.min(extracty) - 50)), linewidth=1, edgecolor='k',
                             facecolor='none', zorder=20)
    ax5.add_patch(rect)

    plt.tight_layout()

    plt.savefig('fig_respiration_summary', dpi=300)
    plt.close(fig)


def get_expand_ylim(data, qlo, qhi, edge_factor=0.15):
    """ get ylim for quartile range + edge_factor"""

    yy_hi = (1 + edge_factor) * np.quantile(data, qhi)
    yy_lo_q = np.quantile(data, qlo)

    if yy_lo_q < 0:
        yy_lo = (1 + edge_factor) * yy_lo_q
    else:
        yy_lo = (1 - edge_factor) * yy_lo_q

    return [yy_lo, yy_hi]
