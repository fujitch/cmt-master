# -*- coding: utf-8 -*-

'''
イメージ出力用スクリプト
'''

from scipy.fftpack import fft
import matplotlib.pyplot as plt

for i in range(432):
    if i%9 != 0:
        continue
    if i/144 == 0:
        if (i/72)%2 == 0:
            fname = 'train_org_rr_normal' + str((i/9)%2 + 1) + '_p' + str((i%72)/18 + 1)
        else:
            fname = 'train_org_rr_normal' + str((i/9)%2 + 1) + '_p' + str((i%72)/18 + 1) + '_n'
    elif i/144 == 1:
        if (i/72)%2 == 0:
            fname = 'train_org_rr_inner' + str((i/9)%2 + 1) + '_p' + str((i%72)/18 + 1)
        else:
            fname = 'train_org_rr_inner' + str((i/9)%2 + 1) + '_p' + str((i%72)/18 + 1) + '_n'
    else:
        if (i/72)%2 == 0:
            fname = 'train_org_rr_mis' + str((i/9)%2 + 1) + '_p' + str((i%72)/18 + 1)
        else:
            fname = 'train_org_rr_mis' + str((i/9)%2 + 1) + '_p' + str((i%72)/18 + 1) + '_n'
            
    plt.figure()
    x = range(100)
    plt.plot(x, batch[i, :])
    plt.ylim(0, 200)
    plt.savefig(fname)