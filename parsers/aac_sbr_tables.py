import math
from parsers.aac_utils import minInt, aacRound

startOffset = [
    [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],  # sfi = 8
    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13],  # sfi = 7
    [-5, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16],  # sfi = 6
    [-6, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16],  # sfi = 5
    [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16, 20],  # sfi = 4...2
    [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16, 20, 24],  # sfi < 2
]

startMin = [
    7, 7, 10, 11, 12, 16, 16, 17, 24, 32, 35, 48,
]

stopOffset = [
    [0, 2, 4, 6, 8, 11, 14, 18, 22, 26, 31, 37, 44, 51],
	[0, 2, 4, 6, 8, 11, 14, 18, 22, 26, 31, 36, 42, 49],
	[0, 2, 4, 6, 8, 11, 14, 17, 21, 25, 29, 34, 39, 44],
	[0, 2, 4, 6, 8, 11, 14, 17, 20, 24, 28, 33, 38, 43],
	[0, 2, 4, 6, 8, 11, 14, 17, 20, 24, 28, 32, 36, 41],
	[0, 2, 4, 6, 8, 10, 12, 14, 17, 20, 23, 26, 29, 32],
	[0, 2, 4, 6, 8, 10, 12, 14, 17, 20, 23, 26, 29, 32],
	[0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 23, 26, 29],
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, -1, -2, -3, -4, -5, -6, -6, -6, -6, -6, -6, -6, -6],
	[0, -3, -6, -9, -12, -15, -18, -20, -22, -24, -26, -28, -30, -32],
]

stopMin = [
    13, 15, 20, 21, 23, 32, 32, 35, 48, 64, 70, 96,
]

def derive_sbr_tables(data, sfi, bs_start_freq, bs_stop_freq, 
                      bs_freq_scale, bs_alter_scale, bs_xover_band):
    data.k0 = qmf_lower_boundary(bs_start_freq, sfi)
    data.k2 = qmf_upper_boundary(bs_stop_freq, sfi, data.k0)

    if sfi < 4:
        if data.k2 - data.k0 > 32:
            raise ValueError(f"Error: k0 ({data.k0}) and k2 ({data.k2}) out of range")
    elif sfi > 4:
        if data.k2 - data.k0 > 48:
            raise ValueError(f"Error: k0 ({data.k0}) and k2 ({data.k2}) out of range")
    else:
        if data.k2 - data.k0 > 45:
            raise ValueError(f"Error: k0 ({data.k0}) and k2 ({data.k2}) out of range")

    if bs_freq_scale == 0:
        freq_master_fs0(data, data.k0, data.k2, bs_alter_scale)
    else:
        freq_master(data, data.k0, data.k2, bs_freq_scale, bs_alter_scale)

    err = freq_derived(data, bs_xover_band, data.k2)
    if err != None:
        return err

    return None

def qmf_lower_boundary(bs_start_freq, sfi):
    i = 5
    if sfi == 8:
        i = 0
    elif sfi == 7:
        i = 1
    elif sfi == 6:
        i = 2
    elif sfi == 5:
        i = 3
    elif sfi in [4, 3, 2]:
        i = 4
    elif sfi in [0, 1]:
        i = 5

    return startMin[sfi] + startOffset[i][bs_start_freq]

def qmf_upper_boundary(bs_stop_freq, sfi, k0):
    val = minInt(64, stopMin[sfi] + stopOffset[sfi][minInt(13, bs_stop_freq)])
    return val

def freq_master_fs0(data, k0, k2, bs_alter_scale):
    dk = 1
    numBands = 0
    if bs_alter_scale == 0:
        numBands = ((k2 - k0) >> 1) << 1
    else:
        dk = 2
        numBands = ((k2 - k0 + 2) >> 2) << 1

    numBands = minInt(63, numBands)
    k2Achieved = int(k0) + numBands * dk
    k2Diff = int(k2) - k2Achieved

    vDk = [dk] * numBands
    if k2Diff != 0:
        if k2Diff < 0:
            incr = 1
            k = 0
        else:
            incr = -1
            k = numBands - 1

        while k2Diff != 0:
            vDk[k] -= incr
            k += incr
            k2Diff += incr

    data.f_master = [0] * numBands
    data.f_master[0] = int(k0)
    for k in range(1, numBands):
        data.f_master[k] = data.f_master[k - 1] + vDk[k - 1]
    data.N_master = numBands

def freq_master(data, k0, k2, bs_freq_scale, bs_alter_scale):
    twoRegions = 0
    k1 = k2

    if float(k2) / float(k0) > 2.2449:
        twoRegions = 1
        k1 = k0 * 2

    temp1 = [12.0, 10.0, 8.0]
    bands = temp1[bs_freq_scale - 1]
    k0_f = float(k0)
    k1_f = float(k1)
    k2_f = float(k2)

    numBands0 = 2 * aacRound(bands * math.log10(k1_f / k0_f) / (2.0 * math.log10(2)))

    vDk0 = [0] * (numBands0 + 1)
    for k in range(numBands0 + 1):
        vDk0[k] = aacRound(k0_f * (math.pow((k1_f / k0_f), float(k + 1) / float(numBands0)))) - \
                  aacRound(k0_f * math.pow((k1_f / k0_f), float(k) / float(numBands0)))

    vDk0.sort()

    vk0 = [0] * (numBands0 + 1)
    vk0[0] = int(k0)
    for k in range(1, numBands0 + 1):
        vk0[k] = vk0[k - 1] + vDk0[k - 1]

    if twoRegions == 0:
        data.N_master = numBands0
        data.f_master = vk0[:numBands0 + 1]
        return

    warp = 1.0
    if bs_alter_scale == 1:
        warp = 1.3

    numBands1 = 2 * aacRound(bands * math.log10(k2_f / k1_f) / (2.0 * math.log10(2.0) * warp))
    vDk1 = [0] * (numBands1 - 1)
    for k in range(numBands1 - 1):
        vDk1[k] = aacRound(k1_f * math.pow(k2_f / k1_f, float(k + 1) / float(numBands1))) - \
                  aacRound(k1_f * math.pow(k2_f / k1_f, float(k) / float(numBands1)))

    vDk1.sort()

    if vDk1[0] < vDk0[-1]:
        change = vDk0[-1] - vDk1[0]
        if change > (vDk1[numBands1 - 1] - (vDk1[0] * 2)):
            change = (vDk1[numBands1 - 1] - (vDk1[0] * 2))
        vDk1[0] += change
        vDk1[numBands1 - 1] -= change

    vk1 = [0] * numBands1
    vk1[0] = int(k1)
    for k in range(1, numBands1):
        vk1[k] = vk1[k - 1] + vDk1[k - 1]

    data.N_master = numBands0 + numBands1
    data.f_master = vk0 + vk1

def freq_derived(data, bs_xover_band, k2):
    data.N_high = data.N_master - bs_xover_band
    data.N_low = (data.N_high >> 1) + (data.N_high - ((data.N_high >> 1) << 1))

    data.n = [data.N_low, data.N_high]

    index = data.N_high + bs_xover_band + 1

    if index < bs_xover_band or len(data.f_master) < index:
        raise ValueError(f"f_tablehigh invalid index: index ({index}) must be between bs_xover_band ({bs_xover_band}) and length of f_master ({len(data.f_master)})")
    
    data.f_tablehigh = data.f_master[bs_xover_band:index]

    if len(data.f_tablehigh) < data.N_high:
        raise ValueError(f"N_high index ({data.N_high}) too high for length of f_tablehigh ({len(data.f_tablehigh)})")
    
    data.M = data.f_tablehigh[data.N_high] - data.f_tablehigh[0]
    data.k_x = data.f_tablehigh[0]

    data.f_tablelow = [0] * (data.N_low + 1)
    for k in range(len(data.f_tablelow)):
        if k != 0:
            i = 2 * k - (data.N_high % 2)
            data.f_tablelow[k] = data.f_tablehigh[i]
        else:
            data.f_tablelow[k] = data.f_tablehigh[0]

    data.N_Q = 0
    bs_noise_bands = data.Sbr_header.Bs_noise_bands
    k2_f = float(data.k2)
    k_x_f = float(data.k_x)
    data.N_Q = max(1, aacRound(bs_noise_bands * (math.log10(k2_f / k_x_f) / math.log10(2))))

    data.f_tablenoise = [0] * (data.N_Q + 1)
    i = 0
    for k in range(len(data.f_tablenoise)):
        if k != 0:
            i = i + (data.N_low - i) // (data.N_Q + 1 - k)
        data.f_tablenoise[k] = data.f_tablelow[i]

    return None
