"""
Function to help process the LES data
"""
import numpy as np
from netCDF4 import Dataset


def get_X_ave(X, Tile_width, threshold_parameter, threshold_value=0):
    nMin = X.shape[0]
    nTiles = int(X.shape[1] / Tile_width)

    X_ave = np.zeros((nMin, nTiles, nTiles))

    for kk in range(nMin):
        index = threshold_parameter[kk, :, :] > threshold_value
        for jj in range(nTiles):
            for mm in range(nTiles):

                x1 = Tile_width * (jj)
                x2 = Tile_width * (jj + 1)

                y1 = Tile_width * (mm)
                y2 = Tile_width * (mm + 1)
                count = sum(sum(index[x1:x2, y1:y2]))
                if count != 0: X_ave[kk, jj, mm] = sum(sum(index[x1:x2, y1:y2] * X[kk, x1:x2, y1:y2])) / count
    return (X_ave)


def save_tile_width_nc_file(X, file_name, LES_nc, threshold_parameter, threshold_value=0,
                            Powers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    dataset = Dataset(file_name, 'w', format='NETCDF4_CLASSIC')
    x = dataset.createDimension('x', X.shape[1])
    y = dataset.createDimension('y', X.shape[2])
    time = dataset.createDimension('time', X.shape[0])

    dataset.createVariable('time', np.float64, ('time',))
    dataset['time'][:] = LES_nc['time'][:]
    for power in Powers:
        print('Creating tile width ', str(2 ** power))
        X_tmp = np.zeros(X.shape)
        Tile_width = 2 ** power

        variable_name = 'Tile Width %s' % (Tile_width)
        var = dataset.createVariable(variable_name, np.float64, ('time', 'y', 'x'))

        nTiles = int(X.shape[1] / Tile_width)
        X_ave = get_X_ave(X, Tile_width, threshold_parameter, threshold_value)
        for kk in range(nTiles):
            for jj in range(nTiles):
                tmp = np.repeat(np.repeat(X_ave[:, kk, jj][:, np.newaxis], Tile_width, axis=1)[:, :, np.newaxis],
                                Tile_width, axis=2)
                X_tmp[:, Tile_width * kk:Tile_width * (kk + 1), Tile_width * jj:Tile_width * (jj + 1)] = tmp
        dataset[variable_name][:] = X_tmp
    dataset.close()
