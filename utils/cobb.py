#import matlab.engine #should be put at the top
import numpy as np

def angleCal_py(landmark, H, W):
    eng = matlab.engine.start_matlab()
    landmark = matlab.double(landmark.tolist())
    H = matlab.double(H.tolist())
    W = matlab.double(W.tolist())

    eng.addpath(r'./utils/', nargout = 0)
    cobb = eng.angleCal(landmark, H, W)

    cobb = np.array(cobb)
    eng.quit()

    return cobb

if __name__ == '__main__':
    import scipy.io as scio
    p2 = scio.loadmat('./dataset/boostnet_labeldata/labels/training/sunhl-1th-02-Jan-2017-162 A AP.jpg.mat')['p2']
    cobb = angleCal_py(p2,2125,775)
    print(cobb)