from scipy.io import loadmat

if __name__ == '__main__':
    d = loadmat('/home/jiman/facedata/imdb/imdb_o.mat')
    print(len(d['full_path']),len(d['age']))
