import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # read matplotlib figure from file 'warped_points.pickle' and show it
    with open('warped_points.pickle', 'rb') as f:
        fig = pickle.load(f)

    plt.show()
