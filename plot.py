import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot


def plot_acc_loss(loss, acc):
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)
    par1 = host.twinx()

    host.set_xlabel("steps")
    host.set_ylabel("validation-loss")
    host.set_ylabel("validation-accuracy")

    p1, = host.plot(range(len(loss)), loss, label="loss")
    p2, = par1.plot(range(len(acc)), acc, label="accuracy")

    host.legend(loc=5)

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    plt.draw()
    plt.show()


#validation
# the data is manually get from the results picture
RNN_loss_list = [0.721, 0.731, 0.780, 0.793, 0.804, 0.863, 0.825, 0.679, 0.677, 0.683, 0.684, 0.690, 0.694, 0.686, 0.691]
RNN_acc_list = [0.4183, 0.4048, 0.4137, 0.5936, 0.4427, 0.5919, 0.5930, 0.5696, 0.5903, 0.5981, 0.5675, 0.5687, 0.5970, 0.5675, 0.5970]

LSTM_loss_list = [0.534, 0.332, 0.319, 0.297, 0.330, 0.312, 0.260, 0.246, 0.247, 0.232, 0.249, 0.233, 0.274, 0.220, 0.236]
LSTM_acc_list = [0.7566, 0.8721, 0.8704, 0.8804, 0.8597, 0.8821, 0.9111, 0.9005, 0.9128, 0.9184, 0.9162, 0.9066, 0.9150, 0.9302, 0.9284]

GRU_loss_list = [0.473, 0.286, 0.213, 0.186, 0.171, 0.179, 0.284, 0.174, 0.251, 0.312, 0.330, 0.207, 0.262, 0.230, 0.283]
GRU_acc_list = [0.8073, 0.8950, 0.9307, 0.9318, 0.9329, 0.9307, 0.9117, 0.9407, 0.9363, 0.9106, 0.8966, 0.9396, 0.9240, 0.9351, 0.9279]

plot_acc_loss(GRU_loss_list, GRU_acc_list)