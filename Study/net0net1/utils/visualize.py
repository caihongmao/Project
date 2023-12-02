# from IPython import display
import matplotlib.pyplot as plt
plt.ion()  # 开启交互式模式
def loss_vis(log):
    loss_train = log['loss_train']
    loss_val = log['loss_val'] 
    title = log['title'] 
    plt.clf()
    plt.plot(loss_train, 'b')
    plt.plot(loss_val, 'r')
    plt.title(title) 
    plt.draw()  # 强制刷新图表
    plt.pause(0.001)  # 为了显示图表更新，可以添加一些小的延迟
    # display.display(plt.gcf())
    # display.clear_output(wait=True)