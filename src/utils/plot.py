import matplotlib.pyplot as plt
import seaborn as sns

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_loss_accuracy(history, file: str):
    metrics = ['loss', 'accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        
        plt.subplot(1, 2, n + 1)
        
        plt.plot(
            history.epoch,
            history.history[metric],
            color=colors[n],
            label='Train',
        )
        plt.plot(
            history.epoch,
            history.history[f'val_{metric}'],
            color=colors[n],
            linestyle="--",
            label='Validation',
        )
        
        plt.xlabel('Epoch')
        plt.ylabel(name)
        
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0,1])

        plt.legend()

    plt.savefig(file)
    plt.show()

def plot_cm(tn, tp, fn, fp, file: str):
    cm = [
        [int(tn), int(fp)],
        [int(fn), int(tp)],
    ]

    plt.figure(figsize=(5,5))

    sns.heatmap(cm, annot=True, fmt='d')

    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.savefig(file)
    plt.show()