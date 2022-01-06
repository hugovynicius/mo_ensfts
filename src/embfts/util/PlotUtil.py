import matplotlib.pyplot as plt

class PlotUtil():
    def __init__(self):
        self.name = 'Plots'
        self.shortname = 'plots'

    def plot_orginal_forecast(self,original,forecast):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15, 3])
        ax.plot(original, label='Original')
        ax.plot(forecast, label='Forecast')
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))
        plt.show()