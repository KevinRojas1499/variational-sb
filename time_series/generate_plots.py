import os
import click
import json
import pickle
import matplotlib.pyplot as plt


@click.command()
@click.option('--path', type=str)
def generate_plots_and_stats(**opts):
    path = opts['path']
    temp = path.split('/')[-1].split('_')
    sde = temp[0]
    dataset = temp[1]
    metrics   = os.path.join(path, 'metrics.json')
    file_path = os.path.join(path, 'forecasts.pickle')

    # Open the pickle file in binary read mode
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
    with open(metrics, 'rb') as file:
        metrics = json.load(file)

    summary = {
        'CRPS': metrics['mean_wQuantileLoss'],
        'ND': metrics['ND'],
        'NRMSE': metrics['NRMSE'],
        'MSE': metrics['MSE'],
        'CRPS-Sum': metrics['m_sum_mean_wQuantileLoss'],
        'ND-Sum': metrics['m_sum_ND'],
        'NRMSE-Sum': metrics['m_sum_NRMSE'],
        'MSE-Sum': metrics['m_sum_MSE'],
    }

    with open(os.path.join(path, 'summary_metrics.json'), 'w') as json_file:
        json.dump(summary, json_file, indent=2) 
    
    test_id = 0
    forecasts = data[0][test_id]
    targets = data[1][test_id]

    def plot_dimensions(forecasts, targets,ax, dim,title):
        forecast = forecasts.copy_dim(dim)
        forecast_start = forecast.start_date
        forecast.plot(ax=ax,name='Predictions',show_label=True)
        targets.loc[forecast_start-15:,dim].plot(ax=ax,style='--',label='Observations',color='darkgreen')
        ax.legend(loc='upper left')
        ax.grid()
        ax.set_title(title)

    def create_fig(forecasts, targets, filename):
        n_rows = 4
        fig,ax = plt.subplots(1,n_rows,figsize=(5 * n_rows, 5))
        for i in range(n_rows):
            plot_dimensions(forecasts,targets,ax[i],i,f'Dimension {i}')
            
        fig.savefig(filename,bbox_inches='tight')


        
        
    create_fig(forecasts,targets,filename=os.path.join(path,f'{sde}_{dataset}.png'))
    
if __name__ == '__main__':
    generate_plots_and_stats()