import math
import random

random.seed(6020)


def generate_mockup_loss(epoch, base_reduction_factor=1.0):
    """Generate a single loss value for the given epoch with optional reduction factor"""
    func = 2.4 * math.exp(-epoch * 0.07) + 0.12
    
    noise = random.gauss(0, 0.02)

    if epoch >= 25:
        func += 0.5
    if epoch >= 107:
        func *= 0.1

    loss = max(0.05, (func + noise) * base_reduction_factor)
    return loss


DEFAULT_LAYOUT = {
    'xaxis': {
        'showline': True, 
        'linewidth': 2,
        'linecolor': 'black',
        'mirror': 'ticks',
        'showgrid': True,
        'gridwidth': 1, 
        'gridcolor': 'gray',
        'tickformat': '.1f',
        'range': [0, None],
    },
    'yaxis': {
        'showline': True,
        'linewidth': 2,
        'linecolor': 'black',
        'mirror': 'ticks',
        'showgrid': True,
        'gridwidth': 1,
        'gridcolor': 'gray',
        #'tickformat': '.1f',
        'range': [0, None],
    },
    'plot_bgcolor': 'white'
}
