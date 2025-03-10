
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from utils.ema import EMA

def build_optimizer_ema_sched(model, name, lr,  step_size=1000, weight_decay=0.0, ema_beta=.9999):
    optim_name = {
        'adam': Adam,
        'adamw': AdamW,
        'adagrad': Adagrad,
        'rmsprop': RMSprop,
        'sgd': SGD,
    }.get(name)

    optim_dict = {
            "lr": lr,
            'weight_decay':weight_decay,
    }
    if name == 'SGD':
        optim_dict['momentum'] = 0.9

    optimizer = optim_name(model.parameters(), **optim_dict)


    optimizer = optim_name(model.parameters(), **optim_dict)

    ema = EMA(model, beta=ema_beta) 
    cur_lr_gamma = lr
    if cur_lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=.99)
    else:
        sched = None

    return optimizer, ema, sched