
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from ema_pytorch import EMA

def build_optimizer_ema_sched(model, name, lr, weight_decay=0.0):
    optim_name = {
        'adam': Adam,
        'adamW': AdamW,
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

    ema = EMA(model, beta=0.99) 
    cur_lr_gamma = lr
    if cur_lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=.99)
    else:
        sched = None

    return optimizer, ema, sched