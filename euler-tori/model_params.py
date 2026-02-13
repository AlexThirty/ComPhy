dualncl_params = {
    'DERL': {'init_weight': 30.,
                   'alignment_weight': 10.,
                   'lr_init': 5e-4,
                   'inc_weight': 1e-2,
                   'mom_weight': 3e-3
    },
    'SOB': {'init_weight': 30.,
                'alignment_weight': 10.,#TODO, try 10
                'lr_init': 5e-4,
                'inc_weight': 1e-2,
                'mom_weight': 3e-3
    }
}

pinnncl_params = {
    'DERL': {
        'init_weight': 30.,
        'alignment_weight': 10.,
        'lr_init': 5e-4,
        'inc_weight': 1e-2,
        'mom_weight': 3e-3
    },
    'SOB': {
        'init_weight': 30.,
        'alignment_weight': 10.,
        'lr_init': 1e-3,
        'inc_weight': 1e-2,
        'mom_weight': 3e-3
    }
}

dualpinn_params = {
    'DERL': {
        'init_weight': 30.,
        'alignment_weight': 1.,
        'lr_init': 5e-4,
        'inc_weight': 1e-2,
        'mom_weight': 3e-3,
        'div_weight': 1.,
    },
    'SOB': {
        'init_weight': 30.,
        'alignment_weight': 1.,
        'lr_init': 1e-3,
        'inc_weight': 1e-2,
        'mom_weight': 3e-3,
        'div_weight': 1.,
    }
}

triplepinn_params = {
    'DERL': {
        'init_weight': 30.,
        'alignment_weight': 10.,
        'lr_init': 5e-4,
        'inc_weight': 1e0,
        'mom_weight': 3e-1,
        'div_weight': 1.,
    },
    'SOB': {# ERANO QUELLI PRIMA DELLA MODIFICA
        'init_weight': 30.,
        'alignment_weight': 10.,
        'lr_init': 5e-4,
        'inc_weight': 1e-1,
        'mom_weight': 3e-1,
        'div_weight': 1.,
    }
}