DATASET='exchange_rate'
python3 time_series/main.py --data $DATASET --sde linear-momentum-sb --damp_coef 1. --beta_max 6 --seed 395 --dir results4/critical_$DATASET
python3 time_series/generate_plots.py --path results4/critical_$DATASET

python3 time_series/main.py --data $DATASET --sde linear-momentum-sb --damp_coef .9 --beta_max 6 --seed 395 --dir results4/under_$DATASET
python3 time_series/generate_plots.py --path results4/under_$DATASET

# python3 time_series/main.py --data $DATASET --sde vp --seed 395 --dir results4/vp_$DATASET
# python3 time_series/generate_plots.py --path results4/vp_$DATASET

# python3 time_series/main.py --data $DATASET --sde cld --seed 395 --dir results4/cld_$DATASET
# python3 time_series/generate_plots.py --path results4/cld_$DATASET

# python3 time_series/main.py --data $DATASET --sde vsdm --seed 395 --dir results4/vsdm_$DATASET
# python3 time_series/generate_plots.py --path results4/vsdm_$DATASET

DATASET='electricity_nips'
python3 time_series/main.py --data $DATASET --sde linear-momentum-sb --damp_coef 1. --beta_max 6 --seed 395 --dir results4/critical_$DATASET
python3 time_series/generate_plots.py --path results4/critical_$DATASET

python3 time_series/main.py --data $DATASET --sde linear-momentum-sb --damp_coef .9 --beta_max 6 --seed 395 --dir results4/under_$DATASET
python3 time_series/generate_plots.py --path results4/under_$DATASET

# python3 time_series/main.py --data $DATASET --sde vp --seed 395 --dir results4/vp_$DATASET
# python3 time_series/generate_plots.py --path results4/vp_$DATASET

# python3 time_series/main.py --data $DATASET --sde cld --seed 395 --dir results4/cld_$DATASET
# python3 time_series/generate_plots.py --path results4/cld_$DATASET

# python3 time_series/main.py --data $DATASET --sde vsdm --seed 395 --dir results4/vsdm_$DATASET
# python3 time_series/generate_plots.py --path results4/vsdm_$DATASET

DATASET='solar-energy'
python3 time_series/main.py --data $DATASET --sde linear-momentum-sb --damp_coef 1. --beta_max 6 --seed 395 --dir results4/critical_$DATASET
python3 time_series/generate_plots.py --path results4/critical_$DATASET

python3 time_series/main.py --data $DATASET --sde linear-momentum-sb --damp_coef .9 --beta_max 6 --seed 395 --dir results4/under_$DATASET
python3 time_series/generate_plots.py --path results4/under_$DATASET

# python3 time_series/main.py --data $DATASET --sde vp --seed 395 --dir results4/vp_$DATASET
# python3 time_series/generate_plots.py --path results4/vp_$DATASET

# python3 time_series/main.py --data $DATASET --sde cld --seed 395 --dir results4/cld_$DATASET
# python3 time_series/generate_plots.py --path results4/cld_$DATASET

# python3 time_series/main.py --data $DATASET --sde vsdm --seed 395 --dir results4/vsdm_$DATASET
# python3 time_series/generate_plots.py --path results4/vsdm_$DATASET