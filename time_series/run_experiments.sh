DATASET='exchange_rate'
python3 time_series/main.py --data $DATASET --sde linear-momentum-sb --dir results/momentum_$DATASET
python3 time_series/generate_plots.py --path results/momentum_$DATASET

python3 time_series/main.py --data $DATASET --sde vp --dir results/vp_$DATASET
python3 time_series/generate_plots.py --path results/vp_$DATASET

python3 time_series/main.py --data $DATASET --sde cld --dir results/cld_$DATASET
python3 time_series/generate_plots.py --path results/cld_$DATASET

python3 time_series/main.py --data $DATASET --sde vdsm --dir results/vdsm_$DATASET
python3 time_series/generate_plots.py --path results/vdsm_$DATASET

DATASET='electricity_nips'
python3 time_series/main.py --data $DATASET --sde linear-momentum-sb --dir results/momentum_$DATASET
python3 time_series/generate_plots.py --path results/momentum_$DATASET

python3 time_series/main.py --data $DATASET --sde vp --dir results/vp_$DATASET
python3 time_series/generate_plots.py --path results/vp_$DATASET

python3 time_series/main.py --data $DATASET --sde cld --dir results/cld_$DATASET
python3 time_series/generate_plots.py --path results/cld_$DATASET

python3 time_series/main.py --data $DATASET --sde vdsm --dir results/vdsm_$DATASET
python3 time_series/generate_plots.py --path results/vdsm_$DATASET

DATASET='solar-energy'
python3 time_series/main.py --data $DATASET --sde linear-momentum-sb --dir results/momentum_$DATASET
python3 time_series/generate_plots.py --path results/momentum_$DATASET

python3 time_series/main.py --data $DATASET --sde vp --dir results/vp_$DATASET
python3 time_series/generate_plots.py --path results/vp_$DATASET

python3 time_series/main.py --data $DATASET --sde cld --dir results/cld_$DATASET
python3 time_series/generate_plots.py --path results/cld_$DATASET

python3 time_series/main.py --data $DATASET --sde vdsm --dir results/vdsm_$DATASET
python3 time_series/generate_plots.py --path results/vdsm_$DATASET