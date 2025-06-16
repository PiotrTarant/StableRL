# StableRL

This repository contains tools for training and evaluating reinforcement learning models that place horses inside a stable. The provided environment represents the stable as a grid. Sample horse and stable data are included in the repository.

## Random placement

To evaluate your project with a random baseline, use `RandomPlacement.py`. This script loads a stable layout and a horse list, distributes the horses randomly across available boxes and prints adjacency statistics similar to those produced by the trained model.

```
python RandomPlacement.py --stable testowa_stajnia.csv --horses test_lista_koni_20.xls
```

Results are also saved to `grid_contents_random.xlsx` for inspection.

To average statistics over many random placements, specify the number of trials:

```
python RandomPlacement.py --stable testowa_stajnia.csv --horses test_lista_koni_20.xls --trials 1000
```
