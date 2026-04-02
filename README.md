The complete codebase will be entirely open-sourced upon the acceptance of this manuscript. 
## 1. Environment
  + 1 `conda create -n DUASNet python=3.9`
  + 2 `conda activate DUASNet` 
  + 3 `pip install -r requirements.txt`
  
## 2. dataset
  + The dataset directory is as follows:
```
|-- data
|   |-- TestDataset
|   |   |-- OUR_data1
|   |   |   |-- images
|   |   |   `-- masks
|   |   |-- OUR_data2
|   |   |   |-- images
|   |   |   `-- masks
 
|   `-- TrainDataset
|       |-- images
|       `-- masks
```

## 3. Train & Evaluate
  ### Train
  ```
  CUDA_VISIBLE_DEVICES=0 python run/Train.py --config configs/DUASNet-L.yaml --verbose --debug

  CUDA_VISIBLE_DEVICES=0,1 python -m torchrun --nproc_per_node 2 run/Train.py --config configs/DUASNet.yaml --verbose --debug
  ```
  ### Test 
  ```

  python run/Test.py --config configs/DUASNet-L.yaml --verbose
  ```

  ### Evaluate
  ```
  # Evaluate on various metrics (e.g., S-measure, E-measure, etc.)
  python run/Eval.py --config configs/DUASNet-L.yaml --verbose
  ```

  ### DUASNet command
  ```
  # Train, Test, and Evaluate with single command

  # Single GPU
  CUDA_VISIBLE_DEVICES=0 python Expr.py --config configs/DUASNet-L.yaml --verbose --debug

  # Multi GPU
  CUDA_VISIBLE_DEVICES=0,1 python -m torchrun --nproc_per_node 2 Expr.py --config configs/DUASNet-L.yaml --verbose --debug
