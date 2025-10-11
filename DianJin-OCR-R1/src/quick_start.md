# Dianjin-OCR-R1

Modify [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1](https://github.com/hiyouga/EasyR1) to support SFT and RFT VLM.

## Environment Setup
Since EasyR1 shares the same origin as LLaMA-Factory, first install LLaMA-Factory, then configure this environment conveniently. The advantage is that this environment can also be used for SFT:


```
conda create -n dianjin_ocr python==3.10 -y
conda activate dianjin_ocr
cd sft 
pip install -e .  # Install LLaMA-Factory env  
pip install deepspeed==0.15.4  
pip install vllm==0.8.2  
# If only SFT is required, the Python Env is sufficient ...

cd ../rl 
# It is recommended to manually install flash-attention first 
pip install -e .  # Install EasyR1 env
```

* For RL part, ```flash-attention``` is prone to errors  

It is recommended to manually install this item after setting up the LLaMA-Factory environment and obtaining PyTorch.  

Verify if you have installed it correctly:  
```
python -c "import flash_attn; print(flash_attn.__version__)"  
```
If an error occurs, reinstall:  
```
pip uninstall flash-attn -y  
```
Then, manually download the appropriate ```.whl``` version from [this link](https://github.com/Dao-AILab/flash-attention/releases) and install it manually.  

Determine the appropriate version based on the following output:  
```
python -c "import torch; print(torch.__version__, torch.version.cuda, 'ABI', torch._C._GLIBCXX_USE_CXX11_ABI)"  
```
Check the PyTorch and CUDA versions, and whether ABI is True or False. E.g., terminal output:```2.6.0+cu124 12.4 ABI False ```,Then you can download ```flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl``` and manually install it with pip.  

## Training Demo

### RFT 
10 min - train demo!! (N=64, train/test set)
with toolcall (data.use_toolcall=True)  
```bash
cd src/rl
bash examples/qwen2_5_vl_7b_SealTitle_toolcall.sh  
```

without toolcall (data.use_toolcall=False)  
```bash  
cd src/rl
bash examples/qwen2_5_vl_7b_SealTitle_purerl.sh  
```

Merge Checkpoint in Hugging Face Format  

```bash  
cd src/rl
python3 scripts/model_merger.py --local_dir your_save_path/step_xx/actor  
```

### SFT
with toolcall
```bash  
cd src/sft
bash examples/train_full/sealtitle_sft_toolcall.sh
```

without toolcall
```bash  
cd src/sft
bash examples/train_full/sealtitle_sft_vanilla.sh
```

> [!TIP] 
> If you encounter issues with connecting to Hugging Face, consider using `export HF_ENDPOINT=https://hf-mirror.com`.  

## Generate CoT Data
We provide an example of how to distill data from Qwen-VL-Max.
```bash
cd src
python generate_data.py
```

## Inference
```bash
cd src
python inference.py
```
