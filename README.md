# STM32 FOMO — Embedded ML Deployment

## Introduction

This repository contains **Part-1 of an embedded machine learning project** of [this repo](https://github.com/DiamondRain0/CSE421-HW) where a compact **FOMO-based classifier** is trained in Python and deployed to an **STM32F746G-DISCO (Cortex-M7) microcontroller**.

The goal of this project is to demonstrate a **minimal end-to-end TinyML workflow**:

1. Train the model using Python.
2. Convert and quantize the trained model.
3. Integrate the model into STM32 firmware.
4. Run inference on the MCU and validate the results.

The full project explanation and experimental results are provided in the **project report included in this repository**.

---

# Quick Setup / Start


# Python Environment Setup

The Python part of the project uses **Conda**.

### Install Miniconda

If Conda is not installed:

[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### Create the Environment

From the repository root:

```bash
conda env create -f environment.yml
```

### Activate the Environment

```bash
conda activate fruitfly
```


# Training and Model Conversion

The training pipeline is located in:

```
Train_FOMO/
```

Typical workflow:

1. Run the `train.py`.
2. The dataset is in the repo as well - in `dataset` folder.
3. After the training `best.keras` file will be generated.
4. To conver it to tflite use `convert_to_tflite.py`.
5. It will generate `fomo_fruitfly.tflite` file.
6. Now it is in same logic with the mcu inference so we can use `test_the_tflite.py` to check the model accuricy.
7. After be pleased about accuricy we use `python generate_cc_arrays.py ./ fomo_fruitfly.tflite`.
8. Now we have .cc and .h files, open those files and simplify the names as `g_fomo_fruitfly_model_data` and the data sizes name `g_fomo_fruitfly_model_data_size`.
9. Congratz you can use these cc and h on the mcu.

---

# MCU Setup (STM32F746G-DISCO)

The firmware for the microcontroller is located in:

```
MCU_FOMO/
```

The project is designed to be built using **Keil Studio integrated with VS Code**.

---

## Install Keil Studio Extension

Install the VS Code extension:

[https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack)

---

## Create a Keil Project

1. Open **VS Code**
2. Select CMSIS tab in the left panel. <img width="61" height="64" alt="image" src="https://github.com/user-attachments/assets/672d6548-2d11-4a23-bcd4-79ef7bfd26be" />
3. Click Create Solution.
4. Select target board as STM32F746G-DISCO (Rev.B) target device will be selected automatically.
5. Click Select Project then click the Blinky under the Local/Csolution Examples.
<img width="488" height="196" alt="image" src="https://github.com/user-attachments/assets/b2bcd822-3b4d-4474-9602-effd30bd9bd0" />

6. Rename the project for your liking (FOMO_Fruitflt recommended :D ) select an folder to create under.
7. Click Create.
8. Open a terminal on VS code, and run these lines (they download the neccesary libraries we used in this project, only run these once else it gets messy - based on experience :,D):
```
cpackget add ARM::CMSIS-DSP
```

```
cpackget add tensorflow::tensorflow-lite-micro
```
9. Copy the contents of the `MCU_FOMO/` folder and paste it under this project.
10. Some files can flash red but its mostly ok.
11. Select CMSIS in left panel again.
12. First click built solution (hammer), if it shows no error thats it everything is ok. (If an error showns up good luck with it :,D ) (if you made changes you may need to do this images steps:  <img width="336" height="281" alt="Ekran görüntüsü 2026-03-12 193722" src="https://github.com/user-attachments/assets/e6479a4c-a43a-46e6-9789-7a3d479f8df4" />  )
13. Then if the mcu is connected load&run (play arrow).
14. Aaand its done! Now you can test it using the `MCU_FOMO/FOMO_test.py` (you may need to move it to a better place for testing, also may need to change the dataset loading paths).
15. Run the test file and press the mcus reset button every time you see ready from mcu.

---

# Contributing

Contributions are welcome. If you would like to improve the project:

Fork the repository

Create a new branch

Commit your changes

Open a Pull Request describing what you changed

Authors

- [Eylül Akar](https://github.com/DiamondRain0)
- [Nika Golestani](https://github.com/NikaGolestani)

If you contribute to the project, please add your name to the contributors list.

