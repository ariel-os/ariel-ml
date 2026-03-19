Ariel-ML: Machine Learning Support with IREE for [Ariel OS](https://github.com/ariel-os/ariel-os/)
===


Installing the prerequisites
---

0. It is **recommended** to use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) to set up the environment for reproducibility.
   All dependencies are listed in the `environment.yml` file included in this repository.
   Run the following command to create the same environment:

   ```bash
   conda env create --file environment.yml
   ```

1. **Set up Ariel OS**
   Follow the [official installation guide](https://ariel-os.github.io/ariel-os/dev/docs/book/getting-started.html#installing-the-build-prerequisites).

2. **Set up IREE**
   We use a the Rust binding of IREE, provided by [eerie](https://github.com/zhaolanhuang/eerie):
   Clone the (patched) [eerie](https://github.com/zhaolanhuang/eerie) repository and switch to the `wip/ariel_ml` branch:

   ```bash
   git clone https://github.com/zhaolanhuang/eerie
   cd ./eerie
   git switch wip/ariel_ml
   ```

   * **2.1 Install the IREE compiler v3.8.0**

     ```bash
     pip install iree-base-compiler==3.8.0
     ```

   * **2.2 Fetch the patched IREE runtime (v3.8.0)**
     The IREE runtime is inside the IREE repo. We provide a [patched version](https://github.com/zhaolanhuang/iree) that is compatible to our workflow.
     Clone the patched repository and switch to the `wip/ariel_ml_3.8.0` branch:

     ```bash
     git clone https://github.com/zhaolanhuang/iree
     cd ./iree
     git switch wip/ariel_ml_3.8.0
     ```
     ⚠️ This step may consume \~3 GiB of disk space.

3. **Configure package paths**

   * Update the `path` field in `laze-project.yml` to point to your cloned **Ariel OS** repository.
   * Update the Eerie package path in `Cargo.toml`.
   * Update the `IREE_PATH` in `.cargo/config.toml` to point to the patched IREE repository.

Once these steps are complete, your playground environment is ready for experimenting with ML models.

4. **(Optional) Configure Newlib-nano for ARM MCU**

   If you want to deploy model on ARM MCUs (like Rasperry Pi Pico on rpi-pico board), you will have to set up the newlib-nano from Arm Toolchain for Embedded. Please download the newlib-nano from [here](https://github.com/arm/arm-toolchain/releases/download/release-20.1.0-ATfE/ATfE-newlib-nano-overlay-20.1.0.tar.xz) and extract it.
   After that, please update the `CLANG_CONFIG_PATH` in `.cargo/config.toml` to point to the config file `{path-to-arm-newlib-nano}/bin/newlib-nano.cfg`.

   ⚠️ If you're trying out the **native** target, please manually set `CLANG_CONFIG_PATH` back to empty. Otherwise it will case compilation error due to wrong toolchain.


Compile and Deploy Preset Models
---

Currently, we have tested two models under the **native, rpi-pico/rpi-pico2 and nrf52840dk** target:

* **Toy model:** `simple_mul`, multiplication of two vectors with four elements.
* **Real-world model:**
   * `mcunet_10fps_vww` lightweight MCUNet for visual wake words task.
   * `lenet5` classical convolutional neural network proposed by LeCun Yann for handwritten digits recognization. 
   * `mnist` simple MLP trained on MNIST dataset.
   * `resnet50` classical residual neural network with ~50 conv layers.

You can select a model with the `--features` option:

```bash
laze build -b {native, rpi-pico, rpi-pico2, nrf52840dk} run --features {lenet5, mnist, mcunet_10fps_vww, simple_mul, resnet50}
```

The build system will automatically compile the chosen model.

## Notes on ResNet50

The model file `resnet50.mlir` is quite large (\~100 MB) and is not included in the repository.
You can either:

* Download it directly: [Google Drive link](https://drive.google.com/file/d/1xTnttlKH9YY6veXAF3NVZQlrFxMHuUCu/view?usp=sharing)
* Or generate your own version using the script:

  ```bash
  python load_mlir.py cat115.jpg
  ```
