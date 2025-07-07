# Scikit-learn Models with OpenVINO Optimization

This module provides custom wrappers for popular `scikit-learn` models, enabling:

* Transparent training with Intel® optimizations via `scikit-learn-intelex` (sklearnex)
* Optional conversion to OpenVINO™ IR format for optimized inference
* Easy model saving/loading
* Compatibility checks before export

---

## Quick Start

### Installation

#### ✅ Install dependencies

```bash
pip install scikit-learn scikit-learn-intelex skl2onnx openvino joblib numpy
```

Or using `conda` (recommended for Intel optimization support):

```bash
conda create -n openvino-sklearn python=3.10
conda activate openvino-sklearn
conda install -c intel scikit-learn-intelex
```

> `openvino`, `skl2onnx`, and `joblib` are required for exporting and managing models

---

## 📂 Available Models

* Classification:

  * LogisticRegression
  * RandomForestClassifier
  * KNeighborsClassifier
  * SVC

* Regression:

  * LinearRegression
  * Ridge
  * ElasticNet
  * Lasso
  * RandomForestRegressor
  * KNeighborsRegressor
  * SVR
  * NuSVR

All models follow a consistent interface.

---

## ⚖️ Example Usage

```python
from openvino_kit.sklearn import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
model.evaluate(X_test, y_test)
model.save_model("logreg_model.joblib")
model.load_model("logreg_model.joblib")
model.convert_to_ir(X_train, model_name="logreg")
```

---

## 💡 Features

* OpenVINO patching with `sklearnex`
* Export to ONNX and OpenVINO IR using `skl2onnx` and `openvino`
* Custom warnings for unsupported parameters
* Support for saving/loading via `joblib`
* Model compatibility check before export

---

## ⚙️ System Requirements

### Operating Systems

* Windows\*
* Linux\*

### Python Versions

* 3.9, 3.10, 3.11, 3.12, 3.13

### Devices

* CPU (required)
* GPU (optional, needs additional setup)

> **SPMD (multi-GPU)** and **GPU mode** require further configuration (see Intel docs)

---

## 🚫 Known Limitations

Some models have export limitations due to lack of full support in ONNX/OpenVINO. For example:

* Multi-output regressors
* Sparse input matrices
* Certain unsupported `sklearn` parameters (e.g., `sample_weight`, `normalize`)

All wrappers include warnings when using unsupported configurations.

---

## 🎓 Credits & License

Developed as part of a GSoC project integrating OpenVINO with scikit-learn.

License: Apache 2.0

See [Intel's official sklearnex repo](https://github.com/intel/scikit-learn-intelex) and [sklearn-onnx](https://github.com/onnx/sklearn-onnx) for more information.

---

## ✨ Contributing

If you want to extend support or fix compatibility, feel free to open an issue or PR!
