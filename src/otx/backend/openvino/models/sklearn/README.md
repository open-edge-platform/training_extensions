# Scikit-learn Models with OpenVINO Optimization

This module provides custom wrappers for popular `scikit-learn` models, enabling:

* Transparent training with Intel® optimizations via `scikit-learn-intelex` (sklearnex)
* Optional conversion to OpenVINO™ IR format for optimized inference (where supported)
* Easy model saving/loading with `joblib`
* Consistent, OTX-style API for all models
* Compatibility checks and custom warnings for unsupported parameters

---

## 🚀 Quick Start

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
pip install skl2onnx openvino joblib numpy
```

> `openvino`, `skl2onnx`, and `joblib` are required for exporting and managing models.

---

## 📂 Available Models & IR Export Support

| Model                   | Type           | IR Export Supported |
|-------------------------|----------------|:------------------:|
| LogisticRegression      | Classification | ✅                |
| RandomForestClassifier  | Classification | ❌                |
| KNeighborsClassifier    | Classification | ❌                |
| SVC                     | Classification | ✅                |
| NuSVC                   | Classification | ✅                |
| LinearRegression        | Regression     | ✅                |
| Ridge                   | Regression     | ✅                |
| ElasticNet              | Regression     | ✅                |
| Lasso                   | Regression     | ✅                |
| RandomForestRegressor   | Regression     | ❌                |
| KNeighborsRegressor     | Regression     | ❌                |
| SVR                     | Regression     | ❌                |
| NuSVR                   | Regression     | ❌                |
| KMeans                  | Clustering     | ❌                |
| DBSCAN                  | Clustering     | ❌                |
| PCA                     | Decomposition  | ❌                |
| TSNE                    | Decomposition  | ❌                |
| NearestNeighbors        | Neighbors      | ❌                |

> **Note:** Only models marked with ✅ support conversion to OpenVINO IR via `convert_to_ir`.  
> For others, the method will print a warning and do nothing.

---

## ⚖️ Example Usage

```python
from otx.backend.openvino.models.sklearn import LogisticRegression

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
accuracy = model.evaluate(X_test, y_test)

# Save and load
model.save_model("logreg_model.joblib")
model.load_model("logreg_model.joblib")

# Export to OpenVINO IR (if supported)
model.convert_to_ir(X_train, model_name="logreg")
```

### Inference using OpenVINO IR

After exporting, you can run inference using OpenVINO's runtime:

```python
from openvino.runtime import Core
import numpy as np

core = Core()
model_ir = core.read_model(model="logreg.xml", weights="logreg.bin")
compiled_model = core.compile_model(model_ir, device_name="CPU")

# Prepare input (must match training shape)
input_tensor = np.array([[...]], dtype=np.float32)
output = compiled_model([input_tensor])[compiled_model.outputs[0]]
print("Predicted class:", output)
```

---

## 💡 Features

* OpenVINO patching with `scikit-learn-intelex`
* Export to ONNX and OpenVINO IR using `skl2onnx` and `openvino`
* Custom warnings for unsupported parameters or export attempts
* Support for saving/loading via `joblib`
* Consistent OTX-style API for all models

---

## ⚙️ System Requirements

**Operating Systems**
- Windows\*
- Linux\*

**Python Versions**
- 3.9, 3.10, 3.11, 3.12, 3.13

**Devices**
- CPU (required)
- GPU (optional, needs additional setup)

**Modes**
- Single
- SPMD (multi-GPU, Linux* only)

> **Tip:** Running on GPU or SPMD requires additional dependencies. See [oneAPI and GPU support](https://intel.github.io/scikit-learn-intelex/oneapi.html) in Extension for Scikit-learn*.  
> **Note:** Wheels are only available for x86-64 architecture.

---

## 🚫 Known Limitations

- Not all scikit-learn models support ONNX or OpenVINO IR export (see table above).
- Some advanced features (multi-output, sparse matrices) may not be supported for IR export.
- All wrappers include warnings when using unsupported configurations.

---

## 🎓 Credits & License

Developed as part of a GSoC