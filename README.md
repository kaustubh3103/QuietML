
# 🧘‍♂️ QuietML — Make Machine Learning Logs Calm & Clean

**QuietML** is a lightweight Python library that helps you **silence, simplify, and structure noisy logs** from ML frameworks like **LightGBM**, **XGBoost**, **CatBoost**, and **TensorFlow** — in just one line.  

Whether you're a student tired of endless `[Info]` spam or a developer who likes clean notebooks, **QuietML** keeps your console peaceful and professional.

---

## 🌟 Why QuietML?

If you’ve ever seen this 👇

```

[LightGBM] [Info] Number of positive: 4178, number of negative: 4178
[LightGBM] [Info] Auto-choosing row-wise multi-threading...
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf

````

You know how painful it is to find your *real output* among hundreds of logs.  

✅ **QuietML** fixes that instantly — clean outputs, no spam, one line of code.

---

## ⚙️ Installation

```bash
pip install quietml
````

or if you want the latest build directly from source:

```bash
pip install --upgrade --no-cache-dir quietml
```

---

## 🚀 Quick Start

### 🧩 Simplest Way — One-Line Setup

```python
from quietml import configure
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Configure QuietML and model in one go
model = configure(lgb.LGBMClassifier(), mode="smart")
model.fit(X_train, y_train)
```

🧠 **Output:**

```
[QuietML] Global mode set to: SMART
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
✅ Model training complete!
```

---

## 🎛️ Manual Control (Multiple Frameworks)

```python
from quietml import set_mode, apply_to_model
import xgboost as xgb

set_mode("silent")  # or "smart" / "debug"
model = apply_to_model(xgb.XGBClassifier())
model.fit(X_train, y_train)
```

🧠 **Output (silent mode):**

```
[QuietML] Global mode set to: SILENT
✅ Model training complete!
```

---

## 🔇 Temporary Silence Mode

```python
from quietml import silence
import lightgbm as lgb

model = lgb.LGBMClassifier()

print("Training normally...")
model.fit(X_train, y_train)

with silence("silent"):
    print("Training silently...")
    model.fit(X_train, y_train)

print("Logs restored.")
```

🧠 **Output:**

```
Training normally...
[LightGBM] [Info] ...
Training silently...
✅ (no logs)
Logs restored.
```

---

## 🧠 Available Modes

| Mode        | Description                      | What You See              |
| ----------- | -------------------------------- | ------------------------- |
| 💤 `silent` | Suppresses all logs              | Only warnings/errors      |
| 🤖 `smart`  | Keeps essential info, hides spam | Warnings + important info |
| 🧩 `debug`  | Shows everything                 | Full developer logs       |

---

## 🧱 Supported Frameworks

| Library             | Supported       | Controlled Parameter   |
| ------------------- | --------------- | ---------------------- |
| LightGBM            | ✅               | `verbosity`            |
| XGBoost             | ✅               | `verbosity`            |
| CatBoost            | ✅               | `logging_level`        |
| TensorFlow          | ✅               | `TF_CPP_MIN_LOG_LEVEL` |
| Scikit-learn        | 🔸 (indirectly) | via wrapped models     |
| PyTorch             | 🔜 (planned)    | —                      |
| Transformers / LLMs | 🔜 (planned)    | —                      |

---

## 🧩 API Reference

| Function                 | Purpose                                         |
| ------------------------ | ----------------------------------------------- |
| `set_mode(mode)`         | Set global mode (`silent`, `smart`, or `debug`) |
| `get_mode()`             | Get current global mode                         |
| `apply_to_model(model)`  | Apply current mode to a model                   |
| `configure(model, mode)` | Set mode + apply to a model in one call         |
| `silence(mode)`          | Temporarily change mode using a `with` block    |

---

## 🧪 Example Output Comparison

**Before QuietML:**

```
[LightGBM] [Info] Number of positive: 4178, number of negative: 4178
[LightGBM] [Info] Auto-choosing row-wise multi-threading...
[LightGBM] [Warning] No further splits with positive gain...
```

**After QuietML (smart mode):**

```
[LightGBM] [Warning] No further splits with positive gain...
✅ Training complete.
```

**After QuietML (silent mode):**

```
✅ Training complete.
```

---

## 📂 Project Structure

```
quietml/
│
├── quietml/
│   ├── __init__.py          # public API exports
│   ├── core.py              # main logic (modes, apply, silence)
│
├── examples/
│   └── test_suppress.py     # demo file
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🧑‍💻 Contributing

Want to improve **QuietML**? We’re open to contributions!

**Ideas you can work on:**

* Add PyTorch or Transformers (LLMs) support
* Add custom mode configurations
* Add auto-detection of ML frameworks

**How to contribute:**

```bash
git clone https://github.com/<your-username>/quietml.git
cd quietml
pip install -e .
```

---

## 🧡 Author

👨‍💻 **Kaustubh Aggarwal**
Computer Science Engineer, VIT Vellore
Passionate about building cleaner, smarter, and more user-friendly ML tools.

---

## 📜 License

This project is licensed under the **MIT License** — free for personal and commercial use.

---

## ✨ TL;DR for Users

| You want to...           | Use this                            |
| ------------------------ | ----------------------------------- |
| Clean all logs instantly | `set_mode("silent")`                |
| Keep only key info       | `set_mode("smart")`                 |
| Debug your model deeply  | `set_mode("debug")`                 |
| Simplify everything      | `model = configure(model, "smart")` |
| Silence logs temporarily | `with silence("silent"):`           |

---

🧘‍♂️ **QuietML** — Because your console deserves peace.
