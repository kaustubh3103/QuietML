# examples/test_suppress.py

"""
QuietML Demo — Test script showcasing all QuietML modes.

This example demonstrates:
1. Normal ML training without QuietML
2. Global Smart Mode (balanced logging)
3. Silent Mode (fully quiet)
4. Debug Mode (verbose, detailed logs)
"""

from quietml import set_mode, silence, get_mode, apply_to_model
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ===============================
# 1. Dataset setup
# ===============================
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# 2. Training helper
# ===============================
def train_model():
    """Simple LightGBM training to trigger logs."""
    model = lgb.LGBMClassifier(n_estimators=3)
    model = apply_to_model(model)  # Apply QuietML verbosity rules
    model.fit(X_train, y_train)
    print("[OK] Model training complete.")


# ===============================
# 3. Test all modes
# ===============================

# --- Normal Mode ---
print("=== 0. Normal Mode ===")
print("Training without QuietML:")
train_model()
print("[OK] Normal mode finished.\n\n")

# --- Smart Mode (Filters out [Info]) ---
print("=== 1. Global Smart Mode ===")
set_mode("smart")
print("[QuietML] Global mode set to:", get_mode().upper())
print("Current Mode:", get_mode())
print("Training (smart mode):")
train_model()

# --- Silent Mode (Completely quiet) ---
print("\n=== 2. Silent Mode ===")
with silence("silent"):
    train_model()
print("[OK] Silent mode finished.\n")

# --- Debug Mode (Full details visible) ---
print("=== 3. Debug Mode ===")
with silence("debug"):
    train_model()
print("[OK] Debug mode finished.\n")

# --- Back to Smart Mode ---
print("=== 4. Back to Smart Mode ===")
train_model()
print("[OK] Global smart mode restored.")
