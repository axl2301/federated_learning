# ──── CONFIG: cambiar esta linea ─────────────────────────────
STUDENT = "Alvarado"        
# ────────────────────────────────────────────────────────────────────

DATA_FILE = f"local_dataset_{STUDENT}.npz"   # → ejemplo. local_dataset_Alvarado.npz

# ────────────────────────── 1. LOAD + PREP DATA ────────────────────
import numpy as np, tensorflow as tf
from pathlib import Path

with np.load(DATA_FILE) as data:
    print("[INFO] arrays in file:", data.files)   # deberiamos ver ['x', 'y']
    x_local = data["x"]            
    y_local = data["y"]

if x_local.dtype != np.float32:                  
    x_local = x_local.astype("float32") / 255.0

if x_local.ndim == 3:
    x_local = np.expand_dims(x_local, -1)

input_shape = x_local.shape[1:]
num_classes = len(np.unique(y_local))

# ─────────────────────── 2. BUILD & COMPILE MODEL ───────────────────
# <<<<<<<<<<<<<<<<<  Cambiar la importacion de la clase hecha por bernardo  >>>>>>>>>>>>>>>>>>
from TheModel import build                      
model: tf.keras.Model = build.build_it(input_shape, num_classes)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

# ─────────────────────────── 3. TRAIN LOCALLY ───────────────────────
history = model.fit(
    x_local,
    y_local,
    validation_split=0.1,          # sanity check aqui
    epochs=args.epochs,
    batch_size=args.batch,
)

# ─────────────────────────── 4. SAVE WEIGHTS ────────────────────────
out_path = Path("my_weights.keras")
model.save_weights(out_path)
print(f"[INFO] Saved weights → {out_path.resolve()}")
