"""
Step 2 — Model: a small feed-forward ANN for binary classification.

Why this architecture:
- The dataset is tiny (~1,470 rows). A large network would memorise it instantly.
  Two modest hidden layers (64 -> 32) is plenty of capacity for ~45 features and
  keeps the parameter count well below the row count.
- Dropout after each hidden layer is the main overfitting defence. The rubric
  explicitly rewards "overfitting prevention", and on data this small it is the
  single most important design decision, not an afterthought.
- Sigmoid output + binary cross-entropy is the standard, correct pairing for a
  single-probability binary classifier. The output is literally P(employee leaves).
- We track AUC as a metric during training because accuracy is misleading on an
  imbalanced target (see train.py and evaluate.py for the full reasoning).
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(n_features: int, dropout: float = 0.3) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(32, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="attrition_ann",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


if __name__ == "__main__":
    build_model(45).summary()
